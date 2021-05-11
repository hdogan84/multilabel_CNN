from typing import Callable
import multiprocessing
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd

from dataset.MultiLabelAudioSet import MultiLabelAudioSet
from config.configuration import DataConfig, ScriptConfig, SystemConfig, LearningConfig
from pathlib import Path
from pytorch_lightning.metrics.utils import to_onehot


def filter_none_values(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class AmmodMultiLabelModule(LightningDataModule):
    def __init__(
        self,
        config: ScriptConfig,
        fit_transform_audio: Callable = None,
        fit_transform_image: Callable = None,
        val_transform_audio: Callable = None,
        val_transform_image: Callable = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_list_filepath: where to find csv data file
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
        
            batch_size: size of batch
        """
        super().__init__(*args, **kwargs)
        d: DataConfig = config.data
        s: SystemConfig = config.system
        l: LearningConfig = config.learning
        self.config = config

        self.train_list_filepath = (
            Path(d.train_list_filepath) if d.train_list_filepath != "None" else None
        )
        self.val_list_filepath = (
            Path(d.val_list_filepath) if d.val_list_filepath != "None" else None
        )

        self.num_workers = (
            s.num_workers if s.num_workers >= 0 else multiprocessing.cpu_count()
        )
        self.random_seed = s.random_seed
        self.batch_size = d.batch_size

        # create augmentation pipelines
        self.fit_transform_audio = fit_transform_audio
        self.fit_transform_image = fit_transform_image
        self.val_transform_audio = val_transform_audio
        self.val_transform_image = val_transform_image

        # create class id dictionary
        class_list = pd.read_csv(d.class_list_filepath, delimiter=";", quotechar="|",)
        self.class_count = len(class_list)
        class_tensor = to_onehot(torch.arange(0, len(class_list)), len(class_list))
        self.class_dict = {
            class_list.iloc[i, 0]: class_tensor[i].float()
            for i in range(0, len(class_list))
        }

        # print(self.class_dict)

    def prepare_data(self):
        # called only on 1 GPU
        # split data into train val and test
        self.train_dataframe = pd.read_csv(
            self.train_list_filepath, delimiter=";", quotechar="|",
        )
        self.val_dataframe = pd.read_csv(
            self.val_list_filepath, delimiter=";", quotechar="|",
        )

    def setup(self, stage=None):
        # called on every GPU
        # Assign train/val datasets for use in dataloaders

        if stage == "fit" or stage is None:
            self.train_set = MultiLabelAudioSet(
                self.config,
                self.train_dataframe,
                self.class_dict,
                transform_image=self.fit_transform_image,
                transform_audio=self.fit_transform_audio,
            )
            self.val_set = MultiLabelAudioSet(
                self.config,
                self.val_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_audio=self.val_transform_audio,
                is_validation=True,
            )
            print("Train set raw size: {}".format(len(self.train_set)))
            print("Validation set raw size: {}".format(len(self.val_set)))
        # Assign test dataset for use in dataloader(s)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=filter_none_values,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=filter_none_values,
        )


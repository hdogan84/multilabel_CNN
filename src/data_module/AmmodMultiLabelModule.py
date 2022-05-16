from typing import Callable
import multiprocessing
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd

#from dataset.MultiLabelAudioSet import MultiLabelAudioSet
from dataset.MultiLabelAudioSetWithNegs import MultiLabelAudioSet
from pathlib import Path
from torchmetrics.utilities.data import to_onehot


from tools.collate_functions import filter_none_values, collate_channel_dimension

def collate_fn(batch):
    return collate_channel_dimension(filter_none_values(batch))
class AmmodMultiLabelModule(LightningDataModule):
    def __init__(
        self,
        config,
        fit_transform_signal: Callable = None,
        fit_transform_image: Callable = None,
        val_transform_signal: Callable = None,
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

        self.config = config

        self.train_list_filepath = (
            Path(config.data.train_list_filepath)
            if config.data.train_list_filepath != "None"
            else None
        )
        self.val_list_filepath = (
            Path(config.data.val_list_filepath)
            if config.data.val_list_filepath != "None"
            else None
        )
        self.test_list_filepath = (
            Path(config.data.test_list_filepath)
            if config.data.test_list_filepath != "None"
            else self.val_list_filepath
        )
        

        self.num_workers = (
            config.system.num_workers
            if config.system.num_workers >= 0
            else multiprocessing.cpu_count()
        )
        self.random_seed = config.system.random_seed
        self.batch_size = config.data.batch_size

        # create augmentation pipelines
        self.fit_transform_signal = fit_transform_signal
        self.fit_transform_image = fit_transform_image
        self.val_transform_signal = val_transform_signal
        self.val_transform_image = val_transform_image

        # create class id dictionary
        class_list = pd.read_csv(
            config.data.class_list_filepath, delimiter=";", quotechar="|",
        )
        self.class_count = len(class_list)
        class_tensor = to_onehot(torch.arange(0, len(class_list)), len(class_list))
        self.class_dict = {
            class_list.iloc[i, 0]: class_tensor[i].float()
            for i in range(0, len(class_list))
        }

        # print(self.class_dict)

    def prepare_data(self):

        # called only on 1 GPU
        # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
        pass

    def setup(self, stage=None):
        # called on every GPU
        # Assign train/val datasets for use in dataloaders

        self.train_dataframe = pd.read_csv(
            self.train_list_filepath, delimiter=";", quotechar="|"
        )
        self.val_dataframe = pd.read_csv(
            self.val_list_filepath, delimiter=";", quotechar="|"
        )
        
        self.test_dataframe = pd.read_csv(
            self.test_list_filepath,  delimiter=";", quotechar="|")
        
        if stage == "fit" or stage is None:
            self.train_set = MultiLabelAudioSet(
                self.config,
                self.train_dataframe,
                self.class_dict,
                transform_image=self.fit_transform_image,
                transform_signal=self.fit_transform_signal,
            )
            self.val_set = MultiLabelAudioSet(
                self.config,
                self.val_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_signal=self.val_transform_signal,
                is_validation=True,
            )
            print("Train set raw size: {}".format(len(self.train_set)))
            print("Validation set raw size: {}".format(len(self.val_set)))
        else:
            self.test_set = MultiLabelAudioSet(
                self.config,
                self.test_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_signal=self.val_transform_signal,
                is_validation=True,
            )
            print("Test set raw size: {}".format(len(self.test_set)))
        # Assign test dataset for use in dataloader(s)



    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size * self.config.validation.batch_size_mulitplier,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size * self.config.validation.batch_size_mulitplier,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

from typing import Callable
import multiprocessing
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
from dataset.AudioSet import AudioSet
from config.configuration import DataConfig, ScriptConfig, SystemConfig, LearningConfig
from pathlib import Path
from pytorch_lightning.metrics.utils import to_onehot


def filter_none_values(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class AmmodSingleLabelModule(LightningDataModule):
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
        self.data_list_filepath = (
            Path(d.data_list_filepath) if d.data_list_filepath != "None" else None
        )
        self.train_list_filepath = (
            Path(d.train_list_filepath) if d.train_list_filepath != "None" else None
        )
        self.val_list_filepath = (
            Path(d.val_list_filepath) if d.val_list_filepath != "None" else None
        )
        self.test_list_filepath = (
            Path(d.test_list_filepath) if d.test_list_filepath != "None" else None
        )
        self.class_list_filepath = d.class_list_filepath
        self.test_split = d.test_split
        self.val_split = d.val_split

        self.num_workers = (
            s.num_workers if s.num_workers >= 0 else multiprocessing.cpu_count()
        )
        self.random_seed = s.random_seed
        self.batch_size = d.batch_size
        self.index_label = d.index_label
        self.fit_transform_audio = fit_transform_audio
        self.fit_transform_image = fit_transform_image
        self.val_transform_audio = val_transform_audio
        self.val_transform_image = val_transform_image
        class_list = pd.read_csv(
            self.class_list_filepath, delimiter=";", quotechar="|",
        )
        self.class_count = len(class_list)
        if d.one_hot_encoding:

            class_tensor = to_onehot(torch.arange(0, len(class_list)), len(class_list))
            self.class_dict = {
                class_list.iloc[i, 0]: class_tensor[i].float()
                for i in range(0, len(class_list))
            }
        else:
            self.class_dict = {
                class_list.iloc[i, 0]: i for i in range(0, len(class_list))
            }
        # print(self.class_dict)

    def prepare_data(self):
        # called only on 1 GPU
        # split data into train val and test
        # if data_list_filepath is defind dataset hast to be split
        if self.data_list_filepath is not None:
            dataframe = pd.read_csv(
                self.data_list_filepath, delimiter=";", quotechar="|",
            )
            if self.test_split > 0:
                test_sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.test_split, random_state=self.random_seed
                )
                val_sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.val_split, random_state=self.random_seed
                )

                fit_index, test_index = next(
                    test_sss.split(dataframe, y=dataframe["labels"])
                )
                fit_dataframe = dataframe.loc[fit_index, :].reset_index(drop=True)
                self.test_dataframe = dataframe.loc[test_index, :].reset_index(
                    drop=True
                )

                train_index, val_index = next(
                    val_sss.split(fit_dataframe, y=fit_dataframe["labels"])
                )
                self.train_dataframe = fit_dataframe.loc[train_index, :].reset_index(
                    drop=True
                )
                self.val_dataframe = fit_dataframe.loc[val_index, :].reset_index(
                    drop=True
                )
                print(len(self.test_dataframe))
            else:
                val_sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.val_split, random_state=self.random_seed
                )
                fit_index, val_index = next(
                    val_sss.split(dataframe, y=dataframe["labels"])
                )
                self.train_dataframe = dataframe.loc[fit_index, :].reset_index(
                    drop=True
                )
                self.val_dataframe = dataframe.loc[val_index, :].reset_index(drop=True)
        else:
            self.train_dataframe = pd.read_csv(
                self.train_list_filepath, delimiter=";", quotechar="|",
            )
            self.val_dataframe = pd.read_csv(
                self.val_list_filepath, delimiter=";", quotechar="|",
            )
            if self.test_list_filepath is not None:
                self.test_dataframe = pd.read_csv(
                    self.test_list_filepath, delimiter=";", quotechar="|",
                )

    def setup(self, stage=None):
        # called on every GPU
        # Assign train/val datasets for use in dataloaders

        if stage == "fit" or stage is None:
            self.train_set = AudioSet(
                self.config,
                self.train_dataframe,
                self.class_dict,
                transform_image=self.fit_transform_image,
                transform_audio=self.fit_transform_audio,
                randomize_audio_segment=True,
            )
            self.val_set = AudioSet(
                self.config,
                self.val_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_audio=self.val_transform_audio,
                randomize_audio_segment=False,
                extract_complete_segment=self.config.validation.complete_segment,
                sub_segment_overlap=self.config.validation.sub_segment_overlap,
                multi_channel_handling=self.config.validation.multi_channel_handling,
                max_segment_length=self.config.validation.max_segment_length,
            )
            print("Train set size: {}".format(len(self.train_set)))
            print("Validation set size: {}".format(len(self.val_set)))
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = AudioSet(
                self.config,
                self.test_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_audio=self.val_transform_audio,
                randomize_audio_segment=False,
            )

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

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=filter_none_values,
        )


from typing import Callable
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from dataset.AudioSet import AudioSet
from pathlib import Path
from config.configuration import DataConfig, ScriptConfig, SystemConfig, LearningConfig
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
)
import albumentations as A


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
        self.data_list_filepath = d.data_list_filepath
        self.class_list_filepath = d.class_list_filepath
        self.test_split = d.test_split
        self.val_split = d.val_split
        self.num_workers = s.num_workers
        self.random_seed = s.random_seed
        self.batch_size = l.batch_size
        self.index_label = d.index_label
        self.fit_transform_audio = fit_transform_audio
        self.fit_transform_image = fit_transform_image
        self.val_transform_audio = val_transform_audio
        self.val_transform_image = val_transform_image

        # self.augment_audio = Compose(
        #     [
        #         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
        #         TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
        #         PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        #         Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        #     ]
        # )
        self.augment_image = A.Compose([A.Resize(224, 224)])

    def prepare_data(self):
        # called only on 1 GPU
        # split data into train val and test
        class_list = pd.read_csv(
            self.class_list_filepath, delimiter=";", quotechar="|",
        )
        self.class_dict = {class_list.iloc[i, 0]: i for i in range(0, len(class_list))}

        dataframe = pd.read_csv(self.data_list_filepath, delimiter=";", quotechar="|",)
        test_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_split, random_state=self.random_seed
        )
        val_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_split, random_state=self.random_seed
        )

        fit_index, test_index = next(test_sss.split(dataframe, y=dataframe["labels"]))
        fit_dataframe = dataframe.loc[fit_index, :].reset_index(drop=True)
        self.test_dataframe = dataframe.loc[test_index, :].reset_index(drop=True)

        train_index, val_index = next(
            val_sss.split(fit_dataframe, y=fit_dataframe["labels"])
        )
        self.train_dataframe = fit_dataframe.loc[train_index, :].reset_index(drop=True)
        self.val_dataframe = fit_dataframe.loc[val_index, :].reset_index(drop=True)

        print(len(self.train_dataframe))
        print(len(self.val_dataframe))
        print(len(self.test_dataframe))

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
            )
            self.val_set = AudioSet(
                self.config,
                self.val_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_audio=self.val_transform_audio,
            )
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = AudioSet(
                self.config,
                self.test_dataframe,
                self.class_dict,
                transform_image=self.val_transform_image,
                transform_audio=self.val_transform_audio,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

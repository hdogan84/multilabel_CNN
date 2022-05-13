from typing import Callable
import multiprocessing
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
import collections
from dataset.ColorSpecAudioSet import ColorSpecAudioSet as MultiLabelAudioSet
from pathlib import Path
from pytorch_lightning.metrics.utils import to_onehot
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule

import re
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def filter_none_values(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return collate(batch)
  

class ColorSpecAmmodMultiLabelModule(AmmodMultiLabelModule):

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
            collate_fn=filter_none_values,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size * self.config.validation.batch_size_mulitplier,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=filter_none_values,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size * self.config.validation.batch_size_mulitplier,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=filter_none_values,
        )
from typing import Callable
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from logging import debug, warn
from random import random
import librosa

# import simpleaudio as sa
from math import ceil
import numpy as np

from tools.audio_tools import (
    read_audio_parts,
    get_mel_spec,
)


class FileListAnalyseSet(Dataset):
    def __init__(
        self,
        config,
        file_list: list,
        transform_image: Callable = None,
        transform_signal: Callable = None,
        step_size: float = 1,
    ):
        """
        Args:
            data_rows : Path to the csv file with annotations.
            data_path (string): Directory of database with derivate directory in it. If none path in csv is used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.transform_signal = transform_signal
        self.transform_image = transform_image
        self.step_size = step_size
        self.segments = []
        for filepath in file_list:
            duration = librosa.get_duration(filename=filepath)  # duration in s
            segment_count = ceil(duration / step_size)
            for i in range(segment_count):
                # on last could be longer the duration use end to calc start

                start_time = (
                    (i * step_size)
                    if i < segment_count - 1
                    else duration - self.config.data.segment_duration
                )
                end_time = (
                    (i * step_size + self.config.data.segment_duration)
                    if i < segment_count - 1
                    else duration
                )
                segment = {
                    "filepath": filepath,
                    "start_time": start_time,
                    "end_time": end_time,
                    "channel": "to_mono",
                }
                self.segments.append(segment)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        debug("Get item index: {}".format(index))
        segment = self.segments[index]
        filepath = segment["filepath"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]

        dummy_class_tensor = torch.zeros(5)  # Dummy tensor to

        audio_data = None
        try:
            audio_data = read_audio_parts(
                filepath,
                [(start_time,end_time)],
                self.config.data.segment_duration,
                self.config.audio_loading.sample_rate,
                channel_mixing_strategy=self.config.audio_loading.channel_mixing_strategy,
                backend=self.config.audio_loading.backend,
            )
        except Exception as error:
            print(error)
            return None

        tensor_list = []
        y_list = []
        index_list = []
        for channel in range(audio_data.shape[0]):
            augmented_signal, y = (
                self.transform_signal(
                    samples=audio_data[0, :],
                    sample_rate=self.config.audio_loading.sample_rate,
                    y=dummy_class_tensor,
                )
                if self.transform_signal is not None
                else (audio_data[channel, :], dummy_class_tensor)
            )
            mel_spec = get_mel_spec(
                augmented_signal,
                self.config.audio_loading.fft_size_in_samples,
                self.config.audio_loading.fft_hop_size_in_samples,
                self.config.audio_loading.sample_rate,
                num_of_mel_bands=self.config.audio_loading.num_of_mel_bands,
                mel_start_freq=self.config.audio_loading.mel_start_freq,
                mel_end_freq=self.config.audio_loading.mel_end_freq,
            )
            debug("Done got mel spec index {}".format(index))
            # format mel_spec to image with one channel
            h, w = mel_spec.shape
            image_data = np.empty(
                (
                    h,
                    w,
                    3
                    if self.config.audio_loading.use_color_channels == "use_all"
                    else 1,
                ),
                dtype=np.uint8,
            )
            if self.config.audio_loading.use_color_channels == "use_all":
                image_data = np.empty((h, w, 3), dtype=np.uint8)
                image_data[:, :, 1] = mel_spec
                image_data[:, :, 2] = mel_spec
            image_data[:, :, 0] = mel_spec

            augmented_image_data = (
                self.transform_image(image=image_data)["image"]
                if self.transform_image is not None
                else image_data
            )
            debug("Done image augmenting index {}".format(index))
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.config.audio_loading.normalize_mean,
                        std=self.config.audio_loading.normalize_std,
                    ),
                ]
            )
            tensor = transform(augmented_image_data)

            tensor_list.append(tensor)
            y_list.append(y)
            index_list.append(torch.tensor(index))

        tensor = torch.stack(tensor_list)
        # print(tensor.shape)
        y = torch.stack(y_list)
        index = torch.stack(index_list)
        # plt.imshow(augmented_image_data, interpolation="nearest")
        # plt.show()
        return tensor, y, index

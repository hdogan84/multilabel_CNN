import argparse
from typing import Callable
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from config.configuration import ScriptConfig, AudioLoadingConfig

# import simpleaudio as sa
import numpy as np
import librosa
import librosa.display
from tools.audio_tools import (
    read_audio_segment,
    get_mel_spec,
    Mixing,
    Padding,
)
from tools.plot_tools import print_mel_spec


class AudioSet(Dataset):
    """Face Landmarks dataset."""

    root_dir: Path

    def __init__(
        self,
        config: ScriptConfig,
        data_rows: ScriptConfig,
        class_dict: dict,
        transform_image: Callable = None,
        transform_audio: Callable = None,
    ):
        """
        Args:
            data_rows : Path to the csv file with annotations.
            data_path (string): Directory of database withr derivate directory in it. If none path in csv is used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        d = config.data
        a = config.audio_loading

        self.data_rows = data_rows
        self.class_dict = class_dict
        self.data_path = d.data_path
        self.index_start_time = d.index_start_time
        self.index_end_time = d.index_end_time
        self.index_filepath = d.index_filepath
        self.index_label = d.index_label

        self.transform_audio = transform_audio
        self.transform_image = transform_image

        self.segment_length = a.segment_length
        self.sample_rate = a.sample_rate
        self.mixing_strategy = a.mixing_strategy
        self.padding_strategy = a.padding_strategy

        self.fft_size_in_samples = a.fft_size_in_samples
        self.fft_hop_size_in_samples = a.fft_hop_size_in_samples
        self.num_of_mel_bands = a.num_of_mel_bands
        self.mel_start_freq = a.mel_start_freq
        self.mel_end_freq = a.mel_end_freq

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        filepath = Path(self.data_rows.iloc[idx, self.index_filepath])
        label = self.data_rows.iloc[idx, self.index_label]
        if self.data_path is not None:
            filepath = self.data_path.joinpath(
                *filepath.parts[len(filepath.parts) - 6 :]
            )

        start = self.data_rows.iloc[idx, self.index_start_time]
        stop = self.data_rows.iloc[idx, self.index_end_time]
        # print(filepath.as_posix())
        audio_data = read_audio_segment(
            filepath,
            start,
            stop,
            self.segment_length,
            self.sample_rate,
            mixing_strategy=self.mixing_strategy,
            padding_strategy=self.padding_strategy,
        )
        # print(len(audio_data))

        augmented_signal = audio_data = (
            self.transform_audio(samples=audio_data, sample_rate=self.sample_rate)
            if self.transform_audio is not None
            else audio_data
        )

        mel_spec = get_mel_spec(
            augmented_signal,
            self.fft_size_in_samples,
            self.fft_hop_size_in_samples,
            self.sample_rate,
            num_of_mel_bands=self.num_of_mel_bands,
            mel_start_freq=self.mel_start_freq,
            mel_end_freq=self.mel_end_freq,
        )
        # format mel_spec to image with one channel
        h, w = mel_spec.shape
        image_data = np.empty((h, w, 1), dtype=np.uint8)
        image_data[:, :, 0] = mel_spec
        # image_data[:, :, 1] = mel_spec
        # image_data[:, :, 2] = mel_spec

        augmented_image_data = (
            self.transform_image(image=image_data)["image"]
            if self.transform_image is not None
            else mel_spec
        )
        tensor = transforms.ToTensor()(augmented_image_data).float()
        # plt.imshow(augmented_image_data, interpolation="nearest")
        # plt.show()
        label_id = self.class_dict[label]
        return tensor, label_id

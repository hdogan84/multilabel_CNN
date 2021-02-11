import functools
import os
import random
import sys
import tempfile
import uuid
import warnings
import random
import librosa
import torch
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt, convolve
from tools.audio_tools import read_audio_segment
from audiomentations.core.utils import (
    calculate_rms,
    calculate_desired_noise_rms,
    get_file_paths,
    convert_decibels_to_amplitude_ratio,
    convert_float_samples_to_int16,
)
from audiomentations.core.transforms_interface import BaseWaveformTransform


class AddSameClassSignal(BaseWaveformTransform):
    """Add pink noise to the samples with random Signal to Noise Ratio (SNR)"""

    def __init__(
        self,
        p=0.5,
        min_ssr=-40,
        max_ssr=3,
        max_n=3,
        padding_strategy="wrap_around",
        channel_mixing_strategy="take_one",
        data_path=None,
        data_list_filepath=None,
        class_list_filepath=None,
        index_filepath=5,
        index_start_time=1,
        index_end_time=2,
        index_label=3,
        index_channels=6,
        delimiter=";",
        quotechar="|",
    ):
        """
        :max_n: how offten ad maximum the same class is added
        :padding_strategy: what to do if signal loaded is to short (wrap_around, silence, random) 
         random means randomly wrap around or silence is added
        :param min_ssr: Minimum signal-to-signal ratio
        :param max_ssr: Maximum signal-to-signal ratio
        :param p:
        """
        super().__init__(p)
        self.min_ssr = min_ssr
        self.max_ssr = max_ssr
        self.max_n = max_n
        self.padding_strategy = padding_strategy
        self.channel_mixing_strategy = channel_mixing_strategy
        dataframe = pd.read_csv(
            data_list_filepath, delimiter=delimiter, quotechar=quotechar
        )
        if data_path is not None:
            dataframe.iloc[:, index_filepath] = dataframe.iloc[:, index_filepath].apply(
                data_path.joinpath
            )
        # if class_list_filepath is set transform class to class_index
        if class_list_filepath is not None:
            class_list = pd.read_csv(class_list_filepath, delimiter=";", quotechar="|",)
            class_dict = {class_list.iloc[i, 0]: i for i in range(0, len(class_list))}
            dataframe.iloc[:, index_label] = dataframe.iloc[:, index_label].apply(
                class_dict.__getitem__
            )

        if index_channels is None:
            tmp_data_rows = list(
                zip(
                    list(range(len(dataframe))),
                    dataframe.iloc[:, index_filepath],
                    dataframe.iloc[:, index_label],
                    dataframe.iloc[:, index_start_time,],
                    dataframe.iloc[:, index_end_time,],
                    np.ones(1, len(dataframe)),
                )
            )
        else:
            tmp_data_rows = list(
                zip(
                    list(range(len(dataframe))),
                    dataframe.iloc[:, index_filepath],
                    dataframe.iloc[:, index_label],
                    dataframe.iloc[:, index_start_time,],
                    dataframe.iloc[:, index_end_time,],
                    dataframe.iloc[:, index_channels,],
                )
            )

        self.class_data_dict = {i: [] for i in dataframe.iloc[:, index_label].unique()}
        for data in tmp_data_rows:
            self.class_data_dict[data[2]].append(data)

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ssr"] = np.random.uniform(self.min_ssr, self.max_ssr)
            self.parameters["n_times"] = round(np.random.uniform(1, self.max_n))
            if self.padding_strategy == "random":
                self.parameters["padding_strategies"] = [
                    random.choice(["wrap_around", "silence"])
                    for i in range(self.parameters["n_times"])
                ]
            else:
                self.parameters["padding_strategies"] = [
                    self.padding_strategy for i in range(self.parameters["n_times"])
                ]

            if self.channel_mixing_strategy == "random":
                self.parameters["channel_mixing_strategy"] = [
                    random.choice(["take_one", "random_mix"])
                    for i in range(self.parameters["n_times"])
                ]
            else:
                self.parameters["channel_mixing_strategies"] = [
                    self.channel_mixing_strategy
                    for i in range(self.parameters["n_times"])
                ]

    # 10**(db/20) -40 db ; 6db ;
    # how offten,
    # mode padding | wraparound | random

    def apply(self, samples, sample_rate, y):
        result = samples
        temp_y = y
        for n in range(self.parameters["n_times"]):
            # if y is one hot encoded reduce index value
            if torch.is_tensor(y) and y.shape[0] > 1:
                for x in range(0, y.shape[0]):
                    if y[x] > 0:
                        temp_y = x
                        break

            same_class_entry = random.choice(self.class_data_dict[temp_y])
            audio_data = read_audio_segment(
                same_class_entry[1],
                same_class_entry[3],
                same_class_entry[4],
                len(samples) / sample_rate,
                sample_rate,
                channel_mixing_strategy=self.parameters["channel_mixing_strategies"][n],
                padding_strategy=self.parameters["padding_strategies"][n],
                randomize_audio_segment=True,
                channel=random.randint(0, same_class_entry[5] - 1),
            )
            # alter volume
            audio_data = audio_data * 10 ** (self.parameters["ssr"] / 20)
            result = result + audio_data

        return result

    def __call__(self, samples, sample_rate, y):
        if not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and len(samples) > 0:
            return self.apply(samples, sample_rate, y)
        return samples

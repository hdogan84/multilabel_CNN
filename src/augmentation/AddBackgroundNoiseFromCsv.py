import functools
import os
import random
import sys
import tempfile
import uuid
import warnings

import librosa
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfilt, convolve
from audiomentations.augmentations.transforms import AddBackgroundNoise
from audiomentations.core.utils import (
    calculate_rms,
    calculate_desired_noise_rms,
    get_file_paths,
    convert_decibels_to_amplitude_ratio,
    convert_float_samples_to_int16,
)


class AddBackgroundNoiseFromCsv(AddBackgroundNoise):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.
    Note that the gain of the added noise is relative to the amount of signal in the input. This
    implies that if the input is completely silent, no noise will be added.
    """

    def __init__(
        self,
        csv_filepath,
        csv_row_index="filepath",
        min_snr_in_db=3,
        max_snr_in_db=30,
        p=0.5,
        delimiter=";",
        quotechar="|",
    ):
        """
        :param sounds_path: Path to a folder that contains sound files to randomly mix in. These
            files can be flac, mp3, ogg or wav.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB
        :param p:
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        try:
            super().__init__(
                sounds_path="not a path",
                min_snr_in_db=min_snr_in_db,
                max_snr_in_db=max_snr_in_db,
                p=p,
            )
        except AssertionError:
            print("do Nothing")

        dataframe = pd.read_csv(csv_filepath, delimiter=delimiter, quotechar=quotechar)

        self.sound_file_paths = dataframe[csv_row_index].tolist()
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

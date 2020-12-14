import functools
import os
import random
import sys
import tempfile
import uuid
import warnings

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, convolve

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    calculate_rms,
    calculate_desired_noise_rms,
    get_file_paths,
    convert_decibels_to_amplitude_ratio,
    convert_float_samples_to_int16,
)


class AddImpulseResponse(BaseWaveformTransform):
    """Convolve the audio with a random impulse response.
    Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
    Impulse responses are represented as wav files in the given ir_path.
    """

    # main
    def __init__(
        self,
        ir_path="/tmp/ir",
        p=0.5,
        lru_cache_size=128,
        leave_length_unchanged: bool = False,
    ):
        """
        :param ir_path: Path to a folder that contains one or more wav files of impulse
        responses. Must be str or a Path instance.
        :param p:
        :param lru_cache_size: Maximum size of the LRU cache for storing impulse response files
        in memory.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        """
        super().__init__(p)
        self.ir_files = get_file_paths(ir_path)
        self.ir_files = [str(p) for p in self.ir_files]
        assert len(self.ir_files) > 0
        self.__load_ir = functools.lru_cache(maxsize=lru_cache_size)(
            AddImpulseResponse.__load_ir
        )
        self.leave_length_unchanged = leave_length_unchanged

    @staticmethod
    def __load_ir(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    # main
    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_files)

    # main
    def apply(self, samples, sample_rate):
        ir, sample_rate2 = self.__load_ir(self.parameters["ir_file_path"], sample_rate)
        if sample_rate != sample_rate2:
            # This will typically not happen, as librosa should automatically resample the
            # impulse response sound to the desired sample rate
            raise Exception(
                "Recording sample rate {} did not match Impulse Response signal"
                " sample rate {}!".format(sample_rate, sample_rate2)
            )
        signal_ir = convolve(samples, ir)
        max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        scale = 0.5 / max_value
        signal_ir *= scale
        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]
        return signal_ir

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddImpulseResponse gets discarded when pickling it."
            " E.g. this means the cache will be not be used when using AddImpulseResponse"
            " together with multiprocessing on Windows"
        )
        del state["_AddImpulseResponse__load_ir"]
        return state
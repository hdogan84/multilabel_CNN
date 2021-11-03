# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import io
import librosa
import math
import torch
import numpy as np
import yaml
import os
from base_audio_handler import AudioHandler
from torchvision import transforms

from scipy.signal import butter, sosfilt
from PIL import Image


class Birdid_254_Handler(AudioHandler):
    """
    A custom model handler implementation.
    """

    sample_rate = 22050
    fft_size_in_samples = 1536
    fft_hop_size_in_samples = 360
    num_of_mel_bands = 128
    mel_start_freq = 20.0
    mel_end_freq = 10300.0  # 16000
    # normalize_mean = 0.456
    # normalize_std = 0.224
    normalize_mean = [0.5, 0.4, 0.3]
    normalize_std = [0.5, 0.3, 0.1]

    segment_duration = 5

    segment_step = 1
    batch_size = 120  # 64 * 5
    convert_to_mono = False

    # NNN
    NumOfLowFreqsInPixelToCutMax = 4
    NumOfHighFreqsInPixelToCutMax = 6
    #ImageSize = 224
    imageHeight = 224
    resizeFactor = (imageHeight/(num_of_mel_bands-NumOfLowFreqsInPixelToCutMax/2.0-NumOfHighFreqsInPixelToCutMax/2.0))
    imageWidth = int(resizeFactor * segment_duration * sample_rate / fft_hop_size_in_samples)
    ImageSize = (imageWidth, imageHeight)

    # NNN
    def __apply_high_pass_filter__(self, input, sample_rate):

        order = 2
        cutoff_frequency = 2000

        sos = butter(
            order, cutoff_frequency, btype="highpass", output="sos", fs=sample_rate
        )
        output = sosfilt(sos, input, axis=0)

        # If anything went wrong (nan in array or max > 1.0 or min < -1.0) --> return original input
        if np.isnan(output).any() or np.max(output) > 1.0 or np.min(output) < -1.0:
            print("Warning filter instability: filter not applied")
            output = input

        # print(type, order, np.min(input), np.max(input), np.min(output), np.max(output))

        return output

    def preprocess_audio_data(self, audio_data):

        # Normalize (to prevent filter errors)
        audio_data /= np.max(audio_data)
        audio_data *= 0.5

        # Apply high pass filter (if no filter issues)
        audio_data = np.transpose(
            audio_data
        )  # [n_channels x n_frames] --> [n_frames x n_channels]

        audio_data = self.__apply_high_pass_filter__(audio_data, self.sample_rate)

        # Make sure audio_data is in correct format for librosa processing
        audio_data = np.transpose(
            audio_data
        )  # [n_frames x n_channels] --> [n_channels x n_frames]

        # Normalize to -3dB
        audio_data /= np.max(audio_data)
        audio_data *= 0.71

        # librosa needs Fortran-contiguous audio buffer (maybe not needed anymore)
        if not audio_data.flags["F_CONTIGUOUS"]:
            audio_data = np.asfortranarray(audio_data)

        return audio_data

    def create_spectogram(self, audio_data):

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.fft_size_in_samples,
            hop_length=self.fft_hop_size_in_samples,
            n_mels=self.num_of_mel_bands,
            fmin=self.mel_start_freq,
            fmax=self.mel_end_freq,
            power=2.0,
        )

        # Convert power spec to dB scale (compute dB relative to peak power)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)

        # NNN
        NumOfLowFreqsInPixelToCut = int(self.NumOfLowFreqsInPixelToCutMax / 2.0)
        NumOfHighFreqsInPixelToCut = int(self.NumOfHighFreqsInPixelToCutMax / 2.0)

        mel_spec = mel_spec[NumOfLowFreqsInPixelToCut:-NumOfHighFreqsInPixelToCut]

        # Flip spectrum vertically (only for better visualization, low freq. at bottom)
        mel_spec = mel_spec[::-1, ...]

        # Normalize values between 0 and 1 (& prevent divide by zero)
        mel_spec -= mel_spec.min()
        mel_spec_max = mel_spec.max()
        if mel_spec_max:
            mel_spec /= mel_spec_max

        max_val = 255.9
        mel_spec *= max_val
        mel_spec = max_val - mel_spec

        return mel_spec

    def postprocess_spec(self, mel_spec):

        # Resize image
        mel_spec_PIL = Image.fromarray(mel_spec.astype(np.uint8))
        
        # mel_spec_PIL = mel_spec_PIL.resize(
        #     (self.ImageSize, self.ImageSize), Image.LANCZOS
        # )

        mel_spec_PIL = mel_spec_PIL.resize((self.ImageSize), Image.LANCZOS)

        mel_spec = np.array(mel_spec_PIL, dtype=np.uint8)  # Cast to int8 ? needed ?

        # one --> three channel rgb image
        h, w = mel_spec.shape
        ThreeChannelSpec = np.empty((h, w, 3), dtype=np.uint8)
        ThreeChannelSpec[:, :, 0] = mel_spec
        ThreeChannelSpec[:, :, 1] = mel_spec
        ThreeChannelSpec[:, :, 2] = mel_spec

        image_data = Image.fromarray(ThreeChannelSpec)  # ? needed ?

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )

        tensor = transform(image_data)

        return tensor

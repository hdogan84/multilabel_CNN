from os import error
from pathlib import Path
from typing import Tuple
import soundfile as sf
import numpy as np
import random
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, stft
from enum import Enum
import logging

logger = logging.getLogger("audio_service")


class Padding:
    SILENCE = "silence"
    WRAP_AROUND = "wrap_around"


class Mixing:
    TAKE_ONE = "take_one"
    RANDOM_MIX = "random_mix"
    TAKE_ALL = "take_all"
    TO_MONO = "to_mono"


def read_audio_segment(
    filepath: Path,
    start: int,
    stop: int,
    desired_length: int,
    sample_rate: int,
    channel_mixing_strategy=Mixing.TAKE_ONE,
    padding_strategy=Padding.SILENCE,
    randomize_audio_segment: bool = False,
    channel: int = 0,
):
    duration = stop - start
    audio_data = []
    if filepath.exists() == False:
        raise Exception("File does not exsts")
    if (
        duration >= desired_length
    ):  # desired_length of audio chunk is greater then wanted part
        max_offset = duration - desired_length
        offset = max_offset * random.random() if randomize_audio_segment else 0 + start
        reading_start = int(offset * sample_rate)
        reading_stop = reading_start + int(desired_length * sample_rate)
        audio_data = sf.read(
            filepath, start=reading_start, stop=reading_stop, always_2d=True
        )[0]
    else:
        reading_start = int(start * sample_rate)
        reading_stop = int(stop * sample_rate)
        audio_data = sf.read(
            filepath, start=reading_start, stop=reading_stop, always_2d=True
        )[0]
    if len(audio_data) == 0:

        raise Exception("Error during reading file 1: {}".format(filepath.as_posix()))
    # IF more then one channel do mixing
    if audio_data.shape[1] > 1:
        if channel_mixing_strategy == Mixing.TAKE_ONE:
            audio_data = audio_data[:, channel]
        else:
            raise NotImplementedError()
    else:
        audio_data = audio_data[:, 0]

    # If segment smaller than desired start padding it
    desired_sample_length = round(desired_length * sample_rate)

    if len(audio_data) < desired_sample_length:
        if padding_strategy == Padding.WRAP_AROUND:
            # logger.debug("cylic")
            padded_audio_data = audio_data.copy()
            # change starting position
            if randomize_audio_segment:
                padded_audio_data = padded_audio_data[
                    int(len(audio_data) * random.random()) : len(audio_data) - 1
                ]
            while desired_sample_length > len(padded_audio_data):
                padded_audio_data = np.append(padded_audio_data, audio_data)
            audio_data = padded_audio_data[:desired_sample_length]
        elif padding_strategy == Padding.SILENCE:
            padding_length = desired_sample_length - len(audio_data)
            if randomize_audio_segment:
                audio_data = np.append(
                    np.full(int(random.random() * padding_length), 0.0000001),
                    audio_data,
                )
            audio_data = np.append(
                audio_data, np.full(desired_sample_length - len(audio_data), 0.0000001)
            )
        else:
            raise NotImplementedError()

    return audio_data


def read_audio_parts(
    filepath: Path,
    parts: list,
    desired_length: int,
    sample_rate: int,
    channel_mixing_strategy=Mixing.TAKE_ONE,
    channel: int = 0,
):

    result = None

    for (start_time, end_time) in parts:
        reading_start = int(start_time * sample_rate)
        reading_stop = int(end_time * sample_rate)
        audio_data, sample_rate = sf.read(
            filepath, start=reading_start, stop=reading_stop, always_2d=True
        )
        audio_data
        if result is None:
            result = audio_data
        else:
            result = np.concatenate((result, audio_data))

    if len(result) == 0:
        print(filepath)
        raise Exception("Error during reading file 2:".format(filepath))
    # IF more then one channel do mixing
    if result.shape[1] > 1:
        if channel_mixing_strategy == Mixing.TAKE_ONE:
            result = result[:, channel]
        elif channel_mixing_strategy == Mixing.TO_MONO:
            result = np.sum(result, axis=1) / result.shape[1]
        else:
            raise NotImplementedError()
    else:
        result = result[:, 0]

    # If segment smaller than desired start padding it
    desired_sample_length = round(desired_length * sample_rate)

    if len(result) < desired_sample_length:
        result = np.append(
            result, np.full(desired_sample_length - len(result), 0.0000001)
        )
    return result


def get_mel_spec(
    audio_data,
    fft_size_in_samples,
    fft_hop_size_in_samples,
    sample_rate,
    num_of_mel_bands=128,
    mel_start_freq=20.0,
    mel_end_freq=16000.0,
):
    if np.isfinite(audio_data).all() is False:
        raise Exception(
            "Error: Audio content not finite {}".format(source_file_path.as_posix())
        )
        return

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=fft_size_in_samples,
        hop_length=fft_hop_size_in_samples,
        n_mels=num_of_mel_bands,
        fmin=mel_start_freq,
        fmax=mel_end_freq,
        #power=2.0,
        power=2.0,
    )

    # Convert power spec to dB scale (compute dB relative to peak power)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)
    # mel_spec = librosa.power_to_db(mel_spec, ref=np.mean, top_db=100)
    # mel_spec_temp=mel_spec[abs(mel_spec)>0]
    # mel_spec[mel_spec==0]=np.min(mel_spec_temp)
    # mel_spec=20*np.log(mel_spec/0.00002)
    # mel_spec[mel_spec<-70]=-70 

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
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


def resample_audio_file(
    source_file_path: Path,
    target_file_path: Path,
    sample_rate=None,
    resampleType="kaiser_fast",
) -> Tuple[float, int]:
    # resample on load to higher prevent instability of the filter
    # logger.debug("Read: {} \n write to: ".format(source_file_path))
    y = []

    y, sr = librosa.load(source_file_path, sr=None, mono=False, res_type=resampleType,)

    if np.isfinite(y).all() is False:
        raise Exception(
            "Error opening audio file for {}".format(source_file_path.as_posix())
        )
        return

    # Normalize to -3 dB
    y /= np.max(y)
    y *= 0.7071

    y = apply_high_pass_filter(y, sr, source_file_path)
    y = librosa.resample(y, sr, sample_rate, res_type=resampleType,)
    if len(y.shape) > 1:
        y = np.transpose(y)  # [nFrames x nChannels] --> [nChannels x nFrames]
        # logger.debug("write to target_file_path: {}".format(target_file_path))
    sf.write(target_file_path, y, sample_rate, "PCM_16")
    samples, channels = y.shape
    return samples / sample_rate, channels


def apply_high_pass_filter(input, sample_rate: int, filePath: Path):
    order = 2
    cutoff_frequency = 2000

    sos = butter(
        order, cutoff_frequency, btype="highpass", output="sos", fs=sample_rate
    )
    output = sosfilt(sos, input, axis=0)

    # If anything went wrong (nan in array or max > 1.0 or min < -1.0) --> return original input
    if np.isnan(output).any() or np.max(output) > 1.0 or np.min(output) < -1.0:
        logger.warn("Warning filter: {}".format(filePath.as_posix()))
        output = input

    # logger.debug(type, order, np.min(input), np.max(input), np.min(output), np.max(output))

    return output

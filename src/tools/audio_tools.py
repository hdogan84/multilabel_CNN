from pathlib import Path
import soundfile as sf
import numpy as np
import random
import librosa
from enum import Enum


class Padding:
    SILENCE = "silence"
    CYCLIC = "cyclic"


class Mixing:
    TAKE_FIRST = "take_first"
    RANDOM_MIX = "random_mix"


def read_audio_segment(
    filepath: Path,
    start: int,
    stop: int,
    desired_length: int,
    sample_rate: int,
    mixing_strategy=Mixing.TAKE_FIRST,
    padding_strategy=Padding.SILENCE,
    randomize_audio_segment: bool = False,
):
    duration = stop - start
    audio_data = []
    if filepath.exists() == False:
        print("File does not exsts")
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

    # IF more then one channel do mixing
    if audio_data.shape[1] > 1:
        if mixing_strategy == Mixing.TAKE_FIRST:
            audio_data = audio_data[:, 0]
        else:
            raise NotImplementedError()
    else:
        audio_data = audio_data[:, 0]

    # If segment smaller than desired start padding it
    desired_sample_length = desired_length * sample_rate

    if len(audio_data) < desired_sample_length:
        if padding_strategy == Padding.CYCLIC:
            # print("cylic")
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


def get_mel_spec(
    audio_data,
    fft_size_in_samples,
    fft_hop_size_in_samples,
    sample_rate,
    num_of_mel_bands=128,
    mel_start_freq=20.0,
    mel_end_freq=16000.0,
):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=fft_size_in_samples,
        hop_length=fft_hop_size_in_samples,
        n_mels=num_of_mel_bands,
        fmin=mel_start_freq,
        fmax=mel_end_freq,
        power=2.0,
    )

    # Convert power spec to dB scale (compute dB relative to peak power)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)

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

import argparse
from typing import Callable
from librosa.core import audio
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from config.configuration import (
    ScriptConfig,
    AudioLoadingConfig,
    ValidationConfig,
    DataConfig,
)

# import simpleaudio as sa
from math import ceil
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
from functools import reduce


class AudioSet(Dataset):
    """Face Landmarks dataset."""

    root_dir: Path

    def __init__(
        self,
        config: ScriptConfig,
        raw_data_rows: list,
        class_dict: dict,
        extract_complete_segment: bool = False,
        sub_segment_overlap: float = 1.0,
        multi_channel_handling: str = "take_first",
        sub_segment_rest_handling: str = "drop",
        max_segment_length: float = None,
        transform_image: Callable = None,
        transform_audio: Callable = None,
        randomize_audio_segment: bool = False,
    ):
        """
        Args:
            data_rows : Path to the csv file with annotations.
            data_path (string): Directory of database withr derivate directory in it. If none path in csv is used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        d: DataConfig = config.data
        a: AudioLoadingConfig = config.audio_loading

        self.class_dict = class_dict
        self.data_path = d.data_path
        self.index_start_time = d.index_start_time
        self.index_end_time = d.index_end_time
        self.index_filepath = d.index_filepath
        self.index_label = d.index_label
        self.index_channels = d.index_channels
        self.multi_channel_handling = multi_channel_handling
        if multi_channel_handling != "take_first":
            if self.index_channels is None:
                raise Exception("Please add index of channels in dataset")

        self.transform_audio = transform_audio
        self.transform_image = transform_image

        self.randomize_audio_segment = randomize_audio_segment
        self.segment_length = a.segment_length
        self.sample_rate = a.sample_rate
        self.channel_mixing_strategy = a.channel_mixing_strategy
        self.padding_strategy = a.padding_strategy

        self.fft_size_in_samples = a.fft_size_in_samples
        self.fft_hop_size_in_samples = a.fft_hop_size_in_samples
        self.num_of_mel_bands = a.num_of_mel_bands
        self.mel_start_freq = a.mel_start_freq
        self.mel_end_freq = a.mel_end_freq

        # combin data_path with filepath with  of all entryies

        if d.data_path is not None:
            raw_data_rows.iloc[:, self.index_filepath] = raw_data_rows.iloc[
                :, self.index_filepath
            ].apply(d.data_path.joinpath)
        # if now index for channels is given play it save and estimate 1 channel
        if self.index_channels is None:
            tmp_data_rows = list(
                zip(
                    list(range(len(raw_data_rows))),
                    raw_data_rows.iloc[:, self.index_filepath],
                    raw_data_rows.iloc[:, self.index_label],
                    raw_data_rows.iloc[:, self.index_start_time,],
                    raw_data_rows.iloc[:, self.index_end_time,],
                    np.ones(1, len(raw_data_rows)),
                )
            )
        else:
            tmp_data_rows = list(
                zip(
                    list(range(len(raw_data_rows))),
                    raw_data_rows.iloc[:, self.index_filepath],
                    raw_data_rows.iloc[:, self.index_label],
                    raw_data_rows.iloc[:, self.index_start_time,],
                    raw_data_rows.iloc[:, self.index_end_time,],
                    raw_data_rows.iloc[:, self.index_channels,],
                )
            )
        self.data_rows = []
        if extract_complete_segment:
            overlap_time = sub_segment_overlap * a.segment_length
            hop_length = a.segment_length - overlap_time
            for data in tmp_data_rows:

                index, filepath, label, start, end, channels = data
                duration = end - start
                if max_segment_length is not None:
                    if duration > max_segment_length:
                        duration = max_segment_length
                        end = start + max_segment_length

                # add hops read complete file
                hops_needed = ceil(duration / hop_length)
                if multi_channel_handling == "take_first":
                    channels = 1
                for channel in range(channels):
                    if hops_needed == 1:
                        self.data_rows.append(
                            (index, filepath, label, start, end, channel)
                        )
                    else:
                        for sub_segment in range(hops_needed - 1):
                            sub_segment_start = sub_segment * hop_length
                            self.data_rows.append(
                                (
                                    index,
                                    filepath,
                                    label,
                                    sub_segment_start,
                                    sub_segment_start + a.segment_length,
                                    channel,
                                )
                            )
                        # add last sub_segment only if length is half duration, prevent to short sub_segment of segment

                        if end - (hops_needed - 1) * hop_length == a.segment_length:
                            self.data_rows.append(
                                (
                                    index,
                                    filepath,
                                    label,
                                    (hops_needed - 1) * hop_length,
                                    end,
                                    channel,
                                )
                            )
                        else:
                            if sub_segment_rest_handling == "drop":
                                pass
                            # move start point so sub_segment has audio segment read length
                            elif sub_segment_rest_handling == "move_start":
                                self.data_rows.append(
                                    (
                                        index,
                                        filepath,
                                        label,
                                        end - a.segment_length,
                                        end,
                                        channel,
                                    )
                                )
                            else:
                                raise Exception(
                                    "sub_segment_rest_handling {} not implemented".format(
                                        sub_segment_rest_handling
                                    )
                                )
                                # TODO: last sample should start earlier and ends at end
                pass

        else:
            for data in tmp_data_rows:
                index, filepath, label, start, end, channels = data
                if multi_channel_handling == "take_first":
                    channels = 1
                for channel in range(channels):
                    self.data_rows.append((index, filepath, label, start, end, channel))

    def __len__(self):
        return len(self.data_rows)

    def __mapToClassIndex__(self, index):
        return self.class_dict[index]

    def __reduceToOneTensor__(self, tensor_list):
        result = tensor_list[0]
        if len(tensor_list) > 1:
            for t in tensor_list[1:]:
                result = result + t
        # reduce values > 1 to one
        result[result > 0] = 1
        return result

    def __getitem__(self, index):
        # if torch.is_tensor(index):
        #     index = index.tolist()
        segment_index = self.data_rows[index][0]
        filepath = Path(self.data_rows[index][1])
        labels = self.data_rows[index][2].split(",")
        labels.append(labels[0])
        label_tensor = self.__reduceToOneTensor__(
            list(map(self.__mapToClassIndex__, labels))
        )

        start = self.data_rows[index][3]
        stop = self.data_rows[index][4]
        # print("get item channel: {}".format(self.data_rows[index][5]))
        audio_data = None
        try:
            audio_data = read_audio_segment(
                filepath,
                start,
                stop,
                self.segment_length,
                self.sample_rate,
                channel_mixing_strategy=self.channel_mixing_strategy,
                padding_strategy=self.padding_strategy,
                randomize_audio_segment=self.randomize_audio_segment,
                channel=self.data_rows[index][5],
            )
        except:
            return None

        augmented_signal, y = audio_data = (
            self.transform_audio(
                samples=audio_data, sample_rate=self.sample_rate, y=label_tensor
            )
            if self.transform_audio is not None
            else (audio_data, label_tensor)
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
            else image_data
        )

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=0.456, std=0.224),]
        )

        tensor = transform(augmented_image_data)
        # plt.imshow(augmented_image_data, interpolation="nearest")
        # plt.show()

        return tensor, y, segment_index

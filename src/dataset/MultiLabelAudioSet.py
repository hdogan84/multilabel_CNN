from typing import Callable
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from logging import debug, warn
from random import random

# import simpleaudio as sa
from math import ceil
import numpy as np

from tools.audio_tools import read_audio_parts, get_mel_spec, Padding


class MultiLabelAudioSet(Dataset):
    def __init__(
        self,
        config,
        data: DataFrame,
        class_dict: dict,
        transform_image: Callable = None,
        transform_signal: Callable = None,
        is_validation: bool = False,
    ):
        """
        Args:
            data_rows : Path to the csv file with annotations.
            data_path (string): Directory of database with derivate directory in it. If none path in csv is used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.class_dict = class_dict
        self.transform_signal = transform_signal
        self.transform_image = transform_image
        self.data = data
        self.is_validation = is_validation
        self.annotation_interval_dict = {}
        annotation_interval_id = 0
        for _, row in data.iterrows():

            if row["class_id"] == "annotation_interval":
                annotation_interval_id += 1
                self.annotation_interval_dict[annotation_interval_id] = {
                    "file_id": row["file_id"],
                    "filepath": Path(config.data.data_root_path).joinpath(
                        row["filepath"]
                    ),
                    "channel_count": int(row["channel_count"]),
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "events": [],
                }
            else:
                self.annotation_interval_dict[annotation_interval_id]["events"].append(
                    {
                        "class_tensor": class_dict[row["class_id"]],
                        "start_time": float(row["start_time"]),
                        "end_time": float(row["end_time"]),
                        "type": row["type"],
                    }
                )

        # create  segments list of annoations intervals
        self.segments = []
        for key in self.annotation_interval_dict:

            annotation_interval = self.annotation_interval_dict[key]
            # workround through away all to short files
            # if(annotation_interval["end_time"] - annotation_interval["start_time"] < self.config.data.segment_duration):
            #     continue
            # validation has different step size then training
            segment_step = (
                self.config.validation.segment_step
                if self.is_validation
                else self.config.data.segment_duration
            )
            # calculate how many segments can be in the annotation intervall
            # ceil means last one is may be longer then the annotation_intervall
            segment_count = ceil(
                (
                    annotation_interval["end_time"]
                    - (
                        annotation_interval["start_time"]
                        + self.config.data.segment_duration
                    )
                )
                / segment_step
                + 1
            )

            for i in range(segment_count):

                start_time = annotation_interval["start_time"] + i * segment_step
                end_time = start_time + self.config.data.segment_duration
                # prevent wraparound in validation
                if self.is_validation and end_time > annotation_interval["end_time"]:
                    end_time = annotation_interval["end_time"]
                    start_time = end_time - self.config.data.segment_duration
                    if start_time < 0:
                        start_time = 0
                self.segments.append(
                    {
                        "start_time": start_time,
                        "annotation_interval": annotation_interval,
                        "end_time": end_time,
                    }
                )

    def __get_zero_tensor(self):
        x = list(self.class_dict.values())[0]
        return torch.zeros(x.size())

    def __len__(self):
        return len(self.segments)

    def __mapToClassIndex__(self, index):
        return self.class_dict[index]

    def __get_segment_parts(self, start_time, end_time, annotation_interval, parts):
        duration = end_time - start_time
        # if duration of this segment part is below min value do not add it
        if duration < self.config.data.min_event_overlap_time:

            return parts
        # start is in annoation interval
        if (
            start_time >= annotation_interval["start_time"]
            and start_time <= annotation_interval["end_time"]
        ):
            if end_time <= annotation_interval["end_time"]:

                # end_time is in annoation_interval add it to parts and return all parts
                parts.append((start_time, end_time))
                return parts
            else:

                sub_part_duration = annotation_interval["end_time"] - start_time

                # append part until end of annoation_interval if it is long enough
                if sub_part_duration >= self.config.data.min_event_overlap_time:
                    parts.append((start_time, annotation_interval["end_time"]))
                # create more parts of rest duration by a wrap around
                return self.__get_segment_parts(
                    annotation_interval["start_time"],
                    annotation_interval["start_time"] + duration - sub_part_duration,
                    annotation_interval,
                    parts,
                )
        else:  # it starts not in the annoation interval so wrap around
            new_start_time = (
                annotation_interval["start_time"]
                + start_time
                - annotation_interval["end_time"]
            )

            return self.__get_segment_parts(
                new_start_time, new_start_time + duration, annotation_interval, parts,
            )

    def __filter_in_segment_factory_(self, segment_parts):
        # there are more then one part in segment because there can be a wrap around in the annoation interval

        def is_in_filter(event):
            for (segment_start, segment_end) in segment_parts:
                if segment_end < event["start_time"]:
                    # event start after segment part ends
                    continue
                if event["end_time"] < segment_start:
                    # event ends before segment part starts
                    continue
                if (
                    event["start_time"] <= segment_start
                    and event["end_time"] >= segment_end
                ):
                    # event starts before the segment part and  ends after segment part
                    return True

                if (
                    event["start_time"] >= segment_start
                    and event["end_time"] <= segment_end
                ):
                    # event starts in the segment and ends in the segment
                    return True

                if (
                    event["start_time"] >= segment_start
                    and segment_end - event["start_time"]
                    >= self.config.data.min_event_overlap_time
                ):
                    # event starts in the segment and is it's occurrence is long enough
                    return True

                if (
                    event["end_time"] <= segment_end
                    and event["end_time"] - segment_start
                    >= self.config.data.min_event_overlap_time
                ):
                    # event ends in segment and is it's occurrence is long enough
                    return True
            return False

        return is_in_filter

    def __get_class_tensor__(self, annoation_interval, segment_parts):
        events = list(
            filter(
                self.__filter_in_segment_factory_(segment_parts),
                annoation_interval["events"],
            )
        )

        result = self.__get_zero_tensor()

        for t in events:
            result = result + t["class_tensor"]
        # reduce values > 1 to one
        result[result > 0] = 1
        return result

    def __getitem__(self, index):
        segment = self.segments[index]
        annotation_interval = segment["annotation_interval"]
        start_time = segment["start_time"]
        filepath = annotation_interval["filepath"]

        if self.is_validation is False:
            start_time += random() * self.config.data.segment_duration

        end_time = start_time + self.config.data.segment_duration
        segment_parts = self.__get_segment_parts(
            start_time, end_time, annotation_interval, []
        )

        class_tensor = self.__get_class_tensor__(annotation_interval, segment_parts)

        # print("get item channel: {}".format(self.data_rows[index][5]))
        audio_data = None
        try:
            # print("Read audio parts filepath: {} in mode {}".format(filepath,self.config.audio_loading.channel_mixing_strategy))
            audio_data = read_audio_parts(
                filepath,
                segment_parts,
                self.config.data.segment_duration,
                self.config.audio_loading.sample_rate,
                channel_mixing_strategy=self.config.audio_loading.channel_mixing_strategy,
                backend=self.config.audio_loading.backend,
                padding_strategy=Padding.SILENCE
                if self.is_validation
                else self.config.audio_loading.padding_strategy,
                randomize_audio_segment=False if self.is_validation else True,
            )
        except Exception as error:
            print(error)
            return None

        # print("Done reading index {} shape {}".format(index,audio_data[0,:]))

        tensor_list = []
        y_list = []
        index_list = []

        for channel in range(audio_data.shape[0]):
            augmented_signal, y = (
                self.transform_signal(
                    samples=audio_data[channel, :],
                    sample_rate=self.config.audio_loading.sample_rate,
                    y=class_tensor,
                )
                if self.transform_signal is not None
                else (audio_data[channel, :], class_tensor)
            )
            debug("Done signal augmenting index {} shape".format(index))
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
            image_data = np.empty((h, w, 1), dtype=np.uint8)
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

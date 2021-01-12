import argparse
from typing import Callable
from pandas import DataFrame
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


class AudioSet(Dataset):
    """Face Landmarks dataset."""

    root_dir: Path

    def __init__(
        self,
        config: ScriptConfig,
        raw_data_rows: list,
        class_dict: dict,
        extract_complete_segment: bool = False,
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
        v: ValidationConfig = config.validation

        self.class_dict = class_dict
        self.data_path = d.data_path
        self.index_start_time = d.index_start_time
        self.index_end_time = d.index_end_time
        self.index_filepath = d.index_filepath
        self.index_label = d.index_label

        self.transform_audio = transform_audio
        self.transform_image = transform_image

        self.randomize_audio_segment = randomize_audio_segment
        self.segment_length = a.segment_length
        self.sample_rate = a.sample_rate
        self.mixing_strategy = a.mixing_strategy
        self.padding_strategy = a.padding_strategy

        self.fft_size_in_samples = a.fft_size_in_samples
        self.fft_hop_size_in_samples = a.fft_hop_size_in_samples
        self.num_of_mel_bands = a.num_of_mel_bands
        self.mel_start_freq = a.mel_start_freq
        self.mel_end_freq = a.mel_end_freq

        tmp_data_rows = list(
            zip(
                list(range(len(raw_data_rows))),
                raw_data_rows.iloc[:, self.index_filepath],
                raw_data_rows.iloc[:, self.index_label],
                raw_data_rows.iloc[:, self.index_start_time,],
                raw_data_rows.iloc[:, self.index_end_time,],
            )
        )
        if extract_complete_segment:
            self.data_rows = []
            overlap_time = v.step_overlap * a.segment_length
            hop_length = a.segment_length - overlap_time
            for data in tmp_data_rows:
                id, filepath, label, start, end = data
                duration = end - start
                # add steps read complete file
                steps_needed = ceil(duration / hop_length)
                if steps_needed == 1:
                    self.data_rows.append(data)
                else:

                    for step in range(steps_needed - 1):
                        step_start = step * hop_length
                        self.data_rows.append(
                            (
                                id,
                                filepath,
                                label,
                                step_start,
                                step_start + a.segment_length,
                            )
                        )
                    # add last step only if length is half duration, prevent to short part of segment
                    step_data = (
                        id,
                        filepath,
                        label,
                        (steps_needed - 1) * hop_length,
                        end,
                    )
                    if step_data[4] - step_data[3] > a.segment_length / 2:
                        self.data_rows.append(step_data)
                pass

        else:
            self.data_rows = tmp_data_rows

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        segment_id = self.data_rows[idx][0]
        filepath = Path(self.data_rows[idx][1])
        label = self.data_rows[idx][2]
        if self.data_path is not None:
            filepath = self.data_path.joinpath(
                *filepath.parts[len(filepath.parts) - 6 :]
            )

        start = self.data_rows[idx][3]
        stop = self.data_rows[idx][4]
        # print(filepath.as_posix())
        audio_data = read_audio_segment(
            filepath,
            start,
            stop,
            self.segment_length,
            self.sample_rate,
            mixing_strategy=self.mixing_strategy,
            padding_strategy=self.padding_strategy,
            randomize_audio_segment=self.randomize_audio_segment,
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
            else image_data
        )
        tensor = transforms.ToTensor()(augmented_image_data).float()
        # plt.imshow(augmented_image_data, interpolation="nearest")
        # plt.show()
        label_id = self.class_dict[label]
        return tensor, label_id, segment_id

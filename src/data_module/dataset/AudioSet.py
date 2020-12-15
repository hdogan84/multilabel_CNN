import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import simpleaudio as sa
import numpy as np
import librosa
import librosa.display
from data_module.dataset.audio_tools import (
    read_audio_segment,
    get_mel_spec,
    Mixing,
    Padding,
)
from data_module.dataset.plot_tools import print_mel_spec


class AudioSet(Dataset):
    """Face Landmarks dataset."""

    root_dir: Path

    def __init__(
        self,
        csv_file: str,
        data_dir: str,
        # read audio parameters
        segment_length: int = 5,
        sample_rate: int = 32000,
        mixing_strategy=Mixing.TAKE_FIRST,
        padding_strategy=Padding.CYCLIC,
        # augmentation methods
        transform_signal=None,
        transform_image=None,
        # data index
        index_filepath: int = 5,
        index_start_time: int = 1,
        index_end_time: int = 2,
        index_label: int = 3,
        # mel_spec parameters
        fft_size_in_samples: int = 1536,
        fft_hop_size_in_samples: int = 360,
        num_of_mel_bands: int = 128,
        mel_start_freq: int = 20,
        mel_end_freq: int = 16000,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory of database withr derivate directory in it. If none path in csv is used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_rows = pd.read_csv(csv_file, delimiter=";", quotechar="|",)
        self.data_dir = Path(data_dir) if data_dir is not None else None

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.mixing_strategy = mixing_strategy
        self.padding_strategy = padding_strategy

        self.transform_signal = transform_signal
        self.transform_image = transform_image

        self.index_start_time = index_start_time
        self.index_end_time = index_end_time
        self.index_filepath = index_filepath
        self.index_label = index_label

        self.fft_size_in_samples = fft_size_in_samples
        self.fft_hop_size_in_samples = fft_hop_size_in_samples
        self.num_of_mel_bands = num_of_mel_bands
        self.mel_start_freq = mel_start_freq
        self.mel_end_freq = mel_end_freq

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        filepath = Path(self.data_rows.iloc[idx, self.index_filepath])
        if self.data_dir is not None:
            filepath = self.data_dir.joinpath(
                *filepath.parts[len(filepath.parts) - 6 :]
            )

        start = self.data_rows.iloc[idx, self.index_start_time]
        stop = self.data_rows.iloc[idx, self.index_end_time]
        print(filepath.as_posix())
        audio_data = read_audio_segment(
            filepath,
            start,
            stop,
            self.segment_length,
            self.sample_rate,
            mixing_strategy=self.mixing_strategy,
            padding_strategy=self.padding_strategy,
        )
        print(len(audio_data))
        if self.transform_signal is not None:
            raise NotImplementedError()

        mel_spec = get_mel_spec(
            audio_data,
            self.fft_size_in_samples,
            self.fft_hop_size_in_samples,
            self.sample_rate,
            num_of_mel_bands=self.num_of_mel_bands,
            mel_start_freq=self.mel_start_freq,
            mel_end_freq=self.mel_end_freq,
        )

        if self.transform_image is not None:
            raise NotImplementedError()

        # if(self.data_dir is not None):

        # image = io.imread(img_name)
        # landmarks = self.data_row.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype("float").reshape(-1, 2)
        # sample = {"image": image, "landmarks": landmarks}

        # if self.transform:
        #     sample = self.transform(sample)

        print_mel_spec(mel_spec, self.sample_rate, self.fft_hop_size_in_samples)
        plt.show()
        return self.data_rows.iloc[idx]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="")
    # parser.add_argument(
    #     "--filename",
    #     metavar="string",
    #     type=str,
    #     nargs="?",
    #     default="labels.csv",
    #     help="target filename for label csv",
    # )
    # parser.add_argument(
    #     "--config",
    #     metavar="path",
    #     type=Path,
    #     nargs="?",
    #     default=CONFIG_FILE_PATH,
    #     help="config file with database credentials",
    # )
    # args = parser.parse_args()
    audio_set = AudioSet(
        csv_file="./data/ammod-selection/labels.csv",
        data_dir="./data/ammod-selection/database",
    )
    fig = plt.figure()

    for i in range(10):  # range(len(audio_set)):
        sample = audio_set[i + 100]

        # print(sample)
        # print(i, sample['image'].shape, sample['landmarks'].shape)

        # ax = plt.subplot(1, 4, i + 1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        # show_landmarks(**sample)

        # if i == 3:
        #     plt.show()
        #     break

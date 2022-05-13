import pandas as pd
from torch import nn
from torch.utils import data
from tools.config import load_yaml_config
import argparse
from torchmetrics import Accuracy, AveragePrecision, F1, AUC
from tools.tensor_helpers import pool_by_segments
from pathlib import Path
from torch.utils.data import DataLoader

# from data_module.ColorSpecAmmodMultiLabelModule import ColorSpecAmmodMultiLabelModule as DataModule
from dataset.FileListAnalyseSet import FileListAnalyseSet
from tools.RunBaseTorchScriptModel import RunBaseTorchScriptModel
from augmentation.signal import ExtendedCompose as SignalCompose, create_signal_pipeline

from tools.collate_functions import filter_none_values, collate_channel_dimension

def collate_fn(batch):
    return collate_channel_dimension(filter_none_values(batch))

import albumentations as A

device = "cuda"
model_filepath = (
    "data/torchserve-models/raw/birdId-europe-254-2103/birdId-europe-254-2103.pt"
)
index_to_name_json_path = (
    "data/torchserve-models/raw/birdId-europe-254-2103/index_to_name.json"  #
)
config_path = "./config/birdId-europ-254.yaml"
output_path = "predictions.csv"
analyse_filepath = (
    "data/local/ammod2021/derivation/1/0/0/0/1ff6a294d5184c2eba8bb2fcf2713796.wav"
)


def analyse(model_filepath, config_filepath, filepath):
    class RunBirdDectector(RunBaseTorchScriptModel):
        def setup_transformations(self):
            NumOfLowFreqsInPixelToCutMax = 4
            NumOfHighFreqsInPixelToCutMax = 6
            imageHeight = 224
            resizeFactor = imageHeight / (
                self.config.audio_loading.num_of_mel_bands
                - NumOfLowFreqsInPixelToCutMax / 2.0
                - NumOfHighFreqsInPixelToCutMax / 2.0
            )
            imageWidth = int(
                resizeFactor
                * self.config.data.segment_duration
                * self.config.audio_loading.sample_rate
                / self.config.audio_loading.fft_hop_size_in_samples
            )
            print("Imagewidth {}".format(imageWidth))

            transform_signal = SignalCompose(
                create_signal_pipeline(
                    self.config.validation.signal_pipeline, self.config
                ),
                shuffle=False,
            )
            transform_image = A.Compose(
                [
                    A.Resize(
                        imageHeight, imageWidth, A.cv2.INTER_LANCZOS4, always_apply=True
                    )
                ]
            )
            return transform_signal, transform_image

        def setup_dataloader(self, transform_signal, transform_image):

            dataset = FileListAnalyseSet(
                self.config,
                [filepath],
                transform_signal=transform_signal,
                transform_image=transform_image,
            )
            data_list = dataset.segments

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.data.batch_size
                * self.config.validation.batch_size_mulitplier,
                shuffle=False,
                collate_fn=collate_fn
            )

            return dataloader, data_list, 0, []

    runBirdDetector = RunBirdDectector(
        model_filepath,
        config_filepath,
        validation=False,
        result_file=True,
        result_filepath="predictions.csv",
        device="cuda:0",
    )

    runBirdDetector.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        metavar="path",
        type=Path,
        nargs="?",
        default=config_path,
        help="config file for all settings",
    )
    parser.add_argument(
        "--file",
        metavar="path",
        type=Path,
        nargs="?",
        default=analyse_filepath,
        help="config file for all settings",
    )

    args = parser.parse_args()
    config_filepath = args.config
    filepath = args.file

    analyse(model_filepath, config_filepath, filepath)


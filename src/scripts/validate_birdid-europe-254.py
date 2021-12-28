import argparse
from pathlib import Path
from tools.data_logger import ResultLogger

# from data_module.ColorSpecAmmodMultiLabelModule import ColorSpecAmmodMultiLabelModule as DataModule
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule as DataModule
from tools.config import load_class_list_from_index_to_name_json
from tools.tensor_helpers import get_class_tensor_transformation_matrix
from tools.RunBaseTorchScriptModel import RunBaseTorchScriptModel
from augmentation.signal import ExtendedCompose as SignalCompose, create_signal_pipeline

import albumentations as A

device = "cuda:2"
model_filepath = (
    "data/torchserve-models/raw/birdId-europe-254-2103/birdId-europe-254-2103.pt"
)
index_to_name_json_path = (
    "data/torchserve-models/raw/birdId-europe-254-2103/index_to_name.json"  #
)
config_path = "./config/birdId-europe-254.yaml"
output_path = "predictions.csv"

import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def validate(config_filepath, model_filepath):
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
            print("Imagewitdh {}".format(imageWidth))

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
            data_module = DataModule(
                self.config, None, None, transform_signal, transform_image,
            )
            data_module.setup("test")
            num_classes = data_module.class_count
            data_loader = data_module.test_dataloader()
            data_set = data_module.test_set
            data_list = [
                {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "filepath": segment["annotation_interval"]["filepath"],
                }
                for segment in data_set.segments
            ]
            class_list = [key for key in data_module.class_dict]
            self.batch_end_logger = ResultLogger(
                data_list,
                model_name="birdid-europe-254",
                version="1",
                output_path="./{}".format(
                    Path(self.config["data"]["test_list_filepath"]).stem
                ),
                class_list=class_list,
            )
            return data_loader, data_list, num_classes, class_list

        def batch_end(self, prediction, ground_truth, segment_indices, batch_index):
            self.batch_end_logger.log_batch(
                prediction, ground_truth, segment_indices, batch_index
            )

        def run_end(self):
            self.batch_end_logger.write_to_json()

        def setup_class_tensor_transform_matrix(self, data_class_list):
            model_class_list = load_class_list_from_index_to_name_json(
                index_to_name_json_path
            )
            class_tensor_transformation_matrix = get_class_tensor_transformation_matrix(
                model_class_list, data_class_list
            ).to(device)
            return class_tensor_transformation_matrix

    runBirdDetector = RunBirdDectector(
        config_filepath, model_filepath, validation=True, device=device,
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

    args = parser.parse_args()
    config_filepath = args.config

    validate(model_filepath, config_filepath)


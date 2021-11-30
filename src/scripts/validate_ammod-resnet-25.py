import argparse
from pathlib import Path

from torch.utils import data

# from data_module.ColorSpecAmmodMultiLabelModule import ColorSpecAmmodMultiLabelModule as DataModule
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule as DataModule
from tools.RunBaseTorchScriptModel import RunBaseTorchScriptModel

device = "cuda:1"
model_filepath = "/home/tsa/projects/bewr/ammod-bird-detector/data/torchserve-models/raw/ammod-resnet-25-1/ammod-resnet-25-1.pt"
config_path = "./config/resnet_multi_label.yaml"
# trainer = Trainer(tpu_cores=8)


def validate(config_filepath, model_filepath):
    class RunBirdDectector(RunBaseTorchScriptModel):
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
                    "channel": segment["channel"],
                }
                for segment in data_set.segments
            ]
            class_list = [key for key in data_module.class_dict]
            return data_loader, data_list, num_classes, class_list

    runBirdDetector = RunBirdDectector(
        config_filepath,
        model_filepath,
        validation=True,
        result_file=True,
        result_filepath="predictions.csv",
        device=device,
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


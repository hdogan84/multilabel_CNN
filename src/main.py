import pytorch_lightning as pl
import argparse
from pathlib import Path

from config.configuration import parse_config, ScriptConfig
from model.CnnBirdDetector import CnnBirdDetector
from data_module.AmmodSingleLabelModule import AmmodSingleLabelModule

# nd without changing a single line of code, you could run on GPUs/TPUs
# 8 GPUs
# trainer = Trainer(max_epochs=1, gpus=8)
# 256 GPUs
# trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)

# TPUs
# trainer = Trainer(tpu_cores=8)
def start_train(config: ScriptConfig):
    data_module = AmmodSingleLabelModule(config)
    model = CnnBirdDetector()
    trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=20)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        metavar="path",
        type=Path,
        nargs="?",
        default="./src/config/default.cfg",
        help="config file for all settings",
    )
    parser.add_argument(
        "--data_dir",
        metavar="path",
        type=Path,
        nargs="?",
        help="ovewrite data directory",
    )
    parser.add_argument(
        "--data_list",
        metavar="path",
        type=Path,
        nargs="?",
        help="ovewrite data path to data csv",
    )
    parser.add_argument(
        "--class_list",
        metavar="path",
        type=Path,
        nargs="?",
        help="overwrite class list",
    )
    args = parser.parse_args()
    config_filepath = args.config
    data_dir = args.data_dir
    data_list_filepath = args.data_list
    class_list_filepath = args.class_list

    config = parse_config(config_filepath)
    if data_dir is not None:
        config.data.data_dir = data_dir
    if class_list_filepath is not None:
        config.data.class_list_filepath = class_list_filepath
    if data_list_filepath is not None:
        config.data.data_list_filepath = data_list_filepath

    start_train(config)

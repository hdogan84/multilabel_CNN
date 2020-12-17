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
        "--env", metavar="path", type=str, nargs="?", help="Environment Var Prefix",
    )

    args = parser.parse_args()
    config_filepath = args.config
    print(args.env)
    config = parse_config(config_filepath, enviroment_prefix=args.env)
    start_train(config)

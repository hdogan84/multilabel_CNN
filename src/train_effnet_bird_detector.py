import pytorch_lightning as pl
from tools.config import load_yaml_config
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from pathlib import Path
from model.CnnBirdDetectorEffNet import CnnBirdDetectorEffNet
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule
from pytorch_lightning import loggers as pl_loggers
from augmentation.signal import ExtendedCompose as SignalCompose, create_signal_pipeline
from tools.lighning_callbacks import (
    SaveConfigToLogs,
    LogFirstBatchAsImage,
    SaveFileToLogs,
)
from pprint import pprint
from logging import debug, warn

from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A

# nd without changing a single line of code, you could run on GPUs/TPUs
# 8 GPUs
# trainer = Trainer(max_epochs=1, gpus=8)
# 256 GPUs
# trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)

# TPUs
# trainer = Trainer(tpu_cores=8)
def start_train(config_filepath, checkpoint_filepath: Path = None):
    config = load_yaml_config(config_filepath)
    fit_transform_audio = SignalCompose(
        create_signal_pipeline(config.augmentation.signal_pipeline, config),
        shuffle=config.augmentation.shuffle_signal_augmentation,
    )
    fit_transform_image = A.Compose(
        [
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                brightness_by_max=True,
                always_apply=False,
                p=0.2,
            ),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.1),
        ]
    )
    data_module = AmmodMultiLabelModule(
        config,
        fit_transform_audio=fit_transform_audio,
        fit_transform_image=fit_transform_image,
    )

    if checkpoint_filepath is None:
        model = CnnBirdDetectorEffNet(data_module.class_count, **config.optimizer)
    else:
        # LOAD CHECKPOINT
        model = CnnBirdDetectorEffNet.load_from_checkpoint(
            checkpoint_filepath.as_posix(), **config.optimizer
        )
    tb_logger = pl_loggers.TensorBoardLogger(
        config.system.log_dir, name=config.system.experiment_name
    )
    # dic = {"brand": "Ford", "model": "Mustang", "year": 1964}

    # Setup Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        save_top_k=3,
        mode="max",
        filename="{val_f1:.2f}-{epoch:002d}",
    )
    checkpoint_callback2 = ModelCheckpoint(
        monitor="val_accuracy",
        save_top_k=3,
        mode="max",
        filename="{val_accuracy:.2f}-{epoch:002d}",
    )
    pl.seed_everything(config.system.random_seed)

    trainer = pl.Trainer(
        gpus=[0], # config.system.gpus,
        max_epochs=config.system.max_epochs,
        progress_bar_refresh_rate=config.system.log_every_n_steps,
        logger=tb_logger,
        log_every_n_steps=config.system.log_every_n_steps,
        deterministic=config.system.deterministic,
        callbacks=[
            # checkpoint_callback,
            # checkpoint_callback2,
            SaveConfigToLogs(config, config_filepath),
            SaveFileToLogs(config),
            LogFirstBatchAsImage(mean=0.456, std=0.224),
        ],
        check_val_every_n_epoch=config.validation.check_val_every_n_epoch,
        accelerator="ddp",
        auto_select_gpus=config.system.auto_select_gpus,
        fast_dev_run=config.system.fast_dev_run,
        # Debugging Settings
        # profiler="simple",
        # precision=16,
        # auto_scale_batch_size="binsearch",
        # limit_train_batches=0.01,
        # limit_val_batches=0.25,
        # overfit_batches=10,
    )
    # trainer.tune(model, data_module)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        metavar="path",
        type=Path,
        nargs="?",
        default="./config/ammod_multi_label.yaml",
        help="config file for all settings",
    )
    parser.add_argument(
        "--env", metavar="path", type=str, nargs="?", help="Environment Var Prefix",
    )
    parser.add_argument(
        "--load", metavar="load", type=Path, nargs="?", help="Load model load",
    )

    args = parser.parse_args()
    config_filepath = args.config
    print(args.env)

    start_train(
        config_filepath, checkpoint_filepath=args.load,
    )


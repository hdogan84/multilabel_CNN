import pytorch_lightning as pl
import argparse
from pathlib import Path
from config.configuration import parse_config, ScriptConfig
from model.CnnBirdDetector import CnnBirdDetector
from data_module.AmmodSingleLabelModule import AmmodSingleLabelModule
from pytorch_lightning import loggers as pl_loggers
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from augmentation.AddBackgroundNoiseFromCsv import AddBackgroundNoiseFromCsv
import albumentations as A

# nd without changing a single line of code, you could run on GPUs/TPUs
# 8 GPUs
# trainer = Trainer(max_epochs=1, gpus=8)
# 256 GPUs
# trainer = Trainer(max_epochs=1, gpus=8, num_nodes=32)

# TPUs
# trainer = Trainer(tpu_cores=8)
def start_train(config: ScriptConfig):

    fit_transform_audio = Compose(
        [
            AddBackgroundNoiseFromCsv(
                "./data/ammod-selection/noise_3000.csv",
                min_snr_in_db=4,
                max_snr_in_db=20,
                p=0.5,
            ),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.25),
            TimeStretch(min_rate=0.9, max_rate=1.10, p=0.25),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.25),
        ]
    )
    fit_transform_image = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.VerticalFlip(p=0.5),
        ]
    )

    data_module = AmmodSingleLabelModule(
        config,
        fit_transform_audio=fit_transform_audio,
        fit_transform_image=fit_transform_image,
    )

    model = CnnBirdDetector(data_module.class_count)
    # LOAD CHECKPOINT

    # model = CnnBirdDetector.load_from_checkpoint(
    #     "./logs/audio_classificator/version_1/checkpoints/epoch=4.ckpt",
    #     data_module.class_count,
    # )
    tb_logger = pl_loggers.TensorBoardLogger(
        config.system.log_dir, name=config.learning.expriment_name
    )

    # save Model with best val_loss
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss",)

    pl.seed_everything(config.system.random_seed)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.learning.max_epochs,
        progress_bar_refresh_rate=10,
        logger=tb_logger,
        log_every_n_steps=config.system.log_every_n_steps,
        deterministic=True,
        # callbacks=[checkpoint_callback],
    )
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

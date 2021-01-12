import pytorch_lightning as pl
import argparse
from pathlib import Path
from config.configuration import parse_config, ScriptConfig
from model.CnnBirdDetector import CnnBirdDetector
from data_module.AmmodSingleLabelModule import AmmodSingleLabelModule
from pytorch_lightning import loggers as pl_loggers
from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    TimeMask,
    FrequencyMask,
)
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
    fit_transform_audio = None
    fit_transform_image = None
    fit_transform_audio = Compose(
        [
            TimeMask(
                min_band_part=config.time_mask.min_band_part,
                max_band_part=config.time_mask.max_band_part,
                fade=config.time_mask.fade,
                p=config.time_mask.p,
            ),
            FrequencyMask(min_frequency_band=0.01, max_frequency_band=0.05, p=0.1),
            AddBackgroundNoiseFromCsv(
                config.add_background_noise_from_csv.filepath,
                min_snr_in_db=config.add_background_noise_from_csv.min_snr_in_db,
                max_snr_in_db=config.add_background_noise_from_csv.max_snr_in_db,
                index_filepath=config.add_background_noise_from_csv.index_filepath,
                delimiter=config.add_background_noise_from_csv.delimiter,
                quotechar=config.add_background_noise_from_csv.quotechar,
                p=config.add_background_noise_from_csv.p,
            ),
            AddGaussianNoise(
                min_amplitude=config.add_gaussian_noise.min_amplitude,
                max_amplitude=config.add_gaussian_noise.max_amplitude,
                p=config.add_gaussian_noise.p,
            ),
            TimeStretch(
                min_rate=config.time_strech.min_rate,
                max_rate=config.time_strech.max_rate,
                leave_length_unchanged=config.time_strech.leave_length_unchanged,
                p=config.time_strech.p,
            ),
            PitchShift(
                min_semitones=config.pitch_shift.min_semitones,
                max_semitones=config.pitch_shift.max_semitones,
                p=config.pitch_shift.p,
            ),
            Shift(
                min_fraction=config.shift.min_fraction,
                max_fraction=config.shift.max_fraction,
                rollover=config.shift.rollover,
                p=config.shift.p,
            ),
        ]
    )
    fit_transform_image = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
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
    data_module = AmmodSingleLabelModule(
        config,
        fit_transform_audio=fit_transform_audio,
        fit_transform_image=fit_transform_image,
    )

    model = CnnBirdDetector(
        data_module.class_count,
        learning_rate=config.learning.learning_rate,
        optimizer_type=config.learning.optimizer_type,
        sgd_momentum=config.learning.sgd_momentum,
        sgd_weight_decay=config.learning.sgd_weight_decay,
        scheduler_type=config.learning.scheduler_type,
        cosine_annealing_lr_t_max=config.learning.cosine_annealing_lr_t_max,
    )
    # LOAD CHECKPOINT

    # model = CnnBirdDetector.load_from_checkpoint(
    #     "./logs/audio_classificator/version_1/checkpoints/epoch=4.ckpt",
    #     data_module.class_count,
    # )
    tb_logger = pl_loggers.TensorBoardLogger(
        config.system.log_dir, name=config.learning.experiment_name
    )
    # dic = {"brand": "Ford", "model": "Mustang", "year": 1964}

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
        # default="./src/config/europe254.cfg",
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

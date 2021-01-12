from typedconfig import Config, key, section, group_key
from typedconfig.source import EnvironmentConfigSource, IniFileConfigSource
from pathlib import Path


class DictConfig(Config):
    def as_dict(self):
        raw_dic = vars(self)
        if self._section_name is None:
            filterdItems = [
                a
                for a in raw_dic.items()
                if a[0] not in ["_section_name", "_cache", "_config_sources",]
            ]
            result_list = []
            for item in filterdItems:
                if isinstance(item[1], DictConfig):
                    result_list.append((item[0], item[1].as_dict()))
            dic = dict(result_list)
            return dic
        else:
            dic = raw_dic["_cache"][self._section_name]
            print(self._section_name)
            return dic


@section("data")
class DataConfig(DictConfig):
    class_list_filepath: Path = key(cast=Path)
    data_list_filepath: str = key(cast=str, required=False, default=None)
    train_list_filepath: str = key(cast=str, required=False, default=None)
    val_list_filepath: str = key(cast=str, required=False, default=None)
    test_list_filepath: str = key(cast=str, required=False, default=None)
    data_path: Path = key(cast=Path)
    index_filepath: int = key(cast=int)
    index_start_time: int = key(cast=int)
    index_end_time: int = key(cast=int)
    index_label: int = key(cast=int)
    test_split: float = key(cast=float, required=False, default=None)
    val_split: float = key(cast=float, required=False, default=None)


@section("system")
class SystemConfig(DictConfig):
    log_dir: str = key(cast=str)
    log_every_n_steps = key(cast=int)
    gpus: int = key(cast=int)
    num_workers: int = key(cast=int)
    random_seed: int = key(cast=int)


@section("learning")
class LearningConfig(DictConfig):
    experiment_name: str = key(cast=str)
    batch_size: int = key(cast=int)
    max_epochs: int = key(cast=int)
    learning_rate: float = key(cast=float)
    optimizer_type: str = key(cast=str, required=False, default=None)
    sgd_momentum: float = key(cast=float, required=False, default=0)
    sgd_weight_decay: float = key(cast=float, required=False, default=0)
    scheduler_type: str = key(cast=str, required=False, default=None)
    cosine_annealing_lr_t_max: float = key(cast=float, required=False, default=0)


@section("audio_loading")
class AudioLoadingConfig(DictConfig):
    segment_length: int = key(cast=int)
    sample_rate: int = key(cast=int)
    mixing_strategy: str = key(cast=str)
    padding_strategy: str = key(cast=str)
    fft_size_in_samples: int = key(cast=int)
    fft_hop_size_in_samples: int = key(cast=int)
    num_of_mel_bands: int = key(cast=int)
    mel_start_freq: int = key(cast=int)
    mel_end_freq: int = key(cast=int)


@section("TimeMask")
class TimeMask(DictConfig):
    min_band_part: float = key(cast=float)
    max_band_part: float = key(cast=float)
    fade: bool = key(cast=bool)
    p: float = key(cast=float, required=False, default=0.0)


# @section("FrequencyMask")
# class TimeMask(DictConfig):
#     min_frequency_band: float = key(cast=float)
#     max_frequency_band: float = key(cast=float)
#     p: float = key(cast=float, required=False, default=0.0)


@section("AddBackgroundNoiseFromCsv")
class AddBackgroundNoiseFromCsv(DictConfig):
    filepath: Path = key(cast=Path)
    index_filepath: int = key(cast=int, required=False, default=0)
    min_snr_in_db: int = key(cast=int)
    max_snr_in_db: int = key(cast=int)
    delimiter: str = key(cast=str, required=False, default=";")
    quotechar: str = key(cast=str, required=False, default="|")
    p: float = key(cast=float, required=False, default=0.0)


@section("AddGaussianNoise")
class AddGaussianNoise(DictConfig):
    min_amplitude: float = key(cast=float)
    max_amplitude: float = key(cast=float)
    p: float = key(cast=float, required=False, default=0.0)


@section("TimeStretch")
class TimeStretch(DictConfig):
    min_rate: float = key(cast=float)
    max_rate: float = key(cast=float)
    leave_length_unchanged: bool = key(cast=bool)
    p: float = key(cast=float, required=False, default=0.0)


@section("PitchShift")
class PitchShift(DictConfig):
    min_semitones: int = key(cast=int)
    max_semitones: int = key(cast=int)
    p: float = key(cast=float, required=False, default=0.0)


@section("Shift")
class Shift(DictConfig):
    min_fraction: float = key(cast=float)
    max_fraction: float = key(cast=float)
    rollover: bool = key(cast=bool)
    p: float = key(cast=float, required=False, default=0.0)


class ScriptConfig(DictConfig):
    data: DataConfig = group_key(DataConfig)
    system: SystemConfig = group_key(SystemConfig)
    learning: LearningConfig = group_key(LearningConfig)
    audio_loading: AudioLoadingConfig = group_key(AudioLoadingConfig)
    time_mask: TimeMask = group_key(TimeMask)
    add_background_noise_from_csv: AddBackgroundNoiseFromCsv = group_key(
        AddBackgroundNoiseFromCsv
    )
    add_gaussian_noise: AddGaussianNoise = group_key(AddGaussianNoise)
    time_strech: TimeStretch = group_key(TimeStretch)
    pitch_shift: PitchShift = group_key(PitchShift)
    shift: Shift = group_key(Shift)


def parse_config(config_filepath: Path, enviroment_prefix: str = None) -> ScriptConfig:
    if config_filepath.exists() is False:
        raise FileNotFoundError(config_filepath)
    config = ScriptConfig()

    if enviroment_prefix is not None:
        print(enviroment_prefix)
        config.add_source(EnvironmentConfigSource(prefix=enviroment_prefix))
    config.add_source(IniFileConfigSource(config_filepath))
    config.read()

    return config

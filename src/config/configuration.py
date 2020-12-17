from typedconfig import Config, key, section, group_key
from typedconfig.source import EnvironmentConfigSource, IniFileConfigSource
from pathlib import Path


@section("data")
class DataConfig(Config):
    class_list_filepath: Path = key(cast=Path)
    data_list_filepath: Path = key(cast=Path)
    data_path: Path = key(cast=Path)
    index_filepath: int = key(cast=int)
    index_start_time: int = key(cast=int)
    index_end_time: int = key(cast=int)
    index_label: int = key(cast=int)
    test_split: float = key(cast=float)
    val_split: float = key(cast=float)


@section("system")
class SystemConfig(Config):
    log_dir: str = key(cast=str)
    gpus: int = key(cast=int)
    num_workers: int = key(cast=int)
    random_seed: int = key(cast=int)


@section("learning")
class LearningConfig(Config):
    batch_size: int = key(cast=int)
    max_epochs: int = key(cast=int)


@section("audio_loading")
class AudioLoadingConfig(Config):
    segment_length: int = key(cast=int)
    sample_rate: int = key(cast=int)
    mixing_strategy: str = key(cast=str)
    padding_strategy: str = key(cast=str)
    fft_size_in_samples: int = key(cast=int)
    fft_hop_size_in_samples: int = key(cast=int)
    num_of_mel_bands: int = key(cast=int)
    mel_start_freq: int = key(cast=int)
    mel_end_freq: int = key(cast=int)


class ScriptConfig(Config):
    data: DataConfig = group_key(DataConfig)
    system: SystemConfig = group_key(SystemConfig)
    learning: LearningConfig = group_key(LearningConfig)
    audio_loading: AudioLoadingConfig = group_key(AudioLoadingConfig)


def parse_config(config_filepath: Path, enviroment_prefix: str = None) -> ScriptConfig:
    if config_filepath.exists() is False:
        raise FileNotFoundError(config_filepath)
    config = ScriptConfig()
    if enviroment_prefix is not None:
        config.add_source(EnvironmentConfigSource(prefix=enviroment_prefix))
    config.add_source(IniFileConfigSource(config_filepath))
    config.read()

    return config

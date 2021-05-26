from config.ModelConfig import ModelConfig


class Data(ModelConfig):
    class_list_filepath: str = "./data/libro_animalis/exported/ammod-multi-class-list.csv"
    train_list_filepath: str = "./data/libro_animalis/exported/ammod-multi-train.csv"
    val_list_filepath: str = "./data/libro_animalis/exported/ammod-multi-val.csv"
    data_root_path: str = "./data/libro_animalis/"
    batch_size: int = 182
    segment_duration: float = 5.0
    min_event_overlap_time: float = 0.2
    wrap_around_probability: float = 0.5


class System(ModelConfig):
    experiment_name: str = "audio_classificator_logging"
    log_dir: str = "./logs"
    log_every_n_steps: int = 5
    gpus: int = 1
    num_workers: int = -1
    random_seed: int = 236762
    deterministic: bool = True
    max_epochs: int = 230


class Optimizer(ModelConfig):
    # optimizers Adam SGD
    optimizer_type: str = "Adam"
    learning_rate: float = 0.0001
    sgd_momentum: float = 0.9
    sgd_weight_decay: float = 1e-4
    # schedulers None CosineAnnealingLR
    # scheduler_type = CosineAnnealingLR
    cosine_annealing_lr_t_max: float = 15


class Validation(ModelConfig):
    complete_segment: bool = True
    # multi channel handling take_first | take_all
    multi_channel_handling: str = "take_all"
    # maximum length of segment which get anlises
    # in seconds or None for whole segment
    max_segment_length: float = 9.0
    # handle to short las subsegment: drop | move_start
    sub_segment_rest_handling: str = "move_start"
    sub_segment_overlap: float = 0.1
    # poolin_methods: mean | meanexp | max
    pooling_method: str = "mean"
    batch_size_mulitplier: float = 3


class AudioLoading(ModelConfig):
    segment_length: bool = 5
    sample_rate: int = 32000
    # multi channel handling take_one | take_all | random_mix
    channel_mixing_strategy: str = "take_one"
    # clyclic | silence
    padding_strategy: str = "wrap_around"
    fft_size_in_samples: int = 1536
    fft_hop_size_in_samples: int = 360
    num_of_mel_bands: int = 128
    mel_start_freq: int = 20
    mel_end_freq: int = 16000


class Augmentation(ModelConfig):
    # TimeMask, FrequencyMask, AddBackgroundNoiseFromCsv, AddGaussianNoise,
    # TimeStretch, PitchShift, Shift, AddPinkNoiseSnr, VolumeControl, AddSameClassSignal
    shuffle_signal_augmentation: bool = False
    signal_pipeline: list = [
        "AddBackgroundNoiseFromCsv",
        # "AddClassSignal",
        "AddPinkNoiseSnr",
        "VolumeControl",
        "FrequencyMask",
        "TimeMask",
        # "TimeStretch",
        "AddGaussianNoise",
    ]
    # Augmentation Methods Signal


class TimeMask(ModelConfig):
    min_band_part: float = 0.05
    max_band_part: float = 0.5
    fade: bool = False
    p: float = 0.2


class FrequencyMask(ModelConfig):
    min_frequency_band: float = 0.05
    max_frequency_band: float = 0.5
    p: float = 0.1


class AddBackgroundNoiseFromCsv(ModelConfig):
    filepath: str = "./data/libro_animalis/exported/noise_3000.csv"
    data_path: str = "./data/libro_animalis/"
    index_filepath: int = 0
    min_snr_in_db: int = 4
    max_snr_in_db: int = 20
    delimiter: str = ";"
    quotechar: str = "|"
    p: float = 0.4


class AddGaussianNoise(ModelConfig):
    min_amplitude: float = 0.001
    max_amplitude: float = 0.015
    p: float = 0.15


class TimeStretch(ModelConfig):
    min_rate: float = 0.9
    max_rate: float = 1.10
    leave_length_unchanged: bool = True
    p: float = 0.20


class PitchShift(ModelConfig):
    min_semitones: int = -2
    max_semitones: int = 2
    p: float = 0.20


class Shift(ModelConfig):
    min_fraction: float = -0.5
    max_fraction: float = 0.5
    rollover: bool = True
    p: float = 0.5


class AddPinkNoiseSnr(ModelConfig):
    p: float = 0.2
    min_snr: float = 5.0
    max_snr: float = 20.0


class VolumeControl(ModelConfig):
    p: float = 0.2
    db_limit: float = 10
    # uniform fade cosine sine random
    mode: str = "cosine"


class AddSameClassSignal(ModelConfig):
    p: float = 0.3
    min_ssr = -20
    max_ssr = 3
    max_n = 3
    padding_strategy: str = "wrap_around"
    channel_mixing_strategy: str = "take_one"
    data_path: str = "./data/ammod-selection/database"
    data_list_filepath: str = "./data/libro_animalis/ammod-selection/train_balanced_labels.csv"
    # set class_list_filepath if you want to transform class into class_index values
    class_list_filepath: str = "./data/libro_animalis/ammod-selection/class-list.csv"
    index_filepath: int = 5
    index_start_time: int = 1
    index_end_time: int = 2
    index_label: int = 3
    index_channels: int = 6
    delimiter: str = ";"
    quotechar: str = "|"


class AddClassSignal(ModelConfig):
    p: float = 0.8
    restriced_to_same_class = False
    min_ssr = -20
    max_ssr = 3
    max_n = 5
    padding_strategy: str = "wrap_around"
    channel_mixing_strategy: str = "take_one"
    data_path: str = "./data/ammod-selection/database"
    data_list_filepath: str = "./data/ammod-selection/database/train_ammod_labels.csv"
    # set class_list_filepath if you want to transform class into class_index values
    class_list_filepath: str = "./data/"
    index_filepath: int = 5
    index_start_time: int = 1
    index_end_time: int = 2
    index_label: int = 3
    index_channels: int = 6
    delimiter: str = ";"
    quotechar: str = "|"


class Config(ModelConfig):
    data: Data = Data()
    system: System = System()
    optimizer: Optimizer = Optimizer()
    validation: Validation = Validation()
    audio_loading: AudioLoading = AudioLoading()
    augmentation: Augmentation = Augmentation()
    time_mask: TimeMask = TimeMask()
    frequency_mask: FrequencyMask = FrequencyMask()
    add_background_noise_from_csv: AddBackgroundNoiseFromCsv = AddBackgroundNoiseFromCsv()
    add_gaussian_noise: AddGaussianNoise = AddGaussianNoise()
    time_strech: TimeStretch = TimeStretch()
    pitch_shift: PitchShift = PitchShift()
    shift: Shift = Shift()
    add_pink_noise_snr: AddPinkNoiseSnr = AddPinkNoiseSnr()
    volume_control: VolumeControl = VolumeControl()
    add_class_signal: AddClassSignal = AddClassSignal()

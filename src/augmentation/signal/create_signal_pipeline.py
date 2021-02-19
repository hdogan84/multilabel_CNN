from inflection import underscore
from config.configuration import ScriptConfig
from augmentation.signal import (
    AddBackgroundNoiseFromCsv,
    AddPinkNoiseSnr,
    VolumeControl,
    AddSameClassSignal,
    AddClassSignal,
)

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AddImpulseResponse,
    AddShortNoises,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    LoudnessNormalization,
    Mp3Compression,
    Normalize,
    PitchShift,
    PolarityInversion,
    Resample,
    Shift,
    TimeMask,
    TimeStretch,
    Trim,
)


signal_transform_dict = {
    "AddGaussianNoise": AddGaussianNoise,
    "TimeStretch": TimeStretch,
    "PitchShift": PitchShift,
    "Shift": Shift,
    "TimeMask": TimeMask,
    "FrequencyMask": FrequencyMask,
    "AddBackgroundNoiseFromCsv": AddBackgroundNoiseFromCsv,
    "AddPinkNoiseSnr": AddPinkNoiseSnr,
    "VolumeControl": VolumeControl,
    "AddSameClassSignal": AddSameClassSignal,
    "AddClassSignal": AddClassSignal,
    "AddGaussianSNR": AddGaussianSNR,
    "AddBackgroundNoise": AddBackgroundNoise,
    "AddImpulseResponse": AddImpulseResponse,
    "AddShortNoises": AddShortNoises,
    "ClippingDistortion": ClippingDistortion,
    "Gain": Gain,
    "LoudnessNormalization": LoudnessNormalization,
    "Mp3Compression": Mp3Compression,
    "Normalize": Normalize,
    "PolarityInversion": PolarityInversion,
    "Resample": Resample,
    "Trim": Trim,
}

from tools.lighning_callbacks import SaveConfigToLogs, LogFirstBatchAsImage
from pprint import pprint

from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A


def create_signal_pipeline(transform_list: list, config: ScriptConfig):
    config_dict = config.as_dict()
    transforms = []
    for transform_name in transform_list:
        print("- {}".format(transform_name))
        transforms.append(
            signal_transform_dict[transform_name](**config_dict[transform_name])
        )
    return transforms

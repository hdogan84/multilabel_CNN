from augmentation.signal.AddBackgroundNoiseFromCsv import AddBackgroundNoiseFromCsv
from augmentation.signal.AddPinkNoiseSnr import AddPinkNoiseSnr
from augmentation.signal.VolumeControl import VolumeControl
from augmentation.signal.ExtendedCompose import ExtendedCompose
from augmentation.signal.AddSameClassSignal import AddSameClassSignal

# Import Audiomentations
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

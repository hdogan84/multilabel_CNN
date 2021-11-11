import numpy as np
from scipy.signal import butter, sosfilt
from audiomentations.core.transforms_interface import BaseWaveformTransform

class ButterFilter(BaseWaveformTransform):
    """
    Apply Butterworth in n order filtering to the input audio. 
    Wrapping of scipy signal butter filtering
    """

    supports_multichannel = False

    def __init__(self, cutoff_freq=20, order=2, filter_type="highpass", p: float = 0.5):
        """
        :param cutoff_freq: array_like The critical frequency or frequencies. For lowpass and highpass
            filters, Wn is a scalar; for bandpass and bandstop filters,
            Wn is a length-2 sequence.
        :param order: The order of the filter, 
        :filter_type: {'lowpass', 'highpass', 'bandpass', 'bandstop'}
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        self.cutoff_freq = cutoff_freq
        self.order = order
        self.filter_type = filter_type

    def randomize_parameters(self, samples: np.array, sample_rate: int = None):
        super().randomize_parameters(samples, sample_rate)

        self.parameters["sos"] = butter(
            self.order,
            self.cutoff_freq,
            btype=self.filter_type,
            output="sos",
            fs=sample_rate,
        )

    def apply(self, samples: np.array, sample_rate: int = None):
        print(samples.shape)
        audio_segment = np.zeros(samples.shape)
        for channel in range(samples.shape[1]):
            audio_segment[:,channel] = sosfilt(self.parameters["sos"], samples[:,channel], axis=0)
        if (
            np.isnan(audio_segment).any()
            or np.max(audio_segment) > 1.0
            or np.min(audio_segment) < -1.0
        ):
            print("Warning filter instability: filter not applied")
            audio_segment = samples
        return audio_segment


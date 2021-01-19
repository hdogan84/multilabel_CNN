import colorednoise as cn
import numpy as np
from audiomentations.augmentations.transforms import BaseWaveformTransform


class AddPinkNoiseSnr(BaseWaveformTransform):
    """Add pink noise to the samples with random Signal to Noise Ratio (SNR)"""

    def __init__(self, p=0.5, min_snr=5.0, max_snr=20.0):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        :param p:
        """
        super().__init__(p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["snr"] = np.random.uniform(self.min_snr, self.max_snr)

    def apply(self, samples, sample_rate):
        a_signal = np.sqrt(samples ** 2).max()
        a_noise = a_signal / (10 ** (self.parameters["snr"] / 20))
        pink_noise = cn.powerlaw_psd_gaussian(1, len(samples))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (samples + pink_noise * 1 / a_pink * a_noise).astype(samples.dtype)
        return augmented

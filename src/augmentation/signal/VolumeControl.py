import numpy as np
from audiomentations.augmentations.transforms import BaseWaveformTransform


class VolumeControl(BaseWaveformTransform):
    def __init__(self, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(p)
        assert mode in [
            "uniform",
            "fade",
            "cosine",
            "sine",
            "random",
        ], "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit = db_limit
        self.mode = mode

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["db"] = np.random.uniform(-self.db_limit, self.db_limit)
            std = np.std(samples)

    def apply(self, samples, sample_rate):
        db = self.parameters["db"]
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(samples))[::-1] / (len(samples) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(samples)) / len(samples) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(samples)) / len(samples) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = samples * db_translated
        return augmented

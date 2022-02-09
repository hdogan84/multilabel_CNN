from os import error
from pathlib import Path
from typing import Tuple
from librosa.core import audio
import librosa
import soundfile as sf
import numpy as np
import random
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from enum import Enum
import logging
from pydub.utils import mediainfo

filepath = [
    "/home/tsa/projects/bewr/ammod-bird-detector/15c67bf2d03040d089a288480b398dd0.mp3",
    "15c67bf2d03040d089a288480b398dd0.wav",
]

for i in filepath:
    y, sr = librosa.load(i, sr=None)
    print(librosa.get_duration(y, sr=sr))
    print(sr)
    print(mediainfo(i)["duration"])
    print(mediainfo(i)["sample_rate"])


# -*- coding: utf-8 -*-
# source activate /net/mfnstore-lin/export/tsa_transfer/PythonProjects/VirtualEnvs/BirdId18
from __future__ import print_function, division
import os
import numpy as np
import time
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

#LogDir = '00_TestRuns/'
#LogDir = './'

BatchSizeForTesting = 660

Use44100Hz = False
UseMelSpec = True

TrainLastFullyConnectedLayerOnly = 0

#####  Model Dependent Stuff  ####
ModelName = 'inception_v3'
InputSize = 299
ImageMean = [0.5, 0.5, 0.5]
ImageStd = [0.5, 0.5, 0.5]
UseInception = 1
UseCadenePretrainedModels = 0

NumOfClasses = 254 # Europe Birds

AlmostZero = 1.0/(2**64)
# #AlmostZero = 1.0/(2**32)

if Use44100Hz:
	SampleRate = 44100
	FftSizeInSamples = 3072
	FftHopSizeInSamples = 768
else:
	SampleRate = 22050
	FftSizeInSamples = 1536
	FftHopSizeInSamples = 360

SegmentDuration = 5.0

ImageSize = InputSize

# Mel Params
NumOfMelBands = 310
MelStartFreq = 160.0
MelEndFreq = 10300.0
NumOfLowFreqsInPixelToCutMax = 6
NumOfHighFreqsInPixelToCutMax = 10

#NormalizationModeTrain = 'LibrosaPowerToDb' #'SegmentBased' #'FileBased' 'Random' 'SegmentBased' 'LibrosaPowerToDb'
NormalizationModeVal = 'LibrosaPowerToDb' #'SegmentBased' # 'Random' 'SegmentBased'




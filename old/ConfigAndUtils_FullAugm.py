# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import pickle
import time
import random
import librosa
import soundfile as sf
import json
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image, ImageFile # plot and save Image
import matplotlib.pyplot as plt

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

from scipy.signal import butter, lfilter

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import skewnorm

from scipy.special import expit # sigmoid
from sklearn import metrics

import ast

# kaggle docker image
RootDir = '/tmp/working/'

#LogDir = RootDir + 'KaggleBirdsongRecognition2020/jobs/00_1_Test/'
LogDir = './'

RandomSeed = 0 #61 #1424 #61

TrainDeterministic = False # True False

BatchSizeForTesting = 128 #480 #660 #330 #660 (300? for 1x 1080 Ti)



NumOfEpochsBeforeFirstValidation = 0
NumOfEpochsBeforeNextValidation = 1 #4



SegmentOverlapForTestFilesInPerc = 10 #20 #10 #80 #10  # HopLength = Duration - Duration*SegmentOverlapForTestFilesInPerc*0.01 (e.g. 5-5*60*0.01=2)
ValidateOnlyBeginning = True # True False


ColorJitterAmount = 0.3 #0.2 #0.3


#####  Model Dependent Stuff  ####

# InceptionV3
#ModelName = 'inception_v3'
#ModelName = 'resnet50'
ModelName = 'efficientnet-b3'

# NNN
UseAdaptiveMaxPool2d = True # True False
IncludeSigmoid = False
SingleChannel = False # ToDo

LabelSmoothingFactor = 0.1 # None

OptimizerType = 'Adam' # Adam SGD
SchedulerType = 'CosineAnnealingLR' # None
T_max = 15 #20 #10


# SampleRate = 22050
# FftSizeInSamples = 1536
# FftHopSizeInSamples = 360

# Arai style
SampleRate = 32000
FftSizeInSamples = 2048
FftHopSizeInSamples = 512


SegmentDuration = 5.0
DurationJitter = 0.4 #0.2 #0 #0.45


NumOfIterationsToSaveCheckpointBetweenEpochs = 0 # 2000



AmpChangeChance = 20 # Not used ?
AmpExpMin = -2.5
AmpExpMax = 0.5



####  Dcase 2018 Noise  ####
DcaseNoiseChance01 =  90
DcaseNoiseAmpExpMin01 = -3.0
DcaseNoiseAmpExpMax01 = 0.25
DcaseNoiseChance02 = 70
DcaseNoiseAmpExpMin02 = -4.0
DcaseNoiseAmpExpMax02 = -1.0
DcaseNoiseChance03 = 30
DcaseNoiseAmpExpMin03 = -2.0
DcaseNoiseAmpExpMax03 = 0.5



AddSameClassChance01 = 40 #30 #40
AddSameClassAmpExpMin01 = -3.0
AddSameClassAmpExpMax01 = 0.5

AddSameClassChance02 = 40 #0 #40
AddSameClassAmpExpMin02 = -3.0
AddSameClassAmpExpMax02 = 0.5

AddSameClassChance03WithBs = 0
AddSameClassAmpExpMin03WithBs = -3.0
AddSameClassAmpExpMax03WithBs = 0.5

# Add audio from file with random class
AddFileWithRandomClass01 = 50 #30 #25 #40 #50
AddFileWithRandomClass02 = 40 #20 #30 #40
AddFileWithRandomClass03 = 30 #10 #0 #30
AddFileWithRandomClass04 = 20 #0 #20


FinalAmpChangeChance = 0
FinalAmpExpMin = -2.0
FinalAmpExpMax = 0.5




ExportPredictionDataDict = 0
SavePredictions = 0


VaryInterpolationFilterChance = 15



CyclicShiftChance = 20 #10 #20 #0 #20

CutSegmentChance = 30 #15 #30 #0 #30



SpectralTimeStretchOpt = {}
SpectralTimeStretchOpt['SegmWidthMin'] = 10
SpectralTimeStretchOpt['SegmWidthMax'] = 100
SpectralTimeStretchOpt['StretchFactorMin'] = 0.9 #0.95 #0.9
SpectralTimeStretchOpt['StretchFactorMax'] = 1.1 #1.05 #1.1

SpectralTimeStretchChance = 50 #25 #50 #0 #50

SpectralFreqStretchOpt = {}
SpectralFreqStretchOpt['SegmHeightMin'] = 10
SpectralFreqStretchOpt['SegmHeightMax'] = 100
SpectralFreqStretchOpt['StretchFactorMin'] = 0.95 #0.97 #0.95  
SpectralFreqStretchOpt['StretchFactorMax'] = 1.05 #1.03 #1.05

SpectralFreqStretchChance = 40 #20 #40 #0 #40


# Mel Params
NumOfMelBands = 128 #200 #128 #310
MelStartFreq = 20.0 #100.0 #20.0 #160.0
MelEndFreq = 16000.0 #10300.0
NumOfLowFreqsInPixelToCutMax = 4 #2 #4 #0 #6
NumOfHighFreqsInPixelToCutMax = 6 #4 #6 #0 #10


ResetBestValResults = True


#PoolingMethod = 'mean' # mean, max, meanexp
#PoolingMethodSoundscapes = 'max' # mean, max, meanexp
PoolingMethod = 'max' # mean, max, meanexp


UseMultiLabel = True # True False


ChunkAugmentationChance = 15 #10 #15 #0
FinalFilterChance = 15 #10 #15 #0 # 15
FinalFilterChance2 = 0 # 0 10



if SampleRate == 22050:
    AudioFilesDcaseDir = RootDir + 'BAD_Challenge_18/AudioFilesPreprocessed/03_3_Hp2000_22050Hz_Norm/'
else:
    AudioFilesDcaseDir = RootDir + 'KaggleBirdsongRecognition2020/data/audio/13_wav_filtered_and_normalized_32000Hz_kaiser_fast/'


# Use ValSounscapes for validation
#ValSoundscapesFoldIx = 7 #'all' #7    #0 # 'all' or FoldIx 0     Folds 0/1 (optimistic) vs 2/6 (pesimistic)



StartDateTimeStr = time.strftime("%y%m%d-%H%M%S")
InterimResultsFid = LogDir + StartDateTimeStr + '-InterimResults.txt'
EpochSummeryFid = LogDir + StartDateTimeStr + '-EpochSummery.txt'
EpochProgressFid = LogDir + StartDateTimeStr + '-EpochProgress.txt'
PrintInterimResult = 1



#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


timeStampStart = time.time() # for performance measurement




if ModelName == 'inception_v3':
    #ImageSize = 299
    ImageSize = (299, 299)

    ImageMean = [0.5, 0.5, 0.5]
    ImageStd = [0.5, 0.5, 0.5]
else:
    #ImageSize = (224, 244)

    
    #ImageSize = (574, 224)  
    #ImageSize = (547, 224)

    # ResizeFactor = (height/(NumOfMelBands-NumOfLowFreqsInPixelToCutMax/2.0-NumOfHighFreqsInPixelToCutMax/2.0) --> 1.75
    # width =  ResizeFactor x SegmentDuration x SampleRate / FftHopSizeInSamples
    # e.g. 547 = (224/(128-0-0)) x 5 x 32000 / 512
    imageHeight = 224
    resizeFactor = (imageHeight/(NumOfMelBands-NumOfLowFreqsInPixelToCutMax/2.0-NumOfHighFreqsInPixelToCutMax/2.0))
    imageWidth = int(resizeFactor * SegmentDuration * SampleRate / FftHopSizeInSamples)
    ImageSize = (imageWidth, imageHeight)
    print('resizeFactor', resizeFactor, 'ImageSize', ImageSize)
    # resizeFactor 1.792 ImageSize (560, 224)

    # ImageMean = [0.485, 0.456, 0.406]
    # ImageStd = [0.229, 0.224, 0.225]

    # Own mean/std
    ImageMean = [0.5, 0.4, 0.3]
    ImageStd = [0.5, 0.3, 0.1]




# Use all train files (and onyl SSs for validation)
#FileNamesTrain = np.loadtxt(MetadataDir + 'fileNamesTrain.txt', dtype=np.str, delimiter='\t').tolist()



#NumOfClasses = 659
#ClassIds = np.loadtxt(MetadataDir + 'ClassIds.txt', dtype=np.str, delimiter='\t').tolist()
#ClassIds = np.loadtxt(MetadataDir + 'ClassIdsOfValSoundscapes.txt', dtype=np.str, delimiter='\t').tolist()
#ClassIds = np.loadtxt(MetadataDir + 'ClassIdsUsSelectionPlus.txt', dtype=np.str, delimiter='\t').tolist()

ClassIds = np.loadtxt(RootDir + 'KaggleBirdsongRecognition2020/metadata/classIds.txt', dtype=np.str, delimiter='\t').tolist()



NumOfClasses = len(ClassIds)
print('NumOfClasses', NumOfClasses)

# Create ClassIxOfClassIdDict
ClassIxOfClassIdDict = {}
for ClassId in ClassIds:
    ClassIxOfClassIdDict[ClassId] = ClassIds.index(ClassId)



if UseMultiLabel: print('MultiLabel')
else: print('SingleLabel')





# NNN
# Test
# Create pandas val_df
# Attr.: filename, startTime, endTime, (path)
#val_df = pd.DataFrame(np.array(FileNamesValSoundscapesVal), columns=['filename'])




# Load own train_df
#trainSet_df = pd.read_csv(RootDir + 'KaggleBirdsongRecognition2020/metadata/trainSet_df_03_NoBgSpecies.csv', sep=';')
trainSet_df = pd.read_csv(RootDir + 'KaggleBirdsongRecognition2020/metadata/trainSet_df_05_NoBgSpecies_32000Hz.csv', sep=';') 
print('trainSet_df.shape', trainSet_df.shape)

# Load noise_df
noise_df = pd.read_csv(RootDir + 'KaggleBirdsongRecognition2020/metadata/noiseSet_df_01_32000Hz.csv', sep=';') 
#noise_df.classIds = noise_df.classIds.apply(ast.literal_eval)
print('noise_df.shape', noise_df.shape)


# Get Arai Train/Val Split
import sklearn.model_selection as sms
def getAraiBaselineTrainValSplit():

    # Remove train_audio/lotduc/XC195038.mp3
    train_df_new = pd.read_csv(RootDir + 'KaggleBirdsongRecognition2020/metadata/train.csv')
    ebird_code = 'lotduc'
    filename = 'XC195038.mp3'
    train_df_new = train_df_new[~((train_df_new["ebird_code"] == ebird_code) &
                    (train_df_new["filename"] == filename))]
    train_df_new = train_df_new.reset_index(drop=True)

    #print('train_df.shape', train_df.shape)
    print('train_df_new.shape', train_df_new.shape)

    # Use StratifiedKFold
    splitter = sms.StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    #for train, test in skf.split(X, y):

    for i, (trn_idx, val_idx) in enumerate(
            #splitter.split(train_df_new, y=train_df_new["ebird_code"])):
            splitter.split(train_df_new, y=train_df_new["ebird_code"])):
        
        # if i not in [0]:
        # #if i not in [0,1,2,3,4]:
        #     continue

        if i == 0: # Use fole 0

            print('fold', i)

            print(len(trn_idx), len(val_idx))
            print(trn_idx[:10], val_idx[:10])

            #trn_df = train_df_new.loc[trn_idx, :].reset_index(drop=True)
            #val_df = train_df_new.loc[val_idx, :].reset_index(drop=True)

            return trn_idx, val_idx

trn_idx, val_idx = getAraiBaselineTrainValSplit()

# Remove 
trainSet_df = trainSet_df[~(trainSet_df["filename"] == 'XC195038')]
trainSet_df = trainSet_df.reset_index(drop=True)
print('trainSet_df.shape', trainSet_df.shape)

train_df = trainSet_df.loc[trn_idx, :].reset_index(drop=True)
val_df = trainSet_df.loc[val_idx, :].reset_index(drop=True)

# When read form csv --> list is represented as string
#print(train_df.classIds.to_string(index=False))
#train_df['classIds'] = pd.eval(train_df['classIds'])

train_df.classIds = train_df.classIds.apply(ast.literal_eval)
val_df.classIds = val_df.classIds.apply(ast.literal_eval)

# classIds = train_df.at[0, 'classIds']
# for classId in classIds:
#     print(classId)

classIdsTrain = ClassIds
#print(classIdsTrain)


# # Create first kaggle data frames
# def createKaggleDataFrames():
#     print('Create kaggel data frames')



print('Init Done.')
print('ElapsedTime [s]: ', (time.time() - timeStampStart))






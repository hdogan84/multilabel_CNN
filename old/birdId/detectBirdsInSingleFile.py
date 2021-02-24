# cd /net/mfnstore-lin/export/tsa_transfer/TrainData/BAD_Challenge_18/Src/01_detectBirdsInSingleFile
# source activate /net/mfnstore-lin/export/tsa_transfer/PythonProjects/VirtualEnvs/BirdId18

import os
import time
import sys
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

import csv
import datetime
from operator import itemgetter
import glob
import subprocess

from scipy.special import expit # sigmoid

import librosa
import soundfile as sf
from PIL import Image

# To get MIME type
import magic
mime = magic.Magic(mime=True)


####################################################################################################
###############################################  Config  ###########################################
####################################################################################################

# Model (train run) depending parameters

NumOfClasses = 2 # 0: NoBird, 1: Bird

ModelName = 'inception_v3'
ImageSize = 299
ImageMean = [0.5, 0.5, 0.5]
ImageStd = [0.5, 0.5, 0.5]

# Detection window size in seconds
SegmentDuration = 4.0

SampleRate = 22050
FftSizeInSamples = 1536
FftHopSizeInSamples = 360

# Mel params
NumOfMelBands = 310
MelStartFreq = 160.0
MelEndFreq = 10300.0
NumOfLowFreqsInPixelToCutMax = 6
NumOfHighFreqsInPixelToCutMax = 10

####################################################################################################


DebugMode = False # True False

ModelStateDictPath = 'BirdDetectionModelStateDict_v01.pth.tar'
#ModelStateDictPath = 'BirdDetectionModelStateDict_v02.pth.tar'

SegmentOverlapForTestFilesInPerc = 75 # --> 1 sec hop size (At leat 18 segments needed for old pytorch version 0.3 ???)

PoolingMethod = 'Max' # Max Mean MeanExponentialPooling


NumOfWorkers = 10

CsvDelimiter = ';'

PrintFreq = 100

MimeTypesWavFile = ['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/vnd.wave']

IntermediateFiles = [] # Hack --> todo better maybe


# GPU check
GpuMode = torch.cuda.is_available()

if GpuMode: BatchSizeForTesting = 590 #660
else: BatchSizeForTesting = 100

# PyTorch version check
PyTorchVersionStr = torch. __version__
if PyTorchVersionStr[0] == '0' and int(PyTorchVersionStr[2]) <= 3:
    PytorchVersionAbove0p3 = False
else:
    PytorchVersionAbove0p3 = True
    

if DebugMode:
    print('PyTorchVersionStr', PyTorchVersionStr)
    print('GpuMode', GpuMode)


# Apply fade in/out of waveform to allow shorter signals and smaller step sizes (reduce receptive field)
FadeInOutMode = False # True False

if FadeInOutMode:

    # Init FadeEnvelope
    FadeIntervalDuration = 2.0 # Duration of remaining interval (rest will be faded out)

    FadeInTime = 0.01   # 10 ms
    FadeOutTime = 0.01

    FadeInTimeInSamples = int(FadeInTime*SampleRate)
    FadeOutTimeInSamples = int(FadeOutTime*SampleRate)
    FadeIntervalDurationInSamples = int(FadeIntervalDuration*SampleRate)

    DurationOfSegmentInSamples = int(SegmentDuration*SampleRate)

    FadeEnvelope = np.ones(FadeIntervalDurationInSamples, dtype=np.float64)

    FadeEnvelope[:FadeInTimeInSamples] = np.linspace(0.0, 1.0, num=FadeInTimeInSamples)
    FadeEnvelope[-FadeOutTimeInSamples:] = np.linspace(1.0, 0.0, num=FadeOutTimeInSamples)

    FadeEnvelopeRest = np.zeros(DurationOfSegmentInSamples-len(FadeEnvelope), dtype=np.float64)

    FadeEnvelope = np.concatenate((FadeEnvelope, FadeEnvelopeRest), axis=0)

    # Adjust overlap to e.g. 50% of FadeIntervalDuration
    HopLengthNew = 0.5*FadeIntervalDuration
    SegmentOverlapForTestFilesInPerc = (HopLengthNew - SegmentDuration) / -(SegmentDuration*0.01)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AudioDatasetSingleFile(Dataset):

    def __init__(self, AudioFid, transform=None):


        self.AudioFid = AudioFid
        self.transform = transform


        # Get Some Audio File Infos
        with sf.SoundFile(AudioFid, 'r') as f:
            self.SampleRate = f.samplerate
            self.NumOfSamples = len(f)
            self.NumOfChannels = f.channels
            self.DurationOfFile = self.NumOfSamples/self.SampleRate

            assert self.SampleRate == SampleRate


        HopLength = SegmentDuration - SegmentDuration * SegmentOverlapForTestFilesInPerc*0.01

        if DebugMode:
            print('SampleRate [Hz]', self.SampleRate)
            print('NumOfSamples', self.NumOfSamples)
            print('NumOfChannels', self.NumOfChannels)
            print('DurationOfFile [s]', self.DurationOfFile)
            print('HopLength [s]', HopLength)

        # NNN
        # Create List of Segments
        # Segment has Attr.: StartTime, ChannelIx
        self.Segments = [] # List of Segments, Segment has Attr.: StartTime, ChannelIx
        self.NumOfSegments = 0 # not realy needed --> len(Segments) can be used
        self.NumOfSegmentsPerChannel = 0
        self.NumOfTimeIntervals = 0


        StartTime = 0.0

        while StartTime < self.DurationOfFile:

            for ChannelIx in range(self.NumOfChannels):
                Segment = {}
                Segment['StartTime'] = StartTime
                Segment['ChannelIx'] = ChannelIx
                Segment['TimeIntervalIx'] = self.NumOfTimeIntervals
                #print(self.NumOfSegments, StartTime, ChannelIx)

                self.Segments.append(Segment)
                self.NumOfSegments += 1

            StartTime += HopLength
            self.NumOfTimeIntervals += 1

        if DebugMode:
            print('NumOfTimeIntervals', self.NumOfTimeIntervals)


    def __len__(self):
        return len(self.Segments)


    def __getitem__(self, SegmentIx):

        Segment = self.Segments[SegmentIx]
        StartTime = Segment['StartTime']
        ChannelIx = Segment['ChannelIx']
        TimeIntervalIx = Segment['TimeIntervalIx']

        #print(SegmentIx, FileName, StartTime)
        
        #SpecImage = getSpecSegmentOfTestFile(FileName, StartTime)
        #SpecImage = getSpecSegmentOfSingleFile(StartTime)
        SpecImage = getSpecSegmentOfSingelFileMultiChannel(self.AudioFid, self.DurationOfFile, StartTime, ChannelIx)

        # Convert ...
        h,w = SpecImage.shape
        c=3
        ThreeChannelSpec = np.empty((h,w,c), dtype=np.uint8)
        ThreeChannelSpec[:,:,0] = SpecImage
        ThreeChannelSpec[:,:,1] = SpecImage
        ThreeChannelSpec[:,:,2] = SpecImage

        SpecImage = Image.fromarray(ThreeChannelSpec)

        if self.transform: SpecImage = self.transform(SpecImage)
        

        DataSample = {'SpecImage': SpecImage, 'TimeIntervalIx': TimeIntervalIx, 'ChannelIx': ChannelIx}

        return DataSample


def resizeSpecImage(SpecImage, ImageSize):

    SpecImagePil = Image.fromarray(SpecImage.astype(np.uint8))

    ImageWidthResizeFactor = ImageSize/SpecImage.shape[1]
    ImageHeightResizeFactor = ImageSize/SpecImage.shape[0]

    InterpolationMethod = Image.LANCZOS
    SpecImagePil = SpecImagePil.resize((ImageSize, ImageSize), InterpolationMethod) 

    # Cast to int8
    SpecImageResized = np.array(SpecImagePil, dtype=np.uint8)

    return SpecImageResized

def getMelSpec(SampleVec):

    
    MelSpec = librosa.feature.melspectrogram(y=SampleVec, sr=SampleRate, n_fft=FftSizeInSamples, hop_length=FftHopSizeInSamples, n_mels=NumOfMelBands, fmin=MelStartFreq, fmax=MelEndFreq, power=2.0)

    MelSpec = librosa.power_to_db(MelSpec, ref=np.max, top_db=100)

    NumOfLowFreqsInPixelToCut = int(NumOfLowFreqsInPixelToCutMax/2.0)
    NumOfHighFreqsInPixelToCut = int(NumOfHighFreqsInPixelToCutMax/2.0)

    if NumOfHighFreqsInPixelToCut:
        MelSpec = MelSpec[NumOfLowFreqsInPixelToCut:-NumOfHighFreqsInPixelToCut]
    else:
        MelSpec = MelSpec[NumOfLowFreqsInPixelToCut:]

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    MelSpec = MelSpec[::-1, ...]

    # Normalize values between 0 and 1
    MelSpec -= MelSpec.min()
    MelSpec /= MelSpec.max()

    MaxVal = 255.9
    MelSpec *= MaxVal
    MelSpec = MaxVal-MelSpec


    return MelSpec


def checkAndConvertAudioFile(AudioSrcFid):

    MimeType = mime.from_file(AudioSrcFid)
    #print('MimeType', MimeType)

    # Not a wav file --> needs conversion
    if MimeType not in MimeTypesWavFile:

        AudioDir = os.path.dirname(AudioSrcFid)
        FileName = os.path.splitext(os.path.basename(AudioSrcFid))[0]
        AudioDstFid = AudioDir + '/' + FileName + '.wav'
        
        subprocess.call(['ffmpeg', '-i', AudioSrcFid, AudioDstFid, '-nostats', '-loglevel', '0', '-y'])

        IntermediateFiles.append(AudioDstFid) # Hack --> todo better maybe
    
    else:
        AudioDstFid = AudioSrcFid

    return AudioDstFid

def preprocessAudioFile(AudioSrcFid):

    # Normalize, HighPass Filter, Resample, Normalize
    AudioDstFid = AudioSrcFid[:-4] + '_preprocessed.wav'
    subprocess.call(['sox', AudioSrcFid, AudioDstFid, '-V0', 'norm', '-4', 'highpass', '2000.0', 'rate', '-I', '22050', 'norm', '-3'])

    IntermediateFiles.append(AudioDstFid) # Hack --> todo better maybe

    return AudioDstFid


def getSpecSegmentOfSingelFileMultiChannel(AudioFid, DurationOfFile, StartTime, ChannelIx):

    #AudioFid = AudioFilesDir + FileName + '.wav'
    #DurationOfFile = MetadataPerFileDict[FileName]['Duration']
    
    DurationOfFileInSamples = int(DurationOfFile*SampleRate)

    DurationOfSegmentInSamples = int(SegmentDuration*SampleRate)

    StartSample = int(StartTime*SampleRate)
    EndSample = StartSample + DurationOfSegmentInSamples

    if StartSample + DurationOfSegmentInSamples > DurationOfFileInSamples:
        AudioData = sf.read(AudioFid)[0]
        if len(AudioData.shape) > 1:
            SampleVec = AudioData[:,ChannelIx]
        else:
            SampleVec = AudioData

        SampleVecRepeated = SampleVec.copy()
        while EndSample > SampleVecRepeated.shape[0]:
            SampleVecRepeated = np.append(SampleVecRepeated, SampleVec)
        SampleVec = SampleVecRepeated[StartSample:EndSample]
    else:
        AudioData = sf.read(AudioFid, start=StartSample, stop=EndSample)[0]
        if len(AudioData.shape) > 1:
            SampleVec = AudioData[:,ChannelIx]
        else:
            SampleVec = AudioData

    if FadeInOutMode:
        SampleVec *= FadeEnvelope

    SpecSegment = getMelSpec(SampleVec)

    SpecSegment = resizeSpecImage(SpecSegment, ImageSize)

    return SpecSegment

def predictSegmentsInParallel(test_loader, model, AudioDatasetObj):

    model.eval() # switch to evaluate mode

    if DebugMode:
        batch_time = AverageMeter()
        end = time.time()

    NumOfTimeIntervals = AudioDatasetObj.NumOfTimeIntervals
    NumOfChannels = AudioDatasetObj.NumOfChannels

    Predictions = np.zeros((NumOfChannels, NumOfTimeIntervals, NumOfClasses), dtype=np.float64)

    # Pytorch >= 0.4
    if PytorchVersionAbove0p3:
        with torch.no_grad():
            for i, sample_batched in enumerate(test_loader):
                input = sample_batched['SpecImage']

                TimeIntervalIxs = sample_batched['TimeIntervalIx']
                ChannelIxs = sample_batched['ChannelIx']

                output = model(input)
                OutputsNp = output.cpu().data.numpy()

                if DebugMode:
                    batch_time.update(time.time() - end) # measure elapsed time
                    end = time.time()
                    if i % PrintFreq == 0:
                        LogStr = time.strftime("%y%m%d-%H%M%S") + '\t'
                        LogStr += ('TEST [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} '.format(
                               i, len(test_loader), batch_time=batch_time))
                        print(LogStr)

                # Collect Predictions Per TimeInterval & Channel
                for SegmentIx in range(len(TimeIntervalIxs)):
                    TimeIntervalIx = TimeIntervalIxs[SegmentIx]
                    ChannelIx = ChannelIxs[SegmentIx]

                    Predictions[ChannelIx, TimeIntervalIx] = OutputsNp[SegmentIx]

    else: # Pytorch 0.3

        for i, sample_batched in enumerate(test_loader):
            input = sample_batched['SpecImage']

            TimeIntervalIxs = sample_batched['TimeIntervalIx']
            ChannelIxs = sample_batched['ChannelIx']

            input_var = torch.autograd.Variable(input, volatile=True)

            output = model(input_var)
            OutputsNp = output.cpu().data.numpy()

            if DebugMode:
                batch_time.update(time.time() - end) # measure elapsed time
                end = time.time()
                if i % PrintFreq == 0:
                    LogStr = time.strftime("%y%m%d-%H%M%S") + '\t'
                    LogStr += ('TEST [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} '.format(
                           i, len(test_loader), batch_time=batch_time))
                    print(LogStr)

            # Collect Predictions Per TimeInterval & Channel
            for SegmentIx in range(len(TimeIntervalIxs)):
                TimeIntervalIx = TimeIntervalIxs[SegmentIx]
                ChannelIx = ChannelIxs[SegmentIx]

                Predictions[ChannelIx, TimeIntervalIx] = OutputsNp[SegmentIx]

    
    Predictions = Predictions.astype(np.float64)

    # Sigmoid normalization
    Predictions = expit(Predictions)

    return Predictions


def writeCsvFile(Predictions, AudioFid):

    ClassNames = ['NoBird', 'Bird']

    # Add StartTime, EndTime as first 2 cols
    Header = ['StartTime [s]', 'EndTime [s]'] + ClassNames

    # Add Start- & EndTimes
    NumOfTimeIntervals = Predictions.shape[1]
    HopLength = SegmentDuration - SegmentDuration * SegmentOverlapForTestFilesInPerc*0.01
    StartTimes = np.arange(0, NumOfTimeIntervals*HopLength, HopLength)

    if FadeInOutMode:
        EndTimes = StartTimes + FadeIntervalDuration
    else:
        EndTimes = StartTimes + SegmentDuration

    NumOfChannels = Predictions.shape[0]

    for ix in range(NumOfChannels):

        PredictionsPerChannel = Predictions[ix]

        CsvFid = AudioFid[:-4] + '_c' + str(ix+1) + '.csv'

        with open(CsvFid, mode='w', newline='', encoding='utf-8') as f:
            cw = csv.writer(f, delimiter=CsvDelimiter)
            cw.writerow(Header)
            Data = np.column_stack((StartTimes, EndTimes, PredictionsPerChannel))
            cw.writerows(Data)


def summarizePredictionsFileBased(Predictions, PoolingMethod):

    #NumOfChannels, NumOfTimeIntervals, NumOfClasses = Predictions.shape
    #print(Predictions.shape, Predictions.dtype)
    #PredictionsReshaped = Predictions.reshape(NumOfChannels*NumOfTimeIntervals, NumOfClasses)
    
    PredictionsReshaped = Predictions.reshape(-1, Predictions.shape[-1])
    

    if PoolingMethod == 'Mean':
        PredictionsFileBased = np.mean(PredictionsReshaped, axis=0) # Simple Avr

    if PoolingMethod == 'Max':
        PredictionsFileBased = np.max(PredictionsReshaped, axis=0)

    if PoolingMethod == 'MeanExponentialPooling':
        PredictionsFileBased = np.mean(PredictionsReshaped ** 2, axis=0)

    
    return PredictionsFileBased


def detect(AudioFid):

    if DebugMode:
        StartTime = time.time()

    # Init model
    model = models.__dict__[ModelName](pretrained=True)

    # Replace Final Layer
    NumOfFeatures = model.fc.in_features
    model.fc = nn.Linear(NumOfFeatures, NumOfClasses)

    # Load state dict
    model.load_state_dict(torch.load(ModelStateDictPath))

    if GpuMode:
        model = torch.nn.DataParallel(model).cuda()
        

    # Convert & Preprocess Audio
    AudioFid = checkAndConvertAudioFile(AudioFid)
    AudioFid = preprocessAudioFile(AudioFid)

    cudnn.benchmark = True # Not working otherwise ?
    normalize = transforms.Normalize(mean=ImageMean, std=ImageStd)

    AudioDatasetSingleFileObj = AudioDatasetSingleFile(AudioFid, transform=transforms.Compose([transforms.ToTensor(),normalize]))

    NumOfegments = len(AudioDatasetSingleFileObj)
    if DebugMode:
        print('NumOfegments', NumOfegments)
        if not PytorchVersionAbove0p3 and NumOfegments < 18:
            print('Warning: At least 18 segments needed if using pytorch version <= 0.3')

    test_loader = torch.utils.data.DataLoader(AudioDatasetSingleFileObj, batch_size=BatchSizeForTesting, shuffle=False, num_workers=NumOfWorkers, pin_memory=True)


    # Predictions Dim: NumOfChannels, NumOfTimeIntervals, NumOfClasses
    Predictions = predictSegmentsInParallel(test_loader, model, AudioDatasetSingleFileObj)


    if DebugMode:
        print('ElapsedTime', time.time()-StartTime)
    
    return Predictions



#############################################################################
#############################################################################
#############################################################################



if __name__ == "__main__":
    
    # Pass audio path as argument
    AudioFid = sys.argv[1]

    Predictions = detect(AudioFid)

    writeCsvFile(Predictions, AudioFid)

    # Summerize predictions
    PredictionsFileBased = summarizePredictionsFileBased(Predictions, PoolingMethod)
    print('Bird probability:', PredictionsFileBased[1])

    # Remove intermediate files
    for f in IntermediateFiles: os.remove(f)
    del IntermediateFiles[:] # Don't use IntermediateFiles = [] !!!


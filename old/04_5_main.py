# Original Sources: https://github.com/pytorch/examples/tree/master/imagenet

import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import shutil
from sklearn import metrics

from scipy.special import expit # sigmoid




#from DataSetUtils import *

#from ConfigAndUtils import *
#from ConfigAndUtils_03_5_SoftAugm import *
#from ConfigAndUtils_NoAugm import *
#from ConfigAndUtils_SoftAugm import *
from ConfigAndUtils_FullAugm import *



# EEE
if 'efficientnet' in ModelName:
    from efficientnet_pytorch import EfficientNet


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')



# Keep track of best validation scores (so far)
EpoRes = {}

EpoRes['lrapBest'] = 0
EpoRes['cMapBest'] = 0
EpoRes['f1Best'] = 0


#print(RandomSeed)
random.seed(RandomSeed)

 # Make it deterministic (not  working if used in train file !!!)
np.random.seed(RandomSeed)
os.environ["PYTHONHASHSEED"] = str(RandomSeed)
torch.manual_seed(RandomSeed)
torch.cuda.manual_seed(RandomSeed)  # type: ignore
torch.cuda.manual_seed_all(RandomSeed)

if TrainDeterministic:
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
else:
    torch.backends.cudnn.deterministic = False  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


#####  For Train Files   #####
class AudioDatasetTrain(Dataset):


    def __init__(self, df, noise_df, transform=None):
        
        self.nParts = len(df.index)
        self.transform = transform


        # Convert DataFrame to list of dicts
        self.audioParts = df.to_dict('records')

        self.noiseParts = noise_df.to_dict('records')
        self.nNoiseParts = len(noise_df.index)



        # Create  partIxsOfClassIdDict (FileNamesTrainOfClassIdDict before)
        self.partIxsOfClassIdDict = {}
        for partIx in range(self.nParts):
            classIds = self.audioParts[partIx]['classIds']
            for classId in classIds:
                if classId not in self.partIxsOfClassIdDict:
                    self.partIxsOfClassIdDict[classId] = []
                self.partIxsOfClassIdDict[classId].append(partIx)


    def __len__(self):
        return self.nParts


    def __getitem__(self, partIx):



        ############################################################################################
        ###########################  Get spec segment with augmentation  ###########################
        ############################################################################################


        # Apply some Jitter to Segment Duration (Valid also for Added Segments)
        DurationOfSegment = SegmentDuration + random.uniform(-DurationJitter, DurationJitter)
        DurationOfSegmentInSamples = int(DurationOfSegment*SampleRate)

        # Enlarge DurationOfSegmentInSamples by Random SegmentLength (to be cutted later at random Position)
        CutSegment = False
        if CutSegmentChance and random.randrange(100) < CutSegmentChance:
            DurationOfSegmentInSamplesOrg = DurationOfSegmentInSamples
            CutSegmentNumOfSamples = random.randint(0, DurationOfSegmentInSamplesOrg)
            DurationOfSegmentInSamples += CutSegmentNumOfSamples
            CutSegment = True


        audioPart = self.audioParts[partIx]


        ClassIdOrClassIds = audioPart['classIds']
        SampleVec = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax)




        # Single label
        if not UseMultiLabel:
            if len(ClassIdOrClassIds) > 1:
                ClassId = random.choice(ClassIdOrClassIds)
                #print(FileName, ClassIdOrClassIds)
            else:
                ClassId = ClassIdOrClassIds[0]
                

        # Multi label
        else:

            ClassIds = ClassIdOrClassIds



            if AddFileWithRandomClass01 and random.randrange(100) < AddFileWithRandomClass01:

                partIx = random.randint(0, self.nParts-1)
                audioPart = self.audioParts[partIx]
                ClassIdOrClassIds = audioPart['classIds']
                SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax)
                ClassIds += ClassIdOrClassIds
                SampleVec += SampleVecAdd

                if AddFileWithRandomClass02 and random.randrange(100) < AddFileWithRandomClass02:
                    partIx = random.randint(0, self.nParts-1)
                    audioPart = self.audioParts[partIx]
                    ClassIdOrClassIds = audioPart['classIds']
                    SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax)
                    ClassIds += ClassIdOrClassIds
                    SampleVec += SampleVecAdd


                    if AddFileWithRandomClass03 and random.randrange(100) < AddFileWithRandomClass03:
                        partIx = random.randint(0, self.nParts-1)
                        audioPart = self.audioParts[partIx]
                        ClassIdOrClassIds = audioPart['classIds']
                        SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax)
                        ClassIds += ClassIdOrClassIds
                        SampleVec += SampleVecAdd

                        if AddFileWithRandomClass04 and random.randrange(100) < AddFileWithRandomClass04:
                            partIx = random.randint(0, self.nParts-1)
                            audioPart = self.audioParts[partIx]
                            ClassIdOrClassIds = audioPart['classIds']
                            SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax)
                            ClassIds += ClassIdOrClassIds
                            SampleVec += SampleVecAdd

        


        ##########################################################################################
        ###################################  Add More Segments  ##################################
        ##########################################################################################

        if UseMultiLabel:
            ClassId = random.choice(ClassIds)

        #FileNamesTrainOfClassId = FileNamesTrainOfClassIdDict[ClassId] # Remove
        partIxsOfClassId = self.partIxsOfClassIdDict[ClassId]



        ####  Add Same Class 01  ####
        if AddSameClassChance01 and random.randrange(100) < AddSameClassChance01:
            partIx = random.randint(0,len(partIxsOfClassId)-1)
            audioPart = self.audioParts[partIx]
            SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AddSameClassAmpExpMin01, AddSameClassAmpExpMax01, Augmentation=False)
            SampleVec += SampleVecAdd

            ####  Add Same Class 02  ####
            if AddSameClassChance02 and random.randrange(100) < AddSameClassChance02:
                partIx = random.randint(0,len(partIxsOfClassId)-1)
                audioPart = self.audioParts[partIx]
                SampleVecAdd = readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AddSameClassAmpExpMin01, AddSameClassAmpExpMax01, Augmentation=False)
                SampleVec += SampleVecAdd




        ##########  Dcase 2018 Noise  ##########
        if DcaseNoiseChance01 and random.randrange(100) < DcaseNoiseChance01:
            partIx = random.randint(0, self.nNoiseParts-1)
            noisePart = self.noiseParts[partIx]
            SampleVecAdd = readAudioFromRandomPosition(noisePart, DurationOfSegmentInSamples, DcaseNoiseAmpExpMin01, DcaseNoiseAmpExpMax01, Augmentation=False)
            SampleVec += SampleVecAdd

        if DcaseNoiseChance02 and random.randrange(100) < DcaseNoiseChance02:
            partIx = random.randint(0, self.nNoiseParts-1)
            noisePart = self.noiseParts[partIx]
            SampleVecAdd = readAudioFromRandomPosition(noisePart, DurationOfSegmentInSamples, DcaseNoiseAmpExpMin02, DcaseNoiseAmpExpMax02, Augmentation=False)
            SampleVec += SampleVecAdd

        if DcaseNoiseChance03 and random.randrange(100) < DcaseNoiseChance03:
            partIx = random.randint(0, self.nNoiseParts-1)
            noisePart = self.noiseParts[partIx]
            SampleVecAdd = readAudioFromRandomPosition(noisePart, DurationOfSegmentInSamples, DcaseNoiseAmpExpMin03, DcaseNoiseAmpExpMax03, Augmentation=False)
            SampleVec += SampleVecAdd




        # Cut Segment of CutSegmentNumOfSamples at random Position 
        if CutSegment:
            CutPosition = random.randint(0, DurationOfSegmentInSamplesOrg)
            SampleVec = np.concatenate((SampleVec[:CutPosition], SampleVec[CutPosition+CutSegmentNumOfSamples:]))
            
        
        # Random Final Amp
        if FinalAmpChangeChance and random.randrange(100) < FinalAmpChangeChance:
            AmpFactor = 10**(random.uniform(FinalAmpExpMin, FinalAmpExpMax))
            SampleVec *= AmpFactor



        SpecImage = getMelSpec(SampleVec, FftSizeInSamples, FftHopSizeInSamples, Augmentation=True)


        
        if SpectralTimeStretchChance and random.randrange(100) < SpectralTimeStretchChance:
            SpecImage = applySpectralTimeStretch(SpecImage, SpectralTimeStretchOpt)

        if SpectralFreqStretchChance and random.randrange(100) < SpectralFreqStretchChance:
            SpecImage = applySpectralFreqStretch(SpecImage, SpectralFreqStretchOpt)


        #if ImageSize != 'Original':
        SpecImage = resizeSpecImage(SpecImage, ImageSize)


        ############################################################################################
        ############################################################################################
        ############################################################################################

        #print(ClassIds)

        if UseMultiLabel:
            # GroundTruth is (one hot encoding) vector
            GroundTruth = np.zeros(NumOfClasses, dtype=np.float32)
            for ClassId in ClassIds:
                ClassIx = ClassIxOfClassIdDict[ClassId]
                GroundTruth[ClassIx] = 1.0

            # NNN
            # Label Smoothing
            if LabelSmoothingFactor:
                # [1. 0. 0. 1. ...] --> [0.901282 0.00128205 0.00128205 0.901282 ...] with LabelSmoothingFactor = 0.1
                GroundTruth *= 1 - LabelSmoothingFactor
                GroundTruth += LabelSmoothingFactor / GroundTruth.shape[0]

        else:
            # GroundTruth is ClassIx
            #assert len(ClassIds) == 1
            #ClassId = ClassIds
            GroundTruth = ClassIxOfClassIdDict[ClassId]


        
        #print('SpecImageStats', SpecImage.dtype, np.min(SpecImage), np.mean(SpecImage), np.max(SpecImage)) # SpecImageStats uint8 13 69.5829280931 107

        # Convert ...
        h,w = SpecImage.shape
        c=3
        ThreeChannelSpec = np.empty((h,w,c), dtype=np.uint8)
        ThreeChannelSpec[:,:,0] = SpecImage
        ThreeChannelSpec[:,:,1] = SpecImage
        ThreeChannelSpec[:,:,2] = SpecImage

        SpecImage = Image.fromarray(ThreeChannelSpec)


        if self.transform: SpecImage = self.transform(SpecImage)
        
        DataSample = {'SpecImage': SpecImage, 'GroundTruth': GroundTruth}

        #if self.transform: DataSample = self.transform(DataSample)

        return DataSample



#####  For val/test audio parts   #####
class AudioDatasetTest(Dataset):

    def __init__(self, df, classIdsTrain, transform=None):

        self.nParts = len(df.index)
        self.classIdsTrain = classIdsTrain
        self.nClassesTrain = len(classIdsTrain)

        self.transform = transform

        self.df = df # Only used for export function

        self.durationOfSegment = SegmentDuration

        
        self.hopLength = self.durationOfSegment - self.durationOfSegment * SegmentOverlapForTestFilesInPerc*0.01
        hopLengthInSamples = int(self.hopLength*SampleRate)

        self.durationOfSegmentInSamples = int(self.durationOfSegment*SampleRate)


        # Validation mode (ground truth is provided)
        self.valMode = False
        if 'classIds' in df.columns:
            self.valMode = True
            # Create ground truth matrix (nParts x nClassesTrain)
            self.gT = np.zeros((self.nParts, self.nClassesTrain), dtype=np.int)
            print('self.gT.shape', self.gT.shape)


        # Expand audio parts with offset & channel infos

        self.segments = []
        self.predictionsPerParts = []
        for partIx in range(self.nParts):
            filename = df.at[partIx, 'filename']
            path = df.at[partIx, 'path']
            startSample = df.at[partIx, 'startSample']
            endSample = df.at[partIx, 'endSample']
            nChannels = df.at[partIx, 'nChannels']

            if self.valMode:
                classIds = df.at[partIx, 'classIds']
                for classId in classIds:
                    classIx = classIdsTrain.index(classId)
                    self.gT[partIx,classIx] = 1

            
            offset = 0
            offsetIx = 0

            # NNN --> WRONG: only first 5 sec. used for validation !!!
            #while offset < self.durationOfSegmentInSamples:
            #partDurationInSamples = endSample - startSample
            #while offset < partDurationInSamples:

            if ValidateOnlyBeginning:
                partDurationInSamples = self.durationOfSegmentInSamples
            else:
                partDurationInSamples = endSample - startSample

            while offset < partDurationInSamples:
                for channelIx in range(nChannels):
                    segment = {}
                    segment['partIx'] = partIx
                    segment['filename'] = filename
                    segment['path'] = path
                    segment['startSample'] = startSample
                    segment['endSample'] = endSample

                    segment['channelIx'] = channelIx
                    segment['offsetIx'] = offsetIx
                    segment['offset'] = offset

                    self.segments.append(segment)

                offset += hopLengthInSamples
                offsetIx += 1

            predictionsPerPart = np.zeros((nChannels, offsetIx, self.nClassesTrain), dtype=np.float64)
            self.predictionsPerParts.append(predictionsPerPart)



        print('len(self.segments)', len(self.segments))



        if self.valMode:
            
            x=1

            # Ground truth post processing

            # Remove cols (predictions) without classes in GT
            # cMap only working if each class is represented by at least one part
            # metrics.average_precision_score(y_true, y_pred, average='macro') gives error: recall = tps / tps[-1]
            
            # self.classIxsNotPresent = np.argwhere(np.all(self.gT == 0, axis=0))
            # self.gT = np.delete(self.gT, np.s_[self.classIxsNotPresent], axis=1)

            # print('self.classIxsNotPresent', self.classIxsNotPresent)


            # Remove rows (parts) without classes in GT
            # rMap only working if each part is represented by at least one class
            # metrics.average_precision_score(y_true, y_pred, average='samples') gives error: recall = tps / tps[-1]
            # label_ranking_average_precision_score gives same result without removing rows
            
            # self.partIxsNotPresent = np.argwhere(np.all(self.gT == 0, axis=1))
            # self.gT = np.delete(self.gT, np.s_[self.partIxsNotPresent], axis=0)

            # --> Preprocessing needs to be applied also for predictions





    def __len__(self):
        return len(self.segments)

    def __getitem__(self, segmentIx):

        segment = self.segments[segmentIx]

        partIx = segment['partIx']
        path = segment['path']
        startSample = segment['startSample']
        endSample = segment['endSample']
        offset = segment['offset']
        channelIx = segment['channelIx']

        # Get audio data

        startIx = startSample + offset
        endIx = startIx + self.durationOfSegmentInSamples
        if endIx < endSample:
            audioData = sf.read(path, start=startIx, stop=endIx, always_2d=True)[0]
            SampleVec = audioData[:,channelIx]

        else:
            # Read entire part
            audioData = sf.read(path, start=startSample, stop=endSample, always_2d=True)[0]
            SampleVec = audioData[:,channelIx]

            endIx = offset + self.durationOfSegmentInSamples
            SampleVecRepeated = SampleVec.copy()
            while endIx > SampleVecRepeated.shape[0]:
                SampleVecRepeated = np.append(SampleVecRepeated, SampleVec)
            SampleVec = SampleVecRepeated[offset:endIx]


        #assert SampleVec.shape[0] == self.durationOfSegmentInSamples


        # NNN
        if not SampleVec.flags['F_CONTIGUOUS']:     # or SampleVec.flags.f_contiguous
            SampleVec = np.asfortranarray(SampleVec)


        SpecImage = getMelSpec(SampleVec, FftSizeInSamples, FftHopSizeInSamples, Augmentation=False)

        SpecImage = resizeSpecImage(SpecImage, ImageSize, RandomizeInterpolationFilter=False)

        # Convert ...
        h,w = SpecImage.shape
        c=3
        ThreeChannelSpec = np.empty((h,w,c), dtype=np.uint8)
        ThreeChannelSpec[:,:,0] = SpecImage
        ThreeChannelSpec[:,:,1] = SpecImage
        ThreeChannelSpec[:,:,2] = SpecImage

        SpecImage = Image.fromarray(ThreeChannelSpec)

        if self.transform: SpecImage = self.transform(SpecImage)

        # DataSample = {'SpecImage': SpecImage, 'FileAndTimeIx': partIx}
        # return DataSample

        return {'SpecImage': SpecImage, 'FileAndTimeIx': partIx, 'segmentIx': segmentIx}    # Raus: 'FileAndTimeIx': partIx


    # Maybe just concat --> predicionsPerSegments[segmentIx]
    def collectPredictionsPerBatch(self, segmentIxs, predictions):
        
        nSegments = len(segmentIxs)
        #print('nSegments', nSegments)

        for ix in range(nSegments): 
            segment = self.segments[segmentIxs[ix]]
            partIx = segment['partIx']
            offsetIx = segment['offsetIx']
            channelIx = segment['channelIx']

            self.predictionsPerParts[partIx][channelIx][offsetIx] = predictions[ix]

    
    def eval(self):

        predictions = np.zeros((self.nParts, self.nClassesTrain), dtype=np.float64)

        for partIx in range(self.nParts):

            predictionsPerPart = self.predictionsPerParts[partIx]

            # Flatten matrix (nChannels, nSegments, nClasses -->  nChannels*nSegments, nClasses)
            # predictionsPerPartReshaped = predictionsPerPart.reshape(nChannels*nSegments, nClasses)
            predictionsPerPartReshaped = predictionsPerPart.reshape(-1, predictionsPerPart.shape[-1])

            if PoolingMethod == 'mean':
                predictions[partIx] = np.mean(predictionsPerPartReshaped, axis=0)
            
            if PoolingMethod == 'meanexp':
                predictions[partIx] = np.mean(predictionsPerPartReshaped ** 2, axis=0)
            
            if PoolingMethod == 'max':
                predictions[partIx] = np.max(predictionsPerPartReshaped, axis=0)



        # Apply sigmoid ?
        #print(np.min(predictions), np.mean(predictions), np.max(predictions))
        if np.min(predictions) < 0.0 or np.max(predictions) > 1.0:
            #print('Apply sigmoid post-processing')
            predictions = expit(predictions)

        # NNN
        # Remove cols (predictions) without classes in GroundTruthValSoundscapes --> for cMap to work 
        #predictions = np.delete(predictions, np.s_[self.classIxsNotPresent], axis=1)

        # # Remove rows (parts) without classes --> for rMap to work (label_ranking_average_precision_score gives same result without removing rows)
        #predictions = np.delete(predictions, np.s_[self.partIxsNotPresent], axis=0)


        lrap = metrics.label_ranking_average_precision_score(self.gT, predictions)
        cMap = metrics.average_precision_score(self.gT, predictions, average='macro') # 'micro' 'macro' 'samples'
        

        # # f1 score
        # if UseMultiLabel:
        #     self.threshold = 0.1
        # else:
        #     self.threshold = 0.999991
            
        # predictionsBinary = predictions > self.threshold
        # f1 = metrics.f1_score(self.gT, predictionsBinary, average='micro') # 'micro', 'macro', 'samples', 'weighted'


        # NNN 

        # Find best f1 score by selecting optimal threshold per class
        beta = 1.0 #2.0 #1.0 #0.5 #1.0  # 0.5: better precision, 2.0: better recall
        predictionsThresholdPerClass = np.copy(predictions)
        self.tresholdsPerClass = []
        np.seterr(divide='ignore', invalid='ignore')
        for classIx in range(self.nClassesTrain):

            precision, recall, thresholds = metrics.precision_recall_curve(self.gT[:,classIx], predictions[:,classIx])

            # Remove last value (see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
            # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold. This ensures that the graph starts on the y axis.
            precision = precision[:-1]
            recall = recall[:-1]
            
            #f1_scores = (2*recall*precision)/(recall+precision)
            f1_scores = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)

            if np.isnan(np.sum(f1_scores)):
                f1_nan = np.isnan(f1_scores)
                #print('NaNs', np.sum(f1_nan))
                f1_scores = f1_scores[~f1_nan]
                precision = precision[~f1_nan]
                recall = recall[~f1_nan]
                thresholds = thresholds[~f1_nan]


            
            threshold_best = thresholds[np.argmax(f1_scores)]
            f1_scores_best = np.max(f1_scores)

            self.tresholdsPerClass.append(threshold_best)

            #print(classIx, classIdsTrain[classIx], f1_scores_best, threshold_best)

            predictionsThresholdPerClass[:,classIx] = predictions[:,classIx] >= threshold_best


        f1 = metrics.f1_score(self.gT, predictionsThresholdPerClass, average='micro') # 'micro', 'macro', 'samples', 'weighted'

        np.seterr(divide='warn', invalid='warn')
        

        # print('lrap', '%.3f'%(lrap*100.0))
        # print('cMap', '%.3f'%(cMap*100.0))
        # print('f1', '%.3f'%(f1*100.0))

        self.evalScores = {}
        self.evalScores['lrap'] = lrap
        self.evalScores['cMap'] = cMap
        self.evalScores['f1'] = f1

        #return lrap, cMap, f1
        return self.evalScores


    def exportPredictionsDict(self, dstDir):

        predictionsDict = {}
        predictionsDict['df'] = self.df
        predictionsDict['classIdsTrain'] = self.classIdsTrain
        predictionsDict['predictionsPerParts'] = self.predictionsPerParts
        predictionsDict['hopLength'] = self.hopLength
        predictionsDict['durationOfSegment'] = self.durationOfSegment

        if self.valMode:
            predictionsDict['gT'] = self.gT
            predictionsDict['evalScores'] = self.evalScores
            #predictionsDict['threshold'] = self.threshold
            predictionsDict['tresholdsPerClass'] = self.tresholdsPerClass
            


        #pickle.dump(predictionsDict, open(dstDir + 'predictionsDict.pkl', 'wb'))
        with open(dstDir + 'predictionsDict.pkl', 'wb') as handle:
            pickle.dump(predictionsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


def applyRandomFilter(input):

    # type: lowpass highpass bandpass bandstop
    # lowcut > 0
    # highcut < SampleRate/2

    order = random.randint(1,5)

    typeIx = random.randint(0,3)

    if typeIx == 0: type = 'lowpass'
    if typeIx == 1: type = 'highpass'
    if typeIx == 2: type = 'bandpass'
    if typeIx == 3: type = 'bandstop'

    # Create filter params
    nyq = 0.5 * SampleRate

    if type.startswith('band'):
        # band pass or stop
        lowcut = random.uniform(1, nyq-1)
        highcut = random.uniform(lowcut+1, nyq-1)
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype=type)
    else:
        # low or high pass
        freqcut = random.uniform(1, nyq-1)
        freq = freqcut / nyq
        b, a = butter(order, freq, btype=type)

    output = lfilter(b, a, input)

    # If anything went wrong (nan in array or max > 1.0 or min < -1.0) --> return original input
    if np.isnan(output).any() or np.max(output) > 1.0 or np.min(output) < -1.0:
        output = input

    #print(type, order, np.min(input), np.max(input), np.min(output), np.max(output))

    return output


def applySpectralTimeStretch(SpecImage, Opt):

    #print(SpecImage.shape)

    ImageHeightOrg = SpecImage.shape[0]
    ImageWidthOrg = SpecImage.shape[1]
    ImageDTypeOrg = SpecImage.dtype

    SpecImageNew = np.zeros((ImageHeightOrg, 0), dtype=ImageDTypeOrg)
    
    StartPos = 0
    while StartPos < ImageWidthOrg:

        SegmWidth = random.randrange(Opt['SegmWidthMin'], Opt['SegmWidthMax'])
        EndPos = StartPos+SegmWidth

        if EndPos > ImageWidthOrg:
            EndPos = ImageWidthOrg
            SegmWidth = EndPos-StartPos

        Segm = SpecImage[:,StartPos:EndPos]

        # Resize via Pil
        SegmWidthNew = int(SegmWidth*random.uniform(Opt['StretchFactorMin'], Opt['StretchFactorMax']))

        if SegmWidthNew and SegmWidthNew != SegmWidth:
            Segm = Image.fromarray(Segm.astype(np.uint8))
            InterpolationMethod = Image.LANCZOS
            Segm = Segm.resize((SegmWidthNew, ImageHeightOrg), InterpolationMethod) 
            Segm = np.array(Segm, dtype=ImageDTypeOrg)

        SpecImageNew = np.concatenate((SpecImageNew, Segm), axis=1)

        StartPos = EndPos

    #print(SpecImageNew.shape)
    return SpecImageNew

def applySpectralFreqStretch(SpecImage, Opt):

    #print(SpecImage.shape)

    ImageHeightOrg = SpecImage.shape[0]
    ImageWidthOrg = SpecImage.shape[1]
    ImageDTypeOrg = SpecImage.dtype

    SpecImageNew = np.zeros((0, ImageWidthOrg), dtype=ImageDTypeOrg)
    
    StartPos = 0
    while StartPos < ImageHeightOrg:

        SegmHeight = random.randrange(Opt['SegmHeightMin'], Opt['SegmHeightMax'])
        EndPos = StartPos+SegmHeight

        if EndPos > ImageHeightOrg:
            EndPos = ImageHeightOrg
            SegmHeight = EndPos-StartPos

        Segm = SpecImage[StartPos:EndPos, :]

        # Resize via Pil
        SegmHeightNew = int(SegmHeight*random.uniform(Opt['StretchFactorMin'], Opt['StretchFactorMax']))

        if SegmHeightNew and SegmHeightNew != SegmHeight:
            Segm = Image.fromarray(Segm.astype(np.uint8))
            InterpolationMethod = Image.LANCZOS
            Segm = Segm.resize((ImageWidthOrg, SegmHeightNew), InterpolationMethod) 
            Segm = np.array(Segm, dtype=ImageDTypeOrg)

        #print(SpecImageNew.shape, Segm.shape)
        SpecImageNew = np.concatenate((SpecImageNew, Segm), axis=0)

        StartPos = EndPos

    #print(SpecImageNew.shape)
    return SpecImageNew



def readAudioFromRandomPosition(audioPart, DurationOfSegmentInSamples, AmpExpMin, AmpExpMax, Augmentation=True):

    
    path = audioPart['path']
    startSample = audioPart['startSample']
    endSample = audioPart['endSample']


    DebugTemp = False #True False # later raus

    # Parameters

    DurationOfChunkMin = 0.5 #0.1
    DurationOfChunkMax = 4.0 #2.0


    TimeStretchFactorSigma = 0.05 # 0.05, 0.1
    PitchShiftInQuarterTonesSigma = 0.5

    ChunkDropoutMaxNum = 1 #2
    ChunkDropoutChance = 15 #10


    nFrames = endSample - startSample # duration of audio part in samples
    assert nFrames > 0


        # Get random start pos
    if DebugTemp:
        offset = int(0.5*nFrames)
    else:
        offset = int(nFrames * random.random())



    # Read from random position and apply time domain augmentation (dropout, time stretch, pitch shift)
    if Augmentation and ChunkAugmentationChance and random.randrange(100) < ChunkAugmentationChance:

        # Get SampleVec twice as long (because of time stretching & chunk dropout)
        DurationOfSegmentInSamplesPlusBuffer = int(DurationOfSegmentInSamples*2.0)

        startIx = startSample + offset
        endIx = startIx + DurationOfSegmentInSamplesPlusBuffer


        if endIx < endSample:
            audioData = sf.read(path, start=startIx, stop=endIx, always_2d=True)[0]
            if audioData.shape[1] > 1:
                channelIx = random.randrange(3) # 0,1: Channels, 2: MonoMix
                if channelIx == 2:
                    SampleVecPlusBuffer = audioData[:,0]
                    SampleVecPlusBuffer += audioData[:,1]
                    SampleVecPlusBuffer *= 0.5
                else:
                    SampleVecPlusBuffer = audioData[:,channelIx]
            else:
                SampleVecPlusBuffer = audioData[:,0]

        else:
            # Read entire part
            audioData = sf.read(path, start=startSample, stop=endSample, always_2d=True)[0]
            if audioData.shape[1] > 1:
                channelIx = random.randrange(3) # 0,1: Channels, 2: MonoMix
                if channelIx == 2:
                    SampleVecPlusBuffer = audioData[:,0]
                    SampleVecPlusBuffer += audioData[:,1]
                    SampleVecPlusBuffer *= 0.5
                else:
                    SampleVecPlusBuffer = audioData[:,channelIx]
            else:
                SampleVecPlusBuffer = audioData[:,0]

            endIx = offset + DurationOfSegmentInSamplesPlusBuffer
            SampleVecRepeated = SampleVecPlusBuffer.copy()
            while endIx > SampleVecRepeated.shape[0]:
                SampleVecRepeated = np.append(SampleVecRepeated, SampleVecPlusBuffer)
            SampleVecPlusBuffer = SampleVecRepeated[offset:endIx]


        # NNN
        if not SampleVecPlusBuffer.flags['F_CONTIGUOUS']:
            SampleVecPlusBuffer = np.asfortranarray(SampleVecPlusBuffer)



        # Read chunks from SampleVecPlusBuffer
        startIx = 0
        ChunkDropoutCounter = 0
        SampleVec = np.empty((0), dtype=np.float64)

        while len(SampleVec) < DurationOfSegmentInSamples:

            #print(path, SampleVecPlusBuffer.shape)

            # Random chunk duration
            DurationOfChunk = random.uniform(DurationOfChunkMin, DurationOfChunkMax)
            
            DurationOfChunkInSamples = int(DurationOfChunk*SampleRate)

            endIx = startIx + DurationOfChunkInSamples

            ChunkSampleVec = SampleVecPlusBuffer[startIx:endIx]

            # Only apply augmenation if length not zero (happens very rarely)
            if ChunkSampleVec.shape[0]:

                # 0: TimeStretch, 1: PitchShift, 2: both
                case = int(3*random.random())

                if case == 0 or case == 2:
                    # Apply time stretching
                    TimeStretchFactor = random.gauss(1.0, TimeStretchFactorSigma)
                    
                    if abs(1.0-TimeStretchFactor) > 0.01:
                        ChunkSampleVec = librosa.effects.time_stretch(ChunkSampleVec, TimeStretchFactor)

                if case == 1 or case == 2:
                    # Apply pitch shifting
                    PitchShiftInQuarterTones = random.gauss(0.0, PitchShiftInQuarterTonesSigma)
                    
                    if abs(PitchShiftInQuarterTones) > 0.1:
                        ChunkSampleVec = librosa.effects.pitch_shift(ChunkSampleVec, SampleRate, n_steps=PitchShiftInQuarterTones, bins_per_octave=24)

                # Append to SampleVec (with dropout)
                if ChunkDropoutCounter < ChunkDropoutMaxNum and random.randrange(100) < ChunkDropoutChance:
                    ChunkDropoutCounter += 1
                else:
                    SampleVec = np.append(SampleVec, ChunkSampleVec)



            startIx = endIx

        # Cut to DurationOfSegmentInSamples
        SampleVec = SampleVec[:DurationOfSegmentInSamples]

    else:

        # No time domain augmentation (dropout, time stretch, pitch shift)

        startIx = startSample + offset
        endIx = startIx + DurationOfSegmentInSamples


        if endIx < endSample:
            audioData = sf.read(path, start=startIx, stop=endIx, always_2d=True)[0]
            if audioData.shape[1] > 1:
                channelIx = random.randrange(3) # 0,1: Channels, 2: MonoMix
                if channelIx == 2:
                    SampleVec = audioData[:,0]
                    SampleVec += audioData[:,1]
                    SampleVec *= 0.5
                else:
                    SampleVec = audioData[:,channelIx]
            else:
                SampleVec = audioData[:,0]

        else:
            # Read entire part
            audioData = sf.read(path, start=startSample, stop=endSample, always_2d=True)[0]
            if audioData.shape[1] > 1:
                channelIx = random.randrange(3) # 0,1: Channels, 2: MonoMix
                if channelIx == 2:
                    SampleVec = audioData[:,0]
                    SampleVec += audioData[:,1]
                    SampleVec *= 0.5
                else:
                    SampleVec = audioData[:,channelIx]
            else:
                SampleVec = audioData[:,0]

            endIx = offset + DurationOfSegmentInSamples
            SampleVecRepeated = SampleVec.copy()
            while endIx > SampleVecRepeated.shape[0]:
                SampleVecRepeated = np.append(SampleVecRepeated, SampleVec)
            SampleVec = SampleVecRepeated[offset:endIx]


        # NNN
        if not SampleVec.flags['F_CONTIGUOUS']:
            SampleVec = np.asfortranarray(SampleVec)



    
    # Apply random filter
    if Augmentation and FinalFilterChance and random.randrange(100) < FinalFilterChance:
        SampleVec = applyRandomFilter(SampleVec)
        if FinalFilterChance2 and random.randrange(100) < FinalFilterChance2:
            SampleVec = applyRandomFilter(SampleVec)



    if CyclicShiftChance and random.randrange(100) < CyclicShiftChance:
        Shift = random.randint(0, len(SampleVec)-1)
        SampleVec = np.roll(SampleVec, Shift)


    # Apply Amp Factor
    if not DebugTemp:
        AmpFactor = 10**(random.uniform(AmpExpMin, AmpExpMax))
        SampleVec *= AmpFactor


    
    return SampleVec


def getMelSpec(SampleVec, FftSizeInSamples, FftHopSizeInSamples, Augmentation=True):

    # Librosa mel-spectrum

    # print(SampleVec.shape)
    # print(SampleVec.flags['F_CONTIGUOUS'])
    
    MelSpec = librosa.feature.melspectrogram(y=SampleVec, sr=SampleRate, n_fft=FftSizeInSamples, hop_length=FftHopSizeInSamples, n_mels=NumOfMelBands, fmin=MelStartFreq, fmax=MelEndFreq, power=2.0)


    # Convert power spec to dB scale (compute dB relative to peak power)
    MelSpec = librosa.power_to_db(MelSpec, ref=np.max, top_db=100)
    


    # Freq Jitter
    if Augmentation and NumOfLowFreqsInPixelToCutMax and NumOfHighFreqsInPixelToCutMax:
        NumOfLowFreqsInPixelToCut = random.randrange(NumOfLowFreqsInPixelToCutMax)
        NumOfHighFreqsInPixelToCut = random.randrange(NumOfHighFreqsInPixelToCutMax)
    else:
        NumOfLowFreqsInPixelToCut = int(NumOfLowFreqsInPixelToCutMax/2.0)
        NumOfHighFreqsInPixelToCut = int(NumOfHighFreqsInPixelToCutMax/2.0)

    if NumOfHighFreqsInPixelToCut:
        #MelSpec[NumOfLowFreqsInPixelToCut:-NumOfHighFreqsInPixelToCut] = 0.0
        MelSpec = MelSpec[NumOfLowFreqsInPixelToCut:-NumOfHighFreqsInPixelToCut]
    else:
        #MelSpec[NumOfLowFreqsInPixelToCut:] = 0.0
        MelSpec = MelSpec[NumOfLowFreqsInPixelToCut:]

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    MelSpec = MelSpec[::-1, ...]



    # Normalize values between 0 and 1 (& prevent divide by zero)
    MelSpec -= MelSpec.min()
    MelSpecMax = MelSpec.max()
    if MelSpecMax: 
        MelSpec /= MelSpecMax

    MaxVal = 255.9
    MelSpec *= MaxVal
    MelSpec = MaxVal-MelSpec

    
    return MelSpec


def resizeSpecImage(SpecImage, ImageSize, RandomizeInterpolationFilter=True):

    SpecImagePil = Image.fromarray(SpecImage.astype(np.uint8))
    #scipy.misc.imresize(arr, size, interp='bilinear', mode=None)[source]
    # Upsampling: Image.BILINEAR, Downsampling: Image.ANTIALIAS ? --> Could be chosen random for augmentation ?
    

    if RandomizeInterpolationFilter and VaryInterpolationFilterChance and random.randrange(100) < VaryInterpolationFilterChance:
        FilterChoiceIx = random.randint(0,4)
        if FilterChoiceIx == 0: InterpolationMethod = Image.NEAREST
        if FilterChoiceIx == 1: InterpolationMethod = Image.BOX
        if FilterChoiceIx == 2: InterpolationMethod = Image.BILINEAR
        if FilterChoiceIx == 3: InterpolationMethod = Image.HAMMING
        if FilterChoiceIx == 4: InterpolationMethod = Image.BICUBIC
    else:
        InterpolationMethod = Image.LANCZOS

    SpecImagePil = SpecImagePil.resize(ImageSize, InterpolationMethod) 
    # Cast to int8
    SpecImageResized = np.array(SpecImagePil, dtype=np.uint8)
    #print('SpecImageResized.shape', SpecImageResized.shape)
    return SpecImageResized





def log(LogFID, String, PrintToTerminal):
    with open(LogFID, 'a') as LogFile: LogFile.write(String + '\n')
    LogFile.close()
    if PrintToTerminal:
        print(String)



def main():

    global EpoRes # Maybe pass as argument (like args)

    args = parser.parse_args()


    #BatchSizeForTesting = args.batch_size*BatchSizeFactorForTesting
    log(InterimResultsFid, '', False) # print newline
    log(InterimResultsFid, str(sys.argv), False) # print arguments

    # Print header to log files
    #HeaderStr = 'DataTime \t EpochIx \t Progress \t LRAP \t cMAP \t f1'
    HeaderStr = 'DateTime \t EpochIx \t LR \t LRAP \t cMAP \t f1 \t Progress' 
    
    log(EpochSummeryFid, HeaderStr, False)
    log(EpochProgressFid, HeaderStr, False)
    
    
    # NNN 
    # Loaded locally saved inception model because of current inception loading bug
    ModelFid = RootDir + 'KaggleBirdsongRecognition2020/data/inception_v3_pretrained_google-1a9a5a14.pth.tar'
    #torch.save(model, ModelFid)
    #model = torch.load(ModelFid)
    #print('Loaded locally saved inception model because of current inception loading bug.')


    # Create model

    # if args.pretrained:

    #     if ModelName == 'inception_v3':
    #         model = torch.load(ModelFid)
    #         print('Loaded locally saved inception model because of current inception loading bug.')
    #     else:
    #         print("=> using pre-trained model '{}'".format(ModelName))
    #         model = models.__dict__[ModelName](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(ModelName))
    #     model = models.__dict__[ModelName]()


    if ModelName == 'inception_v3':
        model = torch.load(ModelFid)
        model.aux_logits = False
        print('Loaded locally saved inception model because of current inception loading bug.')

    if 'resnet' in ModelName:
        print("=> using pre-trained model '{}'".format(ModelName))
        model = models.__dict__[ModelName](pretrained=True)
        if UseAdaptiveMaxPool2d:
            model.avgpool = nn.AdaptiveMaxPool2d(output_size=1)

    if 'efficientnet' in ModelName:
        model = EfficientNet.from_pretrained(ModelName, num_classes=NumOfClasses)
        if UseAdaptiveMaxPool2d:
            model._avg_pooling = nn.AdaptiveMaxPool2d(output_size=1)
    
    else:

        NumOfFeatures = model.fc.in_features

        if IncludeSigmoid:
            model.fc = nn.Sequential(
                nn.Linear(NumOfFeatures, NumOfClasses), 
                nn.Sigmoid())
        else:
            model.fc = nn.Linear(NumOfFeatures, NumOfClasses)


    # # Test
    # print('model._conv_stem', model._conv_stem)
    # from efficientnet_pytorch import utils
    # model._conv_stem = utils.Conv2dStaticSamePadding(3, 40, kernel_size=(3, 3), stride=(2, 2), image_size = ImageSize, bias=False)
    
    # #EfficientNet

    # #print(model)
    # print('model._conv_stem', model._conv_stem)
    # #print('model._conv_stem', model._conv_stem)

    # # Conv2dStaticSamePadding(3, 40, kernel_size=(3, 3), stride=(2, 2), bias=False (static_padding): ZeroPad2d(padding=(0, 1, 0, 1), value=0.0))
    # # def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):




    # Define loss function (criterion) and optimizer
    
    if UseMultiLabel:
        if IncludeSigmoid:
            criterion = nn.BCELoss().cuda()
        else:
            criterion = nn.BCEWithLogitsLoss().cuda()
            #criterion = nn.MultiLabelSoftMarginLoss().cuda(args.gpu) # or MultiLabelMarginLoss ?
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    

    if OptimizerType == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if OptimizerType == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if SchedulerType == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    else:
        scheduler = None



    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            #best_acc1 = checkpoint['best_acc1']
            
            if not ResetBestValResults:
                if 'EpoRes' in checkpoint:
                    EpoRes = checkpoint['EpoRes']

                    # (if new metric was added after resume from older checkpoint)
                    if 'lrapBest' not in EpoRes: EpoRes['lrapBest'] = 0
                    if 'cMapBest' not in EpoRes: EpoRes['cMapBest'] = 0
                    if 'f1Best' not in EpoRes: EpoRes['f1Best'] = 0

            
            # To remove later (handle model saved with DataParallel not removed)
            state_dict = checkpoint['state_dict']
            firstStateDictKey = next(iter(state_dict))
            if 'module.' in firstStateDictKey:
                print('Remove module from state_dict keys')
                # Get State Dict without DataParallel (create new OrderedDict that does not contain `module.`)
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    #print(k)
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                state_dict = new_state_dict
                checkpoint['state_dict'] = state_dict


            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

            EpoRes['Ix'] = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    # Check for module. in state_dict (if not apply DataParallel to use multiple gpus for training)
    firstStateDictKey = next(iter(model.state_dict()))
    if not 'module.' in firstStateDictKey:
        print('Apply torch.nn.DataParallel to model.')
        model = torch.nn.DataParallel(model).cuda()




    normalize = transforms.Normalize(mean=ImageMean, std=ImageStd)


    trainAudioDatasetObj = AudioDatasetTrain(train_df, noise_df, transform=transforms.Compose([
                                transforms.ColorJitter(brightness=ColorJitterAmount, contrast=ColorJitterAmount, saturation=ColorJitterAmount, hue=0.01),
                                transforms.ToTensor(),
                                normalize]))

    print('NumOfTrainParts', len(trainAudioDatasetObj))



    #print('args.worker', args.worker)
    trainLoader = torch.utils.data.DataLoader(trainAudioDatasetObj, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    

    valAudioDatasetObj = AudioDatasetTest(val_df, classIdsTrain, transform=transforms.Compose([transforms.ToTensor(),normalize]))

    valLoader = torch.utils.data.DataLoader(valAudioDatasetObj, batch_size=BatchSizeForTesting, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    #print('NumOfValSoundscapes', len(FileNamesValSoundscapesVal))
    print('NumOfValSoundscapesSegments', len(valAudioDatasetObj))


    if args.evaluate:
        validate(model, criterion, args, valLoader, valAudioDatasetObj)
        return

    for epoch in range(args.start_epoch, args.epochs):

        EpoRes['Ix'] = epoch

        # To log learning rate used for last epoch
        if SchedulerType:
            #lrStr = '%.4f'%scheduler.get_lr()[0]
            lrStr = '%.6f'%scheduler.get_last_lr()[0]
        else:
            lrStr = str(args.lr)



        # train for one epoch
        train(trainLoader, model, criterion, optimizer, scheduler, epoch, args)

        # Save checkpoint (state_dict without module DataParallel stuff)
        # ToDos: add config stuff 
        state_dict = model.state_dict()
        firstStateDictKey = next(iter(state_dict))
        if 'module.' in firstStateDictKey:
            #print('Remove module from state_dict keys')
            state_dict = model.module.state_dict()

        # Save model checkpoint after training epoch
        checkpoint = {}
        checkpoint['epoch'] = epoch + 1
        checkpoint['arch'] = ModelName
        checkpoint['state_dict'] = state_dict
        checkpoint['EpoRes'] = EpoRes
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if SchedulerType:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        else:
            checkpoint['scheduler_state_dict'] = None

        
        checkpointFid = LogDir + 'LastCheckpoint.pth.tar'
        torch.save(checkpoint, checkpointFid)



        # evaluate on validation set
        if epoch >= NumOfEpochsBeforeFirstValidation and epoch % NumOfEpochsBeforeNextValidation == 0:

            lrap, cMap, f1 = validate(model, criterion, args, valLoader, valAudioDatasetObj)

            # Remember best results
            progress = []
            if lrap > EpoRes['lrapBest']:
                EpoRes['lrapBest'] = lrap
                progress.append('lrap')
            if cMap > EpoRes['cMapBest']:
                EpoRes['cMapBest'] = cMap
                progress.append('cMap')
            if f1 > EpoRes['f1Best']:
                EpoRes['f1Best'] = f1
                progress.append('f1')



            

            EpochSummeryStr = time.strftime("%y%m%d-%H%M%S")
            EpochSummeryStr += '\t' + str(EpoRes['Ix'])
            EpochSummeryStr += '\t' + lrStr
            EpochSummeryStr += '\t' + '%.3f'%(lrap*100)
            EpochSummeryStr += '\t' + '%.3f'%(cMap*100.0)
            EpochSummeryStr += '\t' + '%.3f'%(f1*100.0)
            EpochSummeryStr += '\t' + str(progress)

            #print(EpochSummeryStr)
            log(EpochSummeryFid, EpochSummeryStr, True)
            log(InterimResultsFid, EpochSummeryStr, False)

            # Remember Best Model
            if len(progress):
                log(EpochProgressFid, EpochSummeryStr, False)
                checkpointWithProgressFid = LogDir + time.strftime("%y%m%d-%H%M%S") + '-Checkpoint' + (str(epoch)).zfill(2) + '.pth.tar'
                shutil.copyfile(checkpointFid, checkpointWithProgressFid)


def train(trainLoader, model, criterion, optimizer, scheduler, epoch, args):

    global EpoRes

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, sample_batched in enumerate(trainLoader):

        input = sample_batched['SpecImage']
        target = sample_batched['GroundTruth']

        # measure data loading time
        data_time.update(time.time() - end)
        
        #if args.gpu is not None:
        #    input = input.cuda(args.gpu, non_blocking=True)
        
        target = target.cuda(None, non_blocking=True)


        # # compute output
        # if ModelName == 'inception_v3':
        #     output, aux = model(input) # Not working if multi gpu !!!
        # else:
        #     output = model(input)

        output = model(input)

        loss = criterion(output, target)
        
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()



        if i % args.print_freq == 0:
            LogStr = time.strftime("%y%m%d-%H%M%S") + '\t'
            LogStr += ('Epoch [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} \t'
                  'Data {data_time.val:.3f} \t'
                  'Loss {loss.val:.4f} '.format(
                   epoch, i, len(trainLoader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            #print(LogStr)
            log(InterimResultsFid, LogStr, PrintInterimResult)

        
        if NumOfIterationsToSaveCheckpointBetweenEpochs and i % NumOfIterationsToSaveCheckpointBetweenEpochs == 0:

            # Save checkpoint (state_dict without module DataParallel stuff)
            # ToDos: add config stuff 
            state_dict = model.state_dict()
            firstStateDictKey = next(iter(state_dict))
            if 'module.' in firstStateDictKey:
                print('Remove module from state_dict keys')
                state_dict = model.module.state_dict()

            checkpoint = {}
            checkpoint['epoch'] = epoch + 1
            checkpoint['arch'] = ModelName
            checkpoint['state_dict'] = state_dict
            checkpoint['EpoRes'] = EpoRes
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if SchedulerType:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            else:
                checkpoint['scheduler_state_dict'] = None

            checkpointFid = time.strftime("%y%m%d-%H%M%S") + '-Checkpoint' + (str(epoch)).zfill(2) + '-' + (str(i)).zfill(5) +'.pth.tar'
            torch.save(checkpoint, checkpointFid)

    # Update learing rate
    if SchedulerType:
        scheduler.step()



def validate(model, criterion, args, valLoader, valAudioDatasetObj):

    global EpoRes

    model.eval() # switch to evaluate mode


    lrap = 0.0


    ####  Predict Val Soundscapes  ####

    batch_time = AverageMeter()
    
    # Inits to get predictions per file and time interval
    nParts = valAudioDatasetObj.nParts


    with torch.no_grad():

        end = time.time()
        for i, sample_batched in enumerate(valLoader):

            input = sample_batched['SpecImage']

            segmentIxs = sample_batched['segmentIx']

            output = model(input)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                LogStr = time.strftime("%y%m%d-%H%M%S") + '\t'
                LogStr += ('Test2 [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} '.format(
                       i, len(valLoader), batch_time=batch_time))
                log(InterimResultsFid, LogStr, PrintInterimResult)


            # NNN
            # Collect predictions
            valAudioDatasetObj.collectPredictionsPerBatch(segmentIxs, output.cpu().data.numpy())
    



    # Get evaluation metrics
    if valAudioDatasetObj.valMode:

        metrics =  valAudioDatasetObj.eval()
        # Temp
        lrap = metrics['lrap']
        cMap = metrics['cMap']
        f1 = metrics['f1']




    # For evaluation only
    if args.evaluate:
        CheckpointIx = EpoRes['Ix']-1
        EpochSummeryStr = time.strftime("%y%m%d-%H%M%S")
        EpochSummeryStr += '\t' + str(CheckpointIx)
        EpochSummeryStr += '\t' + str(None)
        EpochSummeryStr += '\t' + '%.3f'%(lrap*100.0)
        EpochSummeryStr += '\t' + '%.3f'%(cMap*100.0)
        EpochSummeryStr += '\t' + '%.3f'%(f1*100.0)
        EpochSummeryStr += '\t' + str(None)

        #print(EpochSummeryStr)
        log(EpochSummeryFid, EpochSummeryStr, True)

        
        # Save predictions dict
        checkpointDir = str(os.path.dirname(args.resume))
        if '/' in checkpointDir:
            dstRootDir = checkpointDir + '/Predictions/'
        else:
            dstRootDir = './Predictions/'
        print('dstRootDir', dstRootDir)
        if not os.path.exists(dstRootDir): os.makedirs(dstRootDir)
        dstDir = dstRootDir + 'CP' + str(CheckpointIx) + '/'
        if not os.path.exists(dstDir): os.makedirs(dstDir)

        # NNN
        # Create and export predictionsDict
        valAudioDatasetObj.exportPredictionsDict(dstDir)

        # Save checkpoint (state_dict without module DataParallel stuff)
        # ToDos: add config stuff 
        state_dict = model.state_dict()
        firstStateDictKey = next(iter(state_dict))
        if 'module.' in firstStateDictKey:
            print('Remove module from state_dict keys')
            state_dict = model.module.state_dict()

        checkpoint = {}
        checkpoint['epoch'] = CheckpointIx
        checkpoint['arch'] = ModelName
        checkpoint['state_dict'] = state_dict
        checkpoint['EpoRes'] = EpoRes
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        # if SchedulerType:
        #     checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        # else:
        #     checkpoint['scheduler_state_dict'] = None

        # To remove later
        cfg = {}

        cfg['RootDir'] = RootDir

        cfg['RunId'] = None

        cfg['ModelName'] = ModelName
        cfg['ImageSize'] = ImageSize
        cfg['ImageMean'] = ImageMean
        cfg['ImageStd'] = ImageStd

        cfg['UseAdaptiveMaxPool2d'] = UseAdaptiveMaxPool2d


        cfg['SampleRate'] = SampleRate

        cfg['ResampleType'] = 'kaiser_best' # 'kaiser_best'

        cfg['FftSizeInSamples'] = FftSizeInSamples
        cfg['FftHopSizeInSamples'] = FftHopSizeInSamples

        cfg['SegmentDuration'] = SegmentDuration
        cfg['NumOfMelBands'] = NumOfMelBands
        cfg['MelStartFreq'] = MelStartFreq
        cfg['MelEndFreq'] = MelEndFreq
        cfg['NumOfLowFreqsInPixelToCutMax'] = NumOfLowFreqsInPixelToCutMax
        cfg['NumOfHighFreqsInPixelToCutMax'] = NumOfHighFreqsInPixelToCutMax
        
        cfg['UseMultiLabel'] = UseMultiLabel

        cfg['ClassIds'] = ClassIds

        checkpoint['cfg'] = cfg

        checkpointFid =  dstDir + 'Checkpoint' + (str(CheckpointIx)).zfill(2) +'.pth.tar'
        torch.save(checkpoint, checkpointFid)



    return lrap, cMap, f1



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




if __name__ == '__main__':
    main()
    print('ElapsedTime [s]: ', (time.time() - timeStampStart))

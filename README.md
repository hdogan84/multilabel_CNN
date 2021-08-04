
# Requirements
USE: Docker image from kaggle
gcr.io/kaggle-gpu-images/python

pip install audiomentations
# List of audiomentations
## Waveform transforms
AddBackgroundNoise
AddGaussianNoise
AddGaussianSNR
AddImpulseResponse
AddShortNoises
ClippingDistortion
FrequencyMask
Gain
Mp3Compression
LoudnessNormalization
Normalize
PitchShift
PolarityInversion
Resample
Shift
TimeMask
TimeStretch
Trim
## Spectrogram transforms
SpecChannelShuffle
SpecFrequencyMask

# DataSet multi label
1. file_dict (key: file_id)
  
2. annotation_interval_dict(key: annotation_interval_id)
  * filepath
  * channel_count
  * annotation_interval_id
  * start
  * end
  * events
    * class_id
    * start_time
    * end_time
    * type
3. segments_list
   * annotation_interval_id
   * start_time
->
filepath, [start_time, end_time], class_ids
-> 
samplevec, class_ids_vec

## config parameters
* segment_duration
* min_event_overlap_time
* wrap_around_probability

# TODO 
-Script starting docker

# Build classficator for production
1. Run Build script ./src/build.py
2. copy mar file in build folder to your torchserver an activate it

# Run torchserve for testing
```bash
# got to torchserve folder an run
docker-compose up
```


# export validation matrix
matrix
channel x segment x prediciton

channel x segment x ground truth


## web-service <- classticator api
```js
{
    "fileId": str // file name
    "classIds": [ str,..n ] // Latin names of Classes
    "channels": [
        	[
        {
        	startTime: float // (in s)
        	endTime: float // (in s)
        	predictions: {
        	    logits: [ float,...n]  // prediction for class n number of classes
        	}
        },...
    ]
}
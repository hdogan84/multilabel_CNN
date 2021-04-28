
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

# DataSet multilabel
1. file_dict (key: file_id)
  * file_id
  * filepath
  * channel_count
2. annotation_interval_dict(key: annotation_interval_id)
  * annotation_interval_id
  * file_id
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

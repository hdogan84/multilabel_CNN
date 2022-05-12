
# Project Structure
## root level
```bash
* build # ouput folder of script to build torchserver models
* config # contains all config files for training an validation scripts
* data # contains linked to data folders
* logs # folder contains all experiment logs an created checkpoints
* src # src folder
* torchserve # fodlder of torchserve runtime data an docker-compose file for development
```
## src folder
```bash
* augmentation # augementation function
** image # all augementations on the image(spectrogram) based on https://github.com/albumentations-team/albumentations
** signal # all augementations on the waveform based on audiomentations https://github.com/iver56/audiomentations
* data_module # pytorch_lightning LightningDataModules 
* dataset #  torch Datasets
* model # models based on pl.LightningModule. Use BaseBirdDetector as Partent class for basic logging
* scripts # scripts for different purpusis, building torchserve modles, validate models and create predicttion json files
* tools # helper modules 
* torchserve_handler # torchserver model handlers for production
```

# Best Practices
* New Experiment new config file
* *Better create a new datamodule/dataset/model than create to much if conditions in one Class




# List of audiomentations
## Waveform transforms
AddBackgroundNoise
AddGaussianNoise/
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

# Build classficator for production
1. Run Build script ./src/build.py
2. copy mar file in build folder to your torchserver an activate it

# Run torchserve for testing
```bash
# got to torchserve folder an run
docker-compose up
```

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
```
# How to test/debug torchserve handlers
1. Build mar file with build script
2. Start torchserver with docker-compose file
3. register model
```bash
curl -X POST "http://localhost:8081/models?url=YOUR_PACKAGE_NAME.mar&initial_workers=1"
```
4. run inferencing
```bash
curl http://localhost:8080/predictions/YOUR_PACKAGE_NAME/1 -F "data=@./test-assets/test.wav"
```
5. or run the run_handler_test.sh
# custom handler file

# model_handler.py

"""
ModelHandler defines a Base audio model handler.
"""
import io
import librosa
import math
import torch
import numpy as np
import yaml
import os
import random
import string
import itertools
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label
from torchvision import transforms


class AudioHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    sample_rate = 32000
    fft_size_in_samples = 1536
    fft_hop_size_in_samples = 360
    num_of_mel_bands = 128
    mel_start_freq = 20
    mel_end_freq = 16000
    normalize_mean = 0.456
    normalize_std = 0.224
    segment_duration = 5

    segment_step = 1
    batch_size = 64 * 5
    convert_to_mono = False

    __config__ = None

    def __get_mel_spec__(
        self,
        audio_data,
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.fft_size_in_samples,
            hop_length=self.fft_hop_size_in_samples,
            n_mels=self.num_of_mel_bands,
            fmin=self.mel_start_freq,
            fmax=self.mel_end_freq,
            power=2.0,
        )

        # Convert power spec to dB scale (compute dB relative to peak power)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max, top_db=100)

        # Flip spectrum vertically (only for better visialization, low freq. at bottom)
        mel_spec = mel_spec[::-1, ...]

        # Normalize values between 0 and 1 (& prevent divide by zero)
        mel_spec -= mel_spec.min()
        mel_spec_max = mel_spec.max()
        if mel_spec_max:
            mel_spec /= mel_spec_max

        max_val = 255.9
        mel_spec *= max_val
        mel_spec = max_val - mel_spec

        return mel_spec

    def __mapping_as_array__(self):
        keys = self.mapping.keys()
        result = [None] * len(list(keys))
        for i in keys:
            result[int(i)] = self.mapping[i]
        return result

    def __calc_step_count__(self, duration):
        # calculate size of oversizing
        b = (
            math.floor(duration / self.segment_step) * self.segment_step
            + self.segment_duration
        )
        over = (b - duration) / self.segment_step
        # minimal step count for reaching whole duration
        return math.ceil(math.floor(duration / self.segment_step) - over)

    def __calc_steps__(self, duration):
        step_count = self.__calc_step_count__(duration)
        steps = []
        for n in range(step_count):
            start = n * self.segment_step
            end = n * self.segment_step + self.segment_duration
            steps.append(
                (start, end if end > self.segment_duration else self.segment_duration)
            )
        start = duration - self.segment_duration
        end = duration
        steps.append((start if start >= 0 else 0, duration))
        return steps

    def initialize(self, context):
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        config_filepath = os.path.join(model_dir, "config.yaml")
        if os.path.isfile(config_filepath):
            with open(config_filepath) as f:
                self.__config__ = yaml.safe_load(f)
        if self.__config__ is not None:
            for key in self.__config__.keys():
                if hasattr(self, key):
                    self.__dict__[key] = self.__config__[key]

        # print(self.manifest)
        # print(self.mapping)
        # raise Exception("Sorry, no numbers below zero")

    def preprocess_audio_data(self, audio_data):
        return audio_data

    def create_spectogram(self, audio_data):
        """The spectorgram creation function of an audio data array r

        Args:
            audio_data (numpy array):

        Returns:
           numpy array[3,width,height]: image data
        """
        mel_spec = self.__get_mel_spec__(audio_data)
        # format mel_spec to image with one channel
        h, w = mel_spec.shape
        image_data = np.empty((h, w, 1), dtype=np.uint8)
        image_data[:, :, 0] = mel_spec
        return image_data

    def postprocess_spec(self, image_data):
        """The normalise image function

        Args:
            numpy array[3,width,height]: image data
        Returns:
           torch tensor: image data
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )

        tensor = transform(image_data)
        return tensor

    def preprocess(self, data):
        """The preprocess function of an audiofile program converts the input data to a float list of float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input audio as a list of float tensors.
        """
        if len(data) > 1:
            raise Exception("Audiohandler is not capable of handling batch requests")

        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        audio = data[0].get("data") or data[0].get("body")
        if isinstance(audio, str):
            # if the audio is a string of bytesarray.
            raise Exception(
                "Sending audio as base64 coded string is not supported only file upload"
            )

        # If the audio is sent as bytesarray
        if isinstance(audio, (bytearray, bytes)):

            if "TEMP_FOLDER" not in os.environ:
                raise Exception("Missing ENV Variable TEMP_FOLDER")

            raw_data = []
            sr = -1
            try:
                # only wav files can be opened by soundfile via bytestream
                raw_data, sr = librosa.load(
                    io.BytesIO(audio),
                    sr=self.sample_rate,
                    mono=self.convert_to_mono,
                    offset=0.0,
                    duration=None,
                    res_type="kaiser_best",
                )
            except RuntimeError as e:
                # file seems to be not a wav format so save und try it again
                if "Format not recognised" in str(e) or "File contains data in an unknown format" in str(e):
                    print("catches")
                    temp_filename = (
                        "".join(
                            random.SystemRandom ().choice(
                                string.ascii_letters + string.digits
                            )
                            for _ in range(20)
                        )
                        + ".temp"
                    )

                    temp_filepath = os.environ["TEMP_FOLDER"] + "/" + temp_filename
                    try:
                        out_file = open(
                            temp_filepath, "wb"
                        )  # open for [w]riting as [b]inary
                        out_file.write(audio)
                        out_file.close()
                        raw_data, sr = librosa.load(
                            temp_filepath,
                            sr=self.sample_rate,
                            mono=self.convert_to_mono,
                            offset=0.0,
                            duration=None,
                            res_type="kaiser_best",
                        )
                    finally:
                        if os.path.exists(temp_filepath):
                            os.remove(temp_filepath)
                else:
                    raise e
            # ToDo: Check raw_data dim an transform in 2-dim array if mono
            n_channels = raw_data.shape[0] if len(raw_data.shape) > 1 else 1
            if n_channels == 1:
                raw_data = np.expand_dims(raw_data, axis=0)

            processed_data = self.preprocess_audio_data(raw_data)

            duration = len(processed_data[0]) / self.sample_rate
            self.steps = self.__calc_steps__(duration)
            channel_tensors = []
            for channel in range(n_channels):
                data = []
                for step in self.steps:
                    image_data = self.create_spectogram(
                        processed_data[
                            channel,
                            int(step[0] * self.sample_rate) : int(
                                step[1] * self.sample_rate
                            ),
                        ]
                    )
                    tensor = self.postprocess_spec(image_data)
                    data.append(tensor)
                channel_tensors.append(data)

            return channel_tensors
        raise Exception("No File upload found")

    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """

        with torch.no_grad():
            results = []
            print('inference: batch size: {} '.format(self.batch_size))
          
            for channel in range(len(data)):

                batches = math.ceil(len(data[channel]) / self.batch_size)
                print('inference: Batches: {} '.format(batches))
                channel_results = []
                for batch in range(batches):

                    marshalled_data = torch.stack(
                        data[channel][
                            batch * self.batch_size : batch * self.batch_size
                            + self.batch_size
                        ]
                    ).to(self.device)
                    print('inference: Batch_tensor size: {}'.format(marshalled_data.size()))
                    predictions = self.model(marshalled_data, *args, **kwargs)
                    print('inference: Prediction size: {}'.format(predictions.size()))
                    # flatten batch results 
                    for step_result in predictions.tolist():
                        channel_results.append(step_result)
                results.append(channel_results)

            
            return results

    def postprocess(self, data):
        # create result dictionary
        result = data
        channels = []
        for channel in range(len(result)):
            channel_results = []
            print('postprocess: Result in channel {} len {}'.format(channel,len(result[channel])))

            for index, segment_data in enumerate(data[channel]):
                channel_results.append(
                    {
                        "startTime": self.steps[index][0],
                        "endTime": self.steps[index][1],
                        "predictions": {"logits": segment_data},
                    }
                )
            channels.append(channel_results)
        classIds = None if self.mapping is None else self.__mapping_as_array__()
        result_dict = {"classIds": classIds, "channels": channels}
        # torchserve expects array because auf merging requests into batches
        return [result_dict]

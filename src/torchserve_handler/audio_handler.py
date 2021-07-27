# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""
import io
import librosa
import math
import torch
import numpy as np
import yaml
import os
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

    def __calc_steps__(self, duration):
        # calculate size of oversizing
        b = (
            math.floor(duration / self.segment_step) * self.segment_step
            + self.segment_duration
        )
        over = (b - duration) / self.segment_step
        # minimal step count for reaching whole duration
        return math.ceil(math.floor(duration / self.segment_step) - over)

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
            raw_data, sr = librosa.load(
                io.BytesIO(audio),
                sr=self.sample_rate,
                mono=self.convert_to_mono,
                offset=0.0,
                duration=None,
                res_type="kaiser_best",
            )
            data = []
            channels = raw_data.shape[0]
            duration = len(raw_data[0]) / self.sample_rate
            steps = self.__calc_steps__(duration)
            channel_tensors = []
            for channel in range(channels):
                for n in range(steps):
                    mel_spec = self.__get_mel_spec__(
                        raw_data[
                            channel,
                            n
                            * self.sample_rate : (n + self.segment_duration)
                            * self.sample_rate,
                        ],
                    )
                    # format mel_spec to image with one channel
                    h, w = mel_spec.shape
                    image_data = np.empty((h, w, 1), dtype=np.uint8)
                    image_data[:, :, 0] = mel_spec
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=self.normalize_mean, std=self.normalize_std
                            ),
                        ]
                    )

                    tensor = transform(image_data)
                    data.append(tensor)
                channel_tensors.append(data)

        return channel_tensors

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
            for channel in range(len(data)):

                batches = math.ceil(len(data[channel]) / self.batch_size)
                channel_results = []
                for batch in range(batches):

                    marshalled_data = torch.stack(
                        data[channel][
                            batch * self.batch_size : batch * self.batch_size
                            + self.batch_size
                        ]
                    ).to(self.device)
                    channel_results.append(
                        self.model(marshalled_data, *args, **kwargs).to("cpu")
                    )
                results.append(torch.stack(channel_results).tolist())

        return results

    def postprocess(self, data):
        # crete result dictionary
        result_dict = {"classIds": self.__mapping_as_array__(), "channels": data}
        # torchserve expects array because auf merging requests into batches
        return [result_dict]

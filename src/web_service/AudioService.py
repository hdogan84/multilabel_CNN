from web_service.BaseService import BaseService, PreProcessedData, Predictions
from tools.audio_tools import resample_audio_file
from uuid import uuid4
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from math import ceil
from pytorch_lightning.metrics.utils import to_onehot
from dataset import AudioSet
import torch
import numpy as np
import logging
from config.configuration import ServiceConfig

logger = logging.getLogger("audio_service")


class AudioService(BaseService):
    DEFAULT_STEP_WIDTH: int = 1
    DEFAULT_SAMPLE_LENGTH: int = 5

    SAMPLE_FREQUENCY: int = 32000
    NUM_WORKERS: int = 2
    BATCH_SIZE: int = 16

    def __init__(
        self,
        model_class_name: str = None,
        model_config_filepath: str = None,
        model_checkpoint_filepath: str = None,
        model_hparams_filepath: str = None,
        class_list_filepath: str = None,
        working_directory: str = None,
        service_config: ServiceConfig = None,
    ):
        super().__init__(
            model_class_name=model_class_name,
            model_config_filepath=model_config_filepath,
            model_checkpoint_filepath=model_checkpoint_filepath,
            model_hparams_filepath=model_hparams_filepath,
            class_list_filepath=class_list_filepath,
            working_directory=working_directory,
            service_config=service_config,
        )
        self.class_count = len(self.class_list)
        class_tensor = to_onehot(
            torch.arange(0, len(self.class_list)), len(self.class_list)
        )
        self.class_dict = {
            self.class_list.iloc[i, 0]: class_tensor[i].float()
            for i in range(0, len(self.class_list))
        }

    def pre_proccess_data(self, params):
        logger.debug("query parameter file: {}".format(params.get("file")))
        filepath = params.get("file")
        step_width = (
            params.get("step-width")
            if params.get("step-width") is not None
            else self.DEFAULT_STEP_WIDTH
        )
        sample_length = (
            params.get("sample-width")
            if params.get("sample-width") is not None
            else self.DEFAULT_SAMPLE_LENGTH
        )

        if filepath is None:
            raise ValueError("missing file")

        source_path = self.working_directory.joinpath(filepath)
        logger.debug("source exists {}".format(source_path.exists()))
        target_path = self.working_directory.joinpath(uuid4().hex + source_path.suffix)

        length, channels = resample_audio_file(
            source_path, target_path, sample_rate=self.service_config.sample_rate,
        )

        data_list = []
        for i in range(ceil(length / step_width)):
            start_time = i * step_width
            end_time = start_time + sample_length
            if end_time > length:
                end_time = length
            duration = end_time - start_time
            for channel in range(channels):
                data_list.append(
                    (
                        duration,
                        start_time,
                        end_time,
                        "unkown",
                        self.class_list.shape[0],
                        target_path.as_posix(),
                        channel,
                    )
                )

        data_frame = pd.DataFrame(
            data_list,
            columns=[
                "duration",
                "start_time",
                "end_time",
                "labels",
                "species_count",
                "filepath",
                "channels",
            ],
        )
        dataSet = AudioSet(
            self.config,
            data_frame,
            {
                "unkown": torch.tensor([0])
            },  # fake class dict is sufficient beceause we are infrencing not training
            transform_image=None,
            transform_audio=None,
            randomize_audio_segment=False,
            extract_complete_segment=False,  # the prepared dataframe has the correct segment lenght
            sub_segment_overlap=None,
            multi_channel_handling="take_one",
            max_segment_length=sample_length,
        )
        data_loader = DataLoader(
            dataSet,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            drop_last=False,
            pin_memory=True,
        )

        logger.debug("AudioSerice pre_proccess_data")
        return PreProcessedData(
            data_loader=data_loader,
            data_frame=data_frame,
            record_info={"length": length, "channels": channels},
        )

    def post_process(
        self, predictions: Predictions, pre_processed_data: PreProcessedData, params
    ):
        logger.debug("AudioSerice post_process")
        # reduce result to channels
        channel_results = [[]] * pre_processed_data.record_info.get("channels")

        for index, row in pre_processed_data.data_frame.iterrows():
            channel = row["channels"]
            channel_results[channel].append(
                {
                    "startTime": row["start_time"],
                    "endTime": row["end_time"],
                    "predictions": {"logits": predictions.values[index].tolist()},
                }
            )

        response = {
            "apiVersion": 1,
            "fileId": params.get("file"),
            "classIds": self.class_list["latin_name"].tolist(),
            "channels": channel_results,
        }
        return response

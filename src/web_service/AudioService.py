from web_service.BaseService import BaseService
from tools.audio_tools import resample_audio_file
from uuid import uuid4
from pathlib import Path
from logging import error, debug, warn
from torch.utils.data import DataLoader
from dataset import AudioSet


class AudioService(BaseService):
    SAMPLE_FREQUENCY: int = 32000
    STEP_WIDTH: int = 1
    SAMPLE_LENGTH: int = 5
    NUM_WORKERS: int = -1
    BATCH_SIZE: int = 64

    def pre_proccess_data(self, params):
        debug("query parameter file: {}".format(params.get("file")))
        filepath = params.get("file")

        if filepath is None:
            raise ValueError("missing file")

        source_path = self.working_directory.joinpath(filepath)
        debug("source exists {}".format(source_path.exists()))
        target_path = self.working_directory.joinpath(uuid4().hex + source_path.suffix)

        sampleCount = resample_audio_file(
            source_path, target_path, sample_rate=self.SAMPLE_FREQUENCY
        )

        dataSet = AudioSet(
            self.config,
            self.val_dataframe,
            self.class_dict,
            transform_image=None,
            transform_audio=self.val_transform_audio,
            randomize_audio_segment=False,
            extract_complete_segment=True,
            sub_segment_overlap=self.config.validation.sub_segment_overlap,
            multi_channel_handling="take_all",
            max_segment_length=self.SAMPLE_LENGTH,
        )
        self.dataloader = DataLoader(
            dataSet,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.NUM_WORKERS,
            drop_last=False,
            pin_memory=True,
        )
        debug("sampleCount", sampleCount)
        debug("AudioSerice pre_proccess_data")
        return

    def post_process(self, data):
        debug("AudioSerice post_process")
        return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

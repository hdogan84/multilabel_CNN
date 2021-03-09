import model
import yaml
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from pathlib import Path
from config.configuration import parse_config
from logging import debug, info
from torch.utils.data import DataLoader
from torch import tensor
import numpy

logger = logging.getLogger("audio_service")


class PreProcessedData:
    data_loader: DataLoader
    raw_data: pd.DataFrame

    def __init__(
        self,
        data_loader: DataLoader = None,
        data_frame: pd.DataFrame = None,
        record_info: dict = None,
    ):
        self.data_loader = data_loader
        self.data_frame = data_frame
        self.record_info = record_info


class Predictions:
    values: numpy.ndarray
    def __init__(self, values):
        self.values = values


class BaseService:
    model: LightningModule
    model_class_name: str
    model_checkpoint_filepath: Path
    model_hparams_filepath: Path
    class_list: pd.DataFrame
    working_directory: Path
    device: str

    def __init__(
        self,
        model_class_name: str = None,
        model_config_filepath: str = None,
        model_checkpoint_filepath: str = None,
        model_hparams_filepath: str = None,
        class_list_filepath: str = None,
        working_directory: str = None,
    ):
        debug("init Service")
        self.class_list = pd.read_csv(
            class_list_filepath, delimiter=";", quotechar="|",
        )
        self.working_directory = Path(working_directory)
        with open(model_hparams_filepath) as f:
            args = yaml.safe_load(f)
        modelClass = self.___importModel___(model_class_name)
        self.model = modelClass.load_from_checkpoint(
            model_checkpoint_filepath, args=args
        )

        debug("parse config")
        self.config = parse_config(Path(model_config_filepath))

        # prepare model
        self.device = "cuda"
        self.model.cuda()
        self.model.eval()

    def ___importModel___(self, name) -> LightningModule:
        components = name.split(".")
        mod = model
        for comp in components:
            mod = getattr(mod, comp)
        return mod

    def pre_proccess_data(self, params) -> PreProcessedData:
        raise NotImplementedError

    def inference(self, pre_processed_data: PreProcessedData) -> Predictions:
        data_loader = pre_processed_data.data_loader

        outputs = []

        logger.debug("Start inferencing")
        for samples, _, sample_ids in data_loader:
            samples = samples.to(device=self.device)
            output = self.model(samples)
            outputs.append(output.cpu().detach().numpy())

        logger.debug("Return predictions")

        return Predictions(numpy.vstack(outputs))

    def post_process(
        self, predictions: Predictions, pre_processed_data: PreProcessedData, params
    ):
        raise NotImplementedError


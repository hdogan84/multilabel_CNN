from pytorch_lightning import LightningModule
from pathlib import Path
import model
import yaml


class BaseService:
    pl_model: LightningModule
    model_class_name: str
    model_checkpoint_filepath: Path
    model_hparams_filepath: Path
    class_list_filepath: Path
    working_directory: Path

    def __init__(
        self,
        model_class_name: str = None,
        model_checkpoint_filepath: str = None,
        model_hparams_filepath: str = None,
        class_list_filepath: str = None,
        working_directory: str = None,
    ):
        print("init")
        self.class_list_filepath = Path(class_list_filepath)
        self.working_directory = Path(working_directory)
        with open(model_hparams_filepath) as f:
            args = yaml.safe_load(f)
        modelClass = self.___importModel___(model_class_name)
        self.pl_model = modelClass.load_from_checkpoint(
            model_checkpoint_filepath, args=args
        )

    def ___importModel___(self, name) -> LightningModule:
        components = name.split(".")
        mod = model
        for comp in components:
            mod = getattr(mod, comp)
        return mod

    def pre_proccess_data(self, params):
        raise NotImplementedError

    def inference(self, data):
        return self.pl_model(data)

    def post_process(self, data):
        raise NotImplementedError


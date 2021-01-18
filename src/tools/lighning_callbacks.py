from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from pathlib import Path


class SaveConfigToLogs(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_init_start(self, trainer: Trainer):
        pass

    def on_init_end(self, trainer: Trainer):

        pass

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        print("on_sanity_check_end")
        self.config.save_to(Path(trainer.logger.log_dir).joinpath("config.cfg"))
        writer = trainer.logger.experiment
        writer.add_text(
            "config" "First Batch Training Data", self.config.as_html(), 0,
        )
        pass

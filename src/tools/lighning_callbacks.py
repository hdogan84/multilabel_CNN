from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from pathlib import Path


class SaveConfigToLogs(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        self.config.save_to(Path(trainer.logger.log_dir).joinpath("config.cfg"))
        writer = trainer.logger.experiment
        writer.add_text(
            "config" "First Batch Training Data", self.config.as_html(), 0,
        )
        pass


class LogFirstBatchAsImage(Callback):
    mean: float = None
    std: float = None

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        super().__init__()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # first step print out n images
        if batch_idx == 0 and trainer.current_epoch == 0:
            x, classes, _ = batch
            images = x.cpu().detach().numpy()
            # revert normalising and to_tensor
            images = 255 - ((images * self.std) + self.mean) * 255
            writer = trainer.logger.experiment
            writer.add_images(
                "First Batch Training Data", images, 0, dataformats="NCHW"
            )

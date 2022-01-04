from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from pathlib import Path
from tools.config import save_to_yaml, as_html
from shutil import copyfile


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


class SaveFileToLogs(Callback):
    def __init__(self, filepath, file_name):
        super().__init__()
        self.filepath = filepath
        self.file_name = file_name

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        target_file = Path(trainer.logger.log_dir).joinpath(self.file_name)
        Path(trainer.logger.log_dir).mkdir(parents=True, exist_ok=True)
        copyfile(
            self.filepath, target_file,
        )


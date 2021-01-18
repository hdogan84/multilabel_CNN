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
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # first step print out n images
        if batch_idx == 0 and trainer.current_epoch == 0:
            ##images_tensor = torch.cat(x, 0)
            x, classes, _ = batch
            images = x.cpu().detach().numpy()
            writer = trainer.logger.experiment
            writer.add_images(
                "First Batch Training Data", images, 0, dataformats="NCHW"
            )
            # show model
            # writer.add_graph(self.model, x, verbose=False)

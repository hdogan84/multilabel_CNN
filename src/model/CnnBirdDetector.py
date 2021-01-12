import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning import metrics
from pytorch_lightning.metrics.functional import accuracy, average_precision, f1_score
from pytorch_lightning.metrics.classification import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1,
)
import numpy as np


class CnnBirdDetector(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: list,
        learning_rate: float = 2e-4,
        optimizer_type: str = "Adam",
        sgd_momentum: float = 0,
        sgd_weight_decay: float = 0,
        scheduler_type: str = None,
        cosine_annealing_lr_t_max: float = 0,
    ):

        super().__init__()
        self.save_hyperparameters()
        # Set our init args as class attributes

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.sgd_momentum = sgd_momentum
        self.sgd_weight_decay = sgd_weight_decay
        self.scheduler_type = scheduler_type
        self.cosine_annealing_lr_t_max = cosine_annealing_lr_t_max
        self.num_classes = num_target_classes
        # define model
        self.model = models.resnet50(pretrained=True)
        # set input layer to output of mnist
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(2048, self.num_classes)

        # initialise metrics
        self.metrics_train = torch.nn.ModuleDict({"accuracy": Accuracy()},)
        self.metrics_val = torch.nn.ModuleDict(
            {
                "accuracy": Accuracy(),
                # "average_precision": AveragePrecision(num_classes=self.num_classes),
                # # "c_map": AveragePrecision(num_classes=self.num_classes, average="macro"),
                # "confusion_matrix": ConfusionMatrix(num_classes=self.num_classes),
                # "f1": F1(num_classes=self.num_classes, average="micro"),
            }
        )

    def forward(self, x):

        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):

        # self.logger.experiment.image("Training data", batch, 0)

        x, y, _ = batch
        # first step print out n images
        if batch_idx == 0 and self.current_epoch == 0:
            ##images_tensor = torch.cat(x, 0)
            images = x.cpu().detach().numpy()
            writer = self.logger.experiment
            writer.add_images(
                "First Batch Training Data", images, 0, dataformats="NCHW"
            )
            # show model
            writer.add_graph(self.model, x, verbose=False)
        # forward pass on a batch
        pred = self(x)
        train_loss = F.nll_loss(pred, y)
        self.log("train_loss_step", train_loss)

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
        }
        # cacluate and log all metrics
        for name, metric in self.metrics_train.items():
            self.log(
                "Train {}".format(name.title()),
                metric(pred, y),
                on_step=True,
                on_epoch=False,
            )
        return batch_dictionary

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        return

    def validation_step(self, batch, batch_idx):
        x, y, segment_id = batch
        preds = self(x)
        loss = F.nll_loss(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss_step", loss, prog_bar=True)
        for name, metric in self.metrics_val.items():
            self.log(
                "Val {}".format(name.title()),
                metric(preds, y),
                on_step=False,
                on_epoch=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):

        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        elif self.optimizer_type == "Adam":
            print("Optimizer Adam learning rate: {}".format(self.learning_rate))
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = None

        if self.scheduler_type == "CosineAnnealingLR":
            print(
                "Scheduler CosineAnnealingLR  with t_max: {}".format(
                    self.cosine_annealing_lr_t_max
                )
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.cosine_annealing_lr_t_max
            )
        else:
            return {
                "optimizer": optimizer,
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

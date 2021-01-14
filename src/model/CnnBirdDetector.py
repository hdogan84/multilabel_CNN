import torch
from torch import nn
from torch import tensor
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
from tools.tensor_helpers import pool_by_segments
import numpy as np
from sklearn import metrics


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
        validate_complete_segment: bool = False,
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
        return F.log_softmax(x, dim=1)  # return logits

    def training_step(self, batch, batch_idx):
        # self.logger.experiment.image("Training data", batch, 0)
        x, classes, _ = batch
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
        train_loss = F.nll_loss(pred, classes)
        self.log(
            "train_step_loss", train_loss,
        )
        self.log(
            "train_step_accuracy", accuracy(pred, classes),
        )
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
        }
        # cacluate and log all metrics
        for name, metric in self.metrics_train.items():
            self.log(
                "Train {}".format(name.title()),
                metric(pred, classes),
                on_step=True,
                on_epoch=False,
            )
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x, classes, segment_indices = batch
        preds = self(x)
        loss = F.nll_loss(preds, classes)
        batch_dictionary = {
            "loss": loss,
            "preds": preds,
            "classes": classes,
            "segment_indices": segment_indices,
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        preds = torch.cat([x["preds"] for x in outputs])
        classes = torch.cat([x["classes"] for x in outputs])
        segment_indices = torch.cat([x["segment_indices"] for x in outputs])

        preds_on_segment, _ = pool_by_segments(preds, segment_indices)
        classes_on_segment, _ = pool_by_segments(classes, segment_indices)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log(
            "val_accuracy",
            accuracy(preds_on_segment, classes_on_segment),
            prog_bar=True,
        )
        self.log(
            "f1_score", f1_score(preds_on_segment, classes_on_segment), prog_bar=True,
        )
        classes_on_segment = classes_on_segment.cpu()
        class_matrix = torch.zeros(classes_on_segment.shape[0], self.num_classes)
        class_matrix[range(classes_on_segment.shape[0]), classes_on_segment] = 1
        class_matrix = class_matrix.data.numpy()
        preds_on_segment = preds_on_segment.cpu().data.numpy()
        lrap = metrics.label_ranking_average_precision_score(
            class_matrix, preds_on_segment
        )
        cMap = metrics.average_precision_score(
            class_matrix, preds_on_segment, average="macro"
        )  # 'micro' 'macro' 'samples'
        self.log(
            "lrap", lrap, prog_bar=True,
        )
        self.log(
            "cMap", cMap, prog_bar=True,
        )
        return

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

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, average_precision, f1_score
import torchvision.models as models
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

        # Hardcode some dataset specific attributes
        self.num_classes = num_target_classes

        # Define PyTorch model

        # self.feature_extractor = models.resnet50(pretrained=True)
        # self.feature_extractor.eval()
        # # use the pretrained model to classify cifar-10 (10 image classes)
        # self.classifier = nn.Linear(2048, num_target_classes)
        self.model = models.resnet50(pretrained=True)
        # set input layer to output of mnist
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(2048, self.num_classes)

        # print(self.model)

    def forward(self, x):

        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):

        # self.logger.experiment.image("Training data", batch, 0)

        x, y = batch
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
        pred = self.forward(x)
        # identifying number of correct predections in a given batch
        correct = pred.argmax(dim=1).eq(y).sum().item()
        # identifying total number of labels in a given batch
        total = len(y)

        logits = self(x)
        train_loss = F.nll_loss(logits, y)
        # train_loss = F.cross_entropy(logits, y)

        self.log("train_loss", train_loss)

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": train_loss,
            # info to be used at epoch end
            "correct": correct,
            "total": total,
        }
        return batch_dictionary

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])
        # creating log dictionary
        self.log("loss", avg_loss)
        self.log("Accuracy", correct / total)
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        average_precision_value = average_precision(preds, y)  # multilabel will be lrap
        f1_score_value = f1_score(preds, y)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("average_precision", average_precision_value, prog_bar=True)
        self.log("f1_score", f1_score_value, prog_bar=True)
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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = None

        if self.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.cosine_annealing_lr_t_max
            )
        else:
            scheduler = None
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

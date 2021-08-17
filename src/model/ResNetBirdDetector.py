import torch
from torch import nn
import torchvision.models as models
from torchmetrics.functional import accuracy, average_precision
from model.BaseBirdDetector import BaseBirdDetector

class ResNetBirdDetector(BaseBirdDetector):
    def define_model(self):

        self.model = models.resnet50(pretrained=True)
        # set input layer to output of mnist
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model.fc = nn.Linear(2048, self.num_classes)
        self.sigm = nn.Sigmoid()
        self.Criterion = nn.BCELoss()
        self.isLogitOutput = False

    def forward(self, x):
        x = self.sigm(self.model(x))
        return x

    def training_step(self, batch, batch_idx):
        # self.logger.experiment.image("Training data", batch, 0)
        x, classes, _ = batch
        target = classes.type(torch.int)

        # forward pass on a batch
        preds = self(x)

        loss = self.Criterion(preds, classes)

        self.log(
            "train_step_accuracy",
            accuracy(preds, target, num_classes=self.num_classes),
            prog_bar=True,
        )
        self.log(
            "train_step_average_precision",
            average_precision(preds, target),
            prog_bar=True,
        )

        # logging
        self.log(
            "train_step_loss", loss,
        )
        # self.log(
        #     "train_step_accuracy", accuracy(preds, classes),
        # )
        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": loss,
        }

        return batch_dictionary

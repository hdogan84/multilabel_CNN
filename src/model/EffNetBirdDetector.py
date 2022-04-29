import torch
from torch import nn
from torch import tensor
from torch import sigmoid
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
import pytorch_lightning as pl
import torchvision.models as models
from sklearn.metrics import label_ranking_average_precision_score
from model.BaseBirdDetector import BaseBirdDetector

from torchmetrics.functional import accuracy, average_precision, f1
from torchmetrics import Accuracy, AveragePrecision, F1, ConfusionMatrix

import numpy as np

from sklearn import metrics
from model.BaseBirdDetector import BaseBirdDetector

# https://github.com/rwightman/pytorch-image-models/
# https://github.com/rwightman/pytorch-image-models/blob/master/train.py
import timm
#from transformers import get_cosine_schedule_with_warmup

class EffNetBirdDetector(BaseBirdDetector):
    def define_model(self):
        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=self.num_classes, in_chans=1)
        
        # self.Criterion = F.cross_entropy
        self.Criterion = F.binary_cross_entropy_with_logits
        self.isLogitOutput = True

    def forward(self, x):
        x = self.model(x)
        return x
        # F.softmax(x, dim=1)  # return logits

    def training_step(self, batch, batch_idx):
        # self.logger.experiment.image("Training data", batch, 0)
        x, classes, _ = batch

        # forward pass on a batch
        preds = self(x)

        train_loss = self.Criterion(preds, classes)
        self.log(
            "train_average_precision",
            average_precision(preds, classes, pos_label=1),
            prog_bar=True,
        )
        # logging
        self.log(
            "train_step_loss", train_loss,
        )
        # self.log(
        #     "train_step_accuracy", accuracy(preds, classes),
        # )
        self.log(
            "lr", self.optimizer.param_groups[0]['lr'],
        )

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": train_loss,
        }

        return batch_dictionary


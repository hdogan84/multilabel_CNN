from pytorch_lightning.metrics.functional.classification import get_num_classes
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
from pytorch_lightning.metrics.utils import to_onehot


from torchmetrics.functional import accuracy, average_precision, f1
from torchmetrics import Accuracy, AveragePrecision, F1, ConfusionMatrix

import numpy as np

from sklearn import metrics

# https://github.com/rwightman/pytorch-image-models/
# https://github.com/rwightman/pytorch-image-models/blob/master/train.py
import timm
#from transformers import get_cosine_schedule_with_warmup

class EffNetBirdDetector(pl.LightningModule):
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
        # self.model = models.resnet50(pretrained=True)
        # # set input layer to output of mnist
        # self.model.conv1 = torch.nn.Conv2d(
        #     1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )
        # self.model.fc = nn.Linear(2048, self.num_classes)
        # self.bce = nn.BCELoss()

        self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=self.num_classes, in_chans=1)
        
        # self.Criterion = F.cross_entropy
        self.Criterion = F.binary_cross_entropy_with_logits

        self.Accuracy = Accuracy(dist_sync_on_step=True,num_classes=self.num_classes)
        self.F1 = F1(dist_sync_on_step=True)
        self.AveragePrecision = AveragePrecision(dist_sync_on_step=True)
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

    def validation_step(self, batch, batch_idx):
        x, classes, segment_indices = batch
        target = classes.type(torch.int)
        preds = self(x)
        loss = self.Criterion(preds, classes)
        batch_dictionary = {
            "loss": loss,
            "preds": preds,
            "classes": classes,
            "segment_indices": segment_indices,
        }
        probs = sigmoid(preds)
        self.Accuracy(probs, target)
        self.AveragePrecision(preds, target)
        self.F1(probs, target)
        return batch_dictionary


    def validation_epoch_end(self, outputs):

        self.log("val_accuracy", self.Accuracy.compute(), prog_bar=True, sync_dist=True)
        self.log(
            "val_average_precision",
            self.AveragePrecision.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        value = self.F1.compute()
        self.log("val_f1", value, prog_bar=True, sync_dist=True)
        
        
        return


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):

        if self.optimizer_type == "SGD":
            print(
                "Optimizer SGD learning rate: {} momentum: {} weight_decay: {}".format(
                    self.learning_rate, self.momentum, self.weight_decay
                )
            )
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

        self.optimizer = optimizer
        
        if self.scheduler_type == "CosineAnnealingLR":
            print(
                "Scheduler CosineAnnealingLR  with t_max: {}".format(
                    self.cosine_annealing_lr_t_max
                )
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.cosine_annealing_lr_t_max
            )
        # else:
        #     return {
        #         "optimizer": optimizer,
        #     }

        # NNN 1 cycle with warmup
        #num_train_steps = int(len(train_loader) * args.epochs)
        #num_warmup_steps = int(0.1 * args.epochs * len(train_loader))
        #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        
        # if self.scheduler_type == "cosine_schedule_with_warmup":
        #     # print(
        #     #     "Scheduler CosineAnnealingLR  with t_max: {}".format(
        #     #         self.cosine_annealing_lr_t_max
        #     #     )
        #     # )
        #     num_train_steps = int(1000 * 40)
        #     num_warmup_steps = int(1000 * 2)
        #     scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        # }

        if self.scheduler_type:
            return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            }
        else:
            return {
                "optimizer": optimizer,
            }

import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, AveragePrecision, F1, AUC
from sklearn.metrics import label_ranking_average_precision_score
from tools.tensor_helpers import pool_by_segments

class BaseBirdDetector(pl.LightningModule):
    def __init__(
        self,
        num_target_classes: int,
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
        self.isLogitOutput = True
        # init class metrics
        self.Accuracy = Accuracy(dist_sync_on_step=True, num_classes=self.num_classes)
        self.F1 = F1(dist_sync_on_step=True, num_classes=self.num_classes)
        self.AveragePrecision = AveragePrecision(dist_sync_on_step=True)
        self.AUC = AUC(compute_on_step=True, dist_sync_on_step=True)
        # validation type
        self.channel_wise_validation = True
        self.define_model()

    def define_model(self):
        pass

    def validation_step(self, batch, batch_idx):

        x, classes, segment_indices = batch
        


        preds = self(x)
        # pool segments 
        classes, _ = pool_by_segments(classes,segment_indices)
        preds , _ = pool_by_segments(preds,segment_indices,pooling_method="max")
        target = classes.type(torch.int)

        
        preds_prob = 0
        preds_logit = 0
        
        
        if self.isLogitOutput:
            preds_prob = torch.sigmoid(preds)
            preds_logit = preds
        else:
            preds_prob = preds
            preds_logit = torch.logit(preds)


        loss = self.Criterion(preds, classes)

        self.Accuracy(preds_prob, target)
        self.AveragePrecision(preds_prob, target)
        self.F1(preds_prob, target)
        # self.AUC(preds_prob, target)
        self.log("val_step_loss", loss, prog_bar=True, sync_dist=True)
        batch_dictionary = {
            "loss": loss,
            "preds": preds,
            "preds_logit": preds_logit,
            "preds_prob": preds_prob,
            "classes": classes,
            "segment_indices": segment_indices,
        }
        return batch_dictionary

    def cal_metrics(self, outputs):
        preds_all = torch.cat([x["preds"] for x in outputs])
        classes_all = torch.cat([x["classes"] for x in outputs])

        return {
            "accuracy": self.Accuracy.compute(),
            "average_precision": self.AveragePrecision.compute(),
            "f1": self.F1.compute(),
            # "auc": self.AUC.compute(),
            "lrap": label_ranking_average_precision_score(
                classes_all.cpu().data.numpy(), preds_all.cpu().data.numpy(),
            ),
        }

    def validation_epoch_end(self, outputs):
        metrics = self.cal_metrics(outputs)
        accuracy = metrics["accuracy"]
        average_precision = metrics["average_precision"]
        f1 = metrics["f1"]
        lrap = metrics["lrap"]
        # auc = metrics["auc"]

        # Log metrics to terminal
        self.log("val_accuracy", accuracy, prog_bar=True, sync_dist=True)
        self.log(
            "val_average_precision", average_precision, prog_bar=True, sync_dist=True,
        )
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

        #self.log("val_auc", auc, prog_bar=True, sync_dist=True)

        # Log metrics against epochs in tensorboard ()
        # self.logger.experiment.add_scalar("epoch_val_loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "epoch_val_accuracy", accuracy, self.current_epoch
        )
        self.logger.experiment.add_scalar(
            "epoch_val_average_precision", average_precision, self.current_epoch
        )
        self.logger.experiment.add_scalar("epoch_val_f1", f1, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_val_lrap", lrap, self.current_epoch)

        self.logger.experiment.add_scalar(
            "epoch_lr", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )

        return

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
  
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        metrics = self.cal_metrics(outputs)
        self.log("accuracy", metrics["accuracy"])
        self.log("average_precision", metrics["average_precision"])
        self.log("f1", metrics["f1"])
        self.log("f1", metrics["f1"])
        self.log("lrap", metrics["lrap"])

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
        else:
            return {
                "optimizer": optimizer,
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

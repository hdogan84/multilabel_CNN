import torch
from torch import nn
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
import pytorch_lightning as pl
import torchvision.models as models
from sklearn.metrics import label_ranking_average_precision_score
from torchmetrics.functional import accuracy, average_precision,f1
from torchmetrics.classification import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1,
)
from tools.tensor_helpers import pool_by_segments
import numpy as np

from sklearn import metrics


# sed
import timm
from timm.models.efficientnet import tf_efficientnet_b0_ns
from timm.models.efficientnet import tf_efficientnet_b2_ns
from timm.models.efficientnet import tf_efficientnet_b3_ns
from functools import partial
from torchlibrosa.augmentation import SpecAugmentation
#import torch.nn.functional as F

#encoder = 'tf_efficientnet_b2_ns'
encoder = 'tf_efficientnet_b0_ns'
encoder_params = {}
encoder_params["tf_efficientnet_b0_ns"] = {"features": 1280, "init_op": partial(tf_efficientnet_b0_ns, pretrained=True, drop_path_rate=0.2)}
encoder_params["tf_efficientnet_b2_ns"] = {"features": 1408, "init_op": partial(tf_efficientnet_b2_ns, pretrained=True, drop_path_rate=0.2)}
encoder_params["tf_efficientnet_b3_ns"] = {"features": 1536, "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2)}


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        
class AudioSEDModel(nn.Module):
    #def __init__(self, encoder, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
    def __init__(self, encoder, classes_num):
        super().__init__()

        self.interpolate_ratio = 30  # Downsampled ratio


        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=6, time_stripes_num=2, freq_drop_width=6, freq_stripes_num=2)
        
        # Model Encoder
        self.encoder = encoder_params[encoder]["init_op"]()
        self.fc1 = nn.Linear(encoder_params[encoder]["features"], 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation="linear")
        #self.bn0 = nn.BatchNorm2d(mel_bins)
        #self.bn0 = nn.BatchNorm2d(ImageSize[0]) # ???
        self.bn0 = nn.BatchNorm2d(128) # ToDo
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.fc1)
        init_bn(self.bn0)
    
    def forward(self, input, mixup_lambda=None):
        """Input : (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)
        # # batch_size x 1 x time_steps x freq_bins --> torch.Size([1, 1, 938, 1025])
        # #print('spectrogram_extractor', x.shape)

        # x = self.logmel_extractor(x)
        # # batch_size x 1 x time_steps x mel_bins ---> torch.Size([1, 1, 938, 128])
        # #print('logmel_extractor', x.shape, x.type(), torch.min(x[0,0,:]), torch.mean(x[0,0,:]), torch.max(x[0,0,:]))
        # # logmel_extractor torch.Size([28, 1, 938, 256]) torch.cuda.FloatTensor tensor(-100., device='cuda:0') tensor(-35.4645, device='cuda:0') tensor(-4.3164, device='cuda:0')

        x = input
        #print('input', x.shape) # torch.Size([48, 3, 224, 557])

        # Correct dimension order (after apply ToTensor on image)
        # bs,c,bins,frames --> bs,c,frames,bins
        x = x.transpose(2, 3)
        #print('Correct dimension order', x.shape)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        #print('transpose(1, 3)', x.shape)  # transpose(1, 3) torch.Size([28, 256, 938, 1])
        x = self.bn0(x)
        #print('bn0', x.shape)  # torch.Size([28, 256, 938, 1])
        x = x.transpose(1, 3)
        #print('transpose(1, 3)', x.shape)  # torch.Size([28, 1, 938, 256])

        # Not used at the end for small lr
        #if self.training: x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        # Output shape (batch size, channels, time, frequency)
        # Not needed if already 3 channel
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3]) # torch.Size([28, 1, 938, 256]) --> torch.Size([28, 3, 938, 256])
        #print('expand', x.shape) # torch.Size([48, 3, 224, 557])
        x = self.encoder.forward_features(x)
        #print('self.encoder.forward_features', x.shape)     # torch.Size([28, 1536, 30, 8])
        x = torch.mean(x, dim=3) # try max?
        #print('torch.mean(x, dim=3)', x.shape)              # torch.Size([28, 1536, 30])

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2 # ToCheck if concat or using only max or avg is better
        #print('x = x1 + x2', x.shape)  # torch.Size([28, 1536, 30])

        x = F.dropout(x, p=0.4, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.4, training=self.training)
        #print('F.dropou', x.shape)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Check if logit == clipwise_output --> True if activation="linear"
        #print(logit.shape, clipwise_output.shape)
        #print(torch.all(logit.eq(clipwise_output)))

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            'framewise_output' : framewise_output,
            'logit' : logit,
            'clipwise_output' : clipwise_output
        }

        return output_dict

class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        #self.bce = nn.BCELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        #input_ = input["clipwise_output"]
        input_ = input["logit"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        #input_ = torch.where(input_>1, torch.ones_like(input_), input_)

        target = target.float()

        return self.bce(input_, target)


class SedBirdDetector(pl.LightningModule):
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

        #self.bce = nn.BCELoss()
        # self.Criterion = F.cross_entropy
        #self.Criterion = F.binary_cross_entropy_with_logits

        # sed
        self.model = AudioSEDModel(encoder, self.num_classes)
        self.criterion = PANNsLoss()

    def forward(self, x):

        #print('input.shape', x.shape) # input.shape torch.Size([48, 1, 128, 445]) bs,c,bins,frames
        x = self.model(x)
        return x
        # F.softmax(x, dim=1)  # return logits

    def training_step(self, batch, batch_idx):
        # self.logger.experiment.image("Training data", batch, 0)
        x, classes, _ = batch

        # forward pass on a batch
        output_dict = self(x)
        train_loss = self.criterion(output_dict, classes)
        preds = torch.sigmoid(torch.max(output_dict['framewise_output'], dim=1)[0])
        
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
        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": train_loss,
        }

        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x, classes, segment_indices = batch
        
        # sed
        output_dict = self(x)
        loss = self.criterion(output_dict, classes)
        preds = torch.sigmoid(torch.max(output_dict['framewise_output'], dim=1)[0])

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

        if classes.dim() == 1:
            classes_on_segment = to_onehot(classes_on_segment, self.num_classes)

        self.log("val_loss", avg_loss, prog_bar=True)
        
        val_average_precision = average_precision(preds_on_segment, classes_on_segment, pos_label=1)
        self.log("val_average_precision", val_average_precision, prog_bar=True)
        
        val_f1 = f1(preds_on_segment, classes_on_segment, self.num_classes)
        self.log("val_f1", val_f1, prog_bar=True)
        
        val_lrap = label_ranking_average_precision_score(
                classes_on_segment.cpu().data.numpy(),
                preds_on_segment.cpu().data.numpy(),
            )

        self.log("val_lrap", val_lrap, prog_bar=True)


        # Log metrics against epochs () 
        self.logger.experiment.add_scalar("epoch_val_loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_val_average_precision", val_average_precision, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_val_f1", val_f1, self.current_epoch)
        self.logger.experiment.add_scalar("epoch_val_lrap", val_lrap, self.current_epoch)
        
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

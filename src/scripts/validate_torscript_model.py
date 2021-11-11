import pandas as pd
from torch import nn
from tools.config import load_yaml_config
import argparse
from torchmetrics import Accuracy, AveragePrecision, F1, AUC
from tools.tensor_helpers import pool_by_segments
from pathlib import Path

# from data_module.ColorSpecAmmodMultiLabelModule import ColorSpecAmmodMultiLabelModule as DataModule
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule as DataModule
from tools.config import load_class_list_from_index_to_name_json
from tools.tensor_helpers import (
    transform_class_tensor,
    get_class_tensor_transformation_matrix,
)
from augmentation.signal import ExtendedCompose as SignalCompose, create_signal_pipeline

import torch
import albumentations as A

device = "cuda"
model_path = (
    "data/torchserve-models/raw/birdId-europe-254-2103/birdId-europe-254-2103.pt"
)
index_to_name_json_path = (
    "data/torchserve-models/raw/birdId-europe-254-2103/index_to_name.json"  #
)
config_path = "./config/birdId-europ-254.yaml"
# trainer = Trainer(tpu_cores=8)
def validate(config_filepath):
    config = load_yaml_config(config_filepath)

    NumOfLowFreqsInPixelToCutMax = 4
    NumOfHighFreqsInPixelToCutMax = 6
    imageHeight = 224
    resizeFactor = imageHeight / (
        config.audio_loading.num_of_mel_bands
        - NumOfLowFreqsInPixelToCutMax / 2.0
        - NumOfHighFreqsInPixelToCutMax / 2.0
    )
    imageWidth = int(
        resizeFactor
        * config.data.segment_duration
        * config.audio_loading.sample_rate
        / config.audio_loading.fft_hop_size_in_samples
    )

    val_transform_signal = SignalCompose(
        create_signal_pipeline(config.validation.signal_pipeline, config),
        shuffle=config.augmentation.shuffle_signal_augmentation,
    )
    val_transform_image = A.Compose(
        [
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.5),
            A.Resize(imageHeight, imageWidth, A.cv2.INTER_LANCZOS4, always_apply=True)
        ]
    )
    val_transform_audio = None
    data_module = DataModule(
        config, None, None, val_transform_signal, val_transform_image,
    )
    data_module.setup("test")
    num_classes = data_module.class_count
    data_loader = data_module.test_dataloader()
    model_class_list = load_class_list_from_index_to_name_json(index_to_name_json_path)
    data_class_list = [key for key in data_module.class_dict]
    class_tensor_transformation_matrix = get_class_tensor_transformation_matrix(
        model_class_list, data_class_list
    ).to(device)

    accuracy = Accuracy(num_classes=num_classes).cpu()
    f1 = F1(num_classes=num_classes).cpu()
    averagePrecision = AveragePrecision().to("cpu")

    # aUC = AUC(compute_on_step=True, dist_sync_on_step=True).cpu()
    criterion = nn.BCELoss()
    with torch.no_grad():
        model = torch.jit.load(model_path, map_location=device)
        batch_results = []
        batch_amount = len(data_loader)
        for index, batch in enumerate(data_loader):

            x, classes, segment_indices = batch

            x = x.to(device)
            segment_indices = segment_indices
            preds = model(x)
            preds = transform_class_tensor(preds, class_tensor_transformation_matrix)
            predictions = preds.cpu()
            # pool segments
            classes, _ = pool_by_segments(classes, segment_indices)
            predictions, _ = pool_by_segments(
                predictions, segment_indices, pooling_method="max"
            )
            target = classes.type(torch.int).cpu()
            loss = criterion(predictions, classes)
            accuracy(predictions, target)

            averagePrecision(predictions, target)
            f1(predictions, target)
            # self.AUC(predictions_prob, target)
            batch_dictionary = {
                "loss": loss,
                "predictions": predictions,
                "classes": classes,
                "segment_indices": segment_indices,
            }
            batch_results.append(batch_dictionary)
            print("Batch {}/{} ".format(index, batch_amount))

        metrics_dict = {
            "accuracy": accuracy.compute().item(),
            "average_precision": averagePrecision.compute().item(),
            "f1": f1.compute().item(),
            # "auc": self.AUC.compute(),
            # "lrap": label_ranking_average_precision_score(
            #     classes_all.cpu().data.numpy(), preds_all.cpu().data.numpy(),
            # ),
        }
        print(metrics_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        metavar="path",
        type=Path,
        nargs="?",
        default=config_path,
        help="config file for all settings",
    )

    args = parser.parse_args()
    config_filepath = args.config

    validate(config_filepath)


import pytorch_lightning as pl
from tools.config import load_yaml_config
import argparse
from pathlib import Path
from data_module.ColorSpecAmmodMultiLabelModule import ColorSpecAmmodMultiLabelModule
from data_module.AmmodMultiLabelModule import AmmodMultiLabelModule
import torch
import csv


# trainer = Trainer(tpu_cores=8)
def start_extract(config_filepath):
    config = load_yaml_config(config_filepath)

    data_module = ColorSpecAmmodMultiLabelModule(
        config, fit_transform_signal=None, fit_transform_image=None,
    )
    data_module.setup(stage="test")
    data_set = data_module.test_set
    csv_list = []
    for i in data_set:
        data, y, index_tensor = i
        index = index_tensor[0]
        filename = "{file_id}-{start_time}".format(
            file_id=data_set.segments[index]["annotation_interval"]["file_id"],
            start_time=data_set.segments[index]["start_time"],
        )
        filepath = Path("./out/{}".format(filename))
        print(filepath)
        if not filepath.exists():
            torch.save(i, filepath)

        entry = [
            filename,
            data_set.segments[index]["annotation_interval"]["file_id"],
            data_set.segments[index]["start_time"],
            data_set.segments[index]["end_time"],
        ]
        csv_list.append(entry)
        print(entry)

    fields = ["filename", "file_id", "start_time", "end_time"]

    # data rows of csv file

    with open("data_set.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(csv_list)
    #     data, y, index = i
    #     print(data_set.segments[index[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        metavar="path",
        type=Path,
        nargs="?",
        default="./config/resnet_multi_label.yaml",
        help="config file for all settings",
    )

    args = parser.parse_args()
    config_filepath = args.config

    start_extract(config_filepath)


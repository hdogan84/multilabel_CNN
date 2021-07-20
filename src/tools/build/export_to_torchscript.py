from inquirer import Text, Path, List, prompt
from tools.build.get_checkpoint_path import main as get_checkpoint_path
from tools.config import load_yaml_config
import model as ModelModule
from tools.build.tools import get_list_of_module
import torch
import pandas as pd
import json

print("Please set Service run parameters:")
# num_workers = 2
# batch_size = 16
# service_class_name = AudioService
# model_class_name = CnnBirdDetector
# sample_rate = 32000

build_path = "./build"
logs_path = "./logs"
configs_path = "./config"
docker_files_path = "./docker_files"


def main():

    class_name = prompt(
        [
            List(
                name="value",
                message="Select model class?",
                choices=get_list_of_module(ModelModule),
            )
        ]
    )["value"]

    model_path = get_checkpoint_path(logs_path)
    # load hp params

    config = load_yaml_config(model_path["config"])
    ModelClass = getattr(ModelModule, class_name)
    model = ModelClass.load_from_checkpoint(model_path["checkpoint"], **config)
    model.cpu()
    script = model.to_torchscript()
    torch.jit.save(script, build_path + "/model.pt")

    data_frame = pd.read_csv(model_path["class_list"], delimiter=";")
    result = {}
    for index, value in data_frame["latin_name"].iteritems():
        result["{}".format(index)] = value

    with open(build_path + "/index_to_name.json", "w") as outfile:
        json.dump(result, outfile)
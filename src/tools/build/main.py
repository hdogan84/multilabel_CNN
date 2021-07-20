import os

import pathlib
import shutil
from inquirer import Text, Path, List, prompt
from tools.build.get_model_path import main as get_model_path
from tools.build.merge_to_handler_config import main as merge_to_handler_config
from tools.build.export_to_torchscript import main as export_to_torchscript
from tools.config import save_to_yaml
from tools.build.tools import file_exists, is_int, get_list_of_files

print("Please set Service run parameters:")
# num_workers = 2
# batch_size = 16
# service_class_name = AudioService
# model_class_name = CnnBirdDetector
# sample_rate = 32000

build_path = "./build"
handler_path = "./src/torchserve_handler"
logs_path = "./logs"


def main():
    cmd_string = "torch-model-archiver --serialized-file ./build/model.pt --export-path ./build --handler ./build/handler.py --force"
    if pathlib.Path(build_path).exists():
        # clear build directory
        shutil.rmtree(build_path)
    os.mkdir(build_path)

    model_desc = prompt(
        [
            Text(
                name="model_name",
                message="Enter Model Name!",
            ),
            Text(name="version", message="Enter Model Version", validate=is_int),
        ]
    )
    use_checkpoint = prompt(
        [
            List(
                name="value",
                message="Do you want to use a pytorch lightning checkpoint and model ?",
                choices=["yes", "no"],
            )
        ]
    )
    if use_checkpoint["value"] == "yes":
        export_to_torchscript()
        # if checkpoint is used there is also an index_to_name.json file

        cmd_string += " --extra-files ./build/index_to_name.json,./build/config.yaml"
    else:
        pytorch_script_filepath = prompt(
            [
                Path(
                    name="value",
                    message="Enter path to your pytorch script!",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )["value"]
        shutil.copy(pytorch_script_filepath, build_path + "/model.pt")
        cmd_string += " --extra-files ./build/config.yaml"
    use_config = prompt(
        [
            List(
                name="value",
                message="Do you want to use a pytorch lightning config?",
                choices=["yes", "no"],
            )
        ]
    )

    if use_config["value"] == "yes":

        service_config = merge_to_handler_config()
        save_to_yaml(service_config, build_path + "/config.yaml")

    else:
        pytorch_script_filepath = prompt(
            [
                Path(
                    name="value",
                    message="Enter path to torchserve handler config!",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )["value"]
        shutil.copy(pytorch_script_filepath, build_path + "/config.yaml")

    handler = prompt(
        [
            List(
                name="value",
                message="Choose handler you want to use!",
                choices=get_list_of_files(handler_path),
            )
        ]
    )["value"]
    shutil.copy(handler, build_path + "/handler.py")
    use_requirements = prompt(
        [
            List(
                name="value",
                message="Do you want to add a requirements file?",
                choices=["yes", "no"],
            )
        ]
    )

    if use_requirements["value"] == "yes":
        requirement_filepath = prompt(
            [
                Path(
                    name="value",
                    message="Enter path to your requirement file!",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )["value"]
        shutil.copy(requirement_filepath, build_path + "/requirements.txt")
        cmd_string += " --requirements-file ./build/requirements.txt"

    cmd_string += " --model-name {model_name} --version {version}".format(
        model_name=model_desc["model_name"],
        version=model_desc["version"],
    )

    print(cmd_string)
    os.system(cmd_string)

from inquirer import Text, Path, List, prompt
from tools.build.tools import (
    get_list_of_sub_directories,
    get_list_of_files,
    file_exists,
)
import pathlib


def main(logs):
    model_path = None
    class_list_path = None

    use_checkpoint = prompt(
        [
            List(
                name="value",
                message="Do you want to use a checkpoint from logs?",
                choices=["yes", "no"],
            )
        ]
    )

    if use_checkpoint["value"] is "yes":
        experiment_path = prompt(
            [
                List(
                    name="value",
                    message="Choose experiment!",
                    choices=get_list_of_sub_directories(logs),
                )
            ]
        )
        run_path = prompt(
            [
                List(
                    name="value",
                    message="Choose version!",
                    choices=get_list_of_sub_directories(experiment_path["value"]),
                )
            ]
        )
        model_path = prompt(
            [
                List(
                    name="value",
                    message="Choose checkpoint!",
                    choices=get_list_of_files(run_path["value"] + "/checkpoints"),
                )
            ]
        )
        if pathlib.Path(run_path["value"] + "/class_list.csv").exists():
            class_list_path = {"value": run_path["value"] + "/class_list.csv"}
        else:
            class_list_path = prompt(
                [
                    Path(
                        name="value",
                        message="Enter class list csv path !",
                        path_type=Path.FILE,
                        validate=file_exists,
                    ),
                ]
            )

    else:
        model_path = prompt(
            [
                Path(
                    name="value",
                    message="Enter path to model checkpoint!",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )
        class_list_path = prompt(
            [
                Path(
                    name="value",
                    message="Enter class list csv path !",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )
    return {
        "model_path": model_path["value"],
        "class_list_path": class_list_path["value"],
    }

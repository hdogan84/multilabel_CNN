from inquirer import Path, List, prompt
from tools.build.tools import (
    get_list_of_sub_directories,
    get_list_of_files,
    file_exists,
)


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

    if use_checkpoint["value"] == "yes":
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

    return {
        "checkpoint": model_path["value"],
        "config": run_path["value"] + "/hparams.yaml",
        "class_list": run_path["value"] + "/class-list.csv",
    }

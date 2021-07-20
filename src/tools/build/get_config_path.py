from inquirer import Text, Path, List, prompt
from tools.build.tools import get_list_of_files


def main(config_path):
    use_config = prompt(
        [
            List(
                name="value",
                message="Do you want to use configuration file from config folder?",
                choices=["yes", "no"],
            )
        ]
    )

    if use_config["value"] is "yes":
        config_path = prompt(
            [
                List(
                    name="value",
                    message="Choose configuration file!",
                    choices=get_list_of_files(config_path),
                )
            ]
        )

    else:
        config_path = prompt(
            [
                Path(
                    name="value",
                    message="Enter Path to your configuration",
                    path_type=Path.FILE,
                    validate=file_exists,
                ),
            ]
        )
    return config_path["value"]

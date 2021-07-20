import os

import pathlib
import shutil
from inquirer import Text, Path, List, prompt, Checkbox
from tools.build.get_config_path import main as get_config_path
from tools.build.get_settings import main as get_settings
from tools.config import load_yaml_config
from tools.build.tools import (
    file_exists,
)


configs_path = "./config"


def main():
    config_path = get_config_path(configs_path)
    pt_config = load_yaml_config(config_path)
    config = {}
    config_modules = prompt(
        [
            Checkbox(
                "value",
                message="Which sub configs do you want to merge into the handler config?",
                choices=pt_config.keys(),
            )
        ]
    )
    keys = config_modules["value"]
    for key in keys:
        config.update(pt_config[key])
    settings = get_settings()
    config.update(settings)
    return config

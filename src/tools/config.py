from pathlib import Path
import yaml
from munch import munchify
import json


def load_yaml_config(config_filepath: Path):
    with open(config_filepath) as f:
        config = yaml.safe_load(f)
        # munchify allows attribute style access
        return munchify(config)


def save_to_yaml(config: dict, target_filepath: Path) -> None:
    with open(target_filepath, "w+") as text_file:
        yaml.safe_dump(config, text_file, allow_unicode=True)


def save_to_json(config: dict, target_filepath: Path) -> None:
    with open(target_filepath, "w+") as text_file:
        json.dump(config, text_file)


def as_html(config) -> str:
    lines = []
    for section in config.items():
        lines.append("<br>[{}]".format(section[0]))
        for values in section[1].items():
            lines.append("{} : {}".format(values[0], values[1]))
    return "<br>".join(lines)

def load_class_list_from_index_to_name_json(filepath):
    with open(filepath) as f:
        class_dict = json.load(f)
        class_list = list(range(0,len(class_dict.items())))
        for key in class_dict:
            class_list[int(key)] = class_dict[key]
        return class_list
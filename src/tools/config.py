from pathlib import Path
import yaml
from munch import munchify


def load_yaml_config(config_filepath: Path):
    with open(config_filepath) as f:
        config = yaml.safe_load(f)
        # munchify allows attribute style access
        return munchify(config)


def save_to_yaml(config: dict, target_filepath: Path) -> None:
    with open(target_filepath, "w+") as text_file:
        yaml.safe_dump(config, text_file, allow_unicode=True)


def as_html(config) -> str:
    lines = []
    for section in config.items():
        lines.append("<br>[{}]".format(section[0]))
        for values in section[1].items():
            lines.append("{} : {}".format(values[0], values[1]))
    return "<br>".join(lines)

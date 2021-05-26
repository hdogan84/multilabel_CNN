from typing import NamedTuple
from pathlib import Path


class ModelConfig(NamedTuple):
    def as_dict(self) -> dict:
        result = {}
        excludes = list(dir(ModelConfig))
        excludes.append("__dict__")
        for key in dir(self.__class__):
            if key in excludes:
                pass
            else:
                value = self.__getattribute__(key)
                if isinstance(value, ModelConfig):
                    result[key] = value.as_dict()
                else:
                    result[key] = value

        return result

    def as_ini_string(self) -> str:
        lines = []
        for section in self.as_dict().items():
            lines.append("[{}]".format(section[0]))
            for values in section[1].items():
                lines.append("{} = {}".format(values[0], values[1]))
        return "\n".join(lines)

    def as_html(self) -> str:
        lines = []
        for section in self.as_dict().items():
            lines.append("<br>[{}]".format(section[0]))
            for values in section[1].items():
                lines.append("{} = {}".format(values[0], values[1]))
        return "<br>".join(lines)

    def save_to(self, filepath: Path) -> None:
        with open(filepath, "w+") as text_file:
            print(self.as_ini_string(), file=text_file)


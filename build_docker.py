import inquirer
from os import listdir
from os.path import isfile, join
from pathlib import Path


def get_service_list():
    service_folder = "./src/web_service/"

    files = [
        (Path(f).stem, Path(join(service_folder, f)))
        for f in listdir(service_folder)
        if isfile(join(service_folder, f))
        and f != "__init__.py"
        and f != "BaseService.py"
    ]
    return files


def get_models_list():
    service_folder = "./src/model/"

    files = [
        (Path(f).stem, Path(join(service_folder, f)))
        for f in listdir(service_folder)
        if isfile(join(service_folder, f)) and f != "__init__.py"
    ]
    return files


if __name__ == "__main__":
    get_service_list()
    print(get_models_list())

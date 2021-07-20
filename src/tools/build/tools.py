import os.path
import pathlib
import inspect
import re


def is_int(_, value: str):
    try:
        int(value)
        return True
    except ValueError as _:
        print("\nPlease enter a number")
        return False


def file_exists(_, value: str):
    tmp = pathlib.Path(value)
    return tmp.exists()


# create a list of all Python files names of module folder
def get_list_of_module(module, ignored_files=[]):
    result = os.listdir(
        os.path.abspath(os.path.dirname((inspect.getsourcefile(module))))
    )
    return [
        val[:-3]
        for val in result
        if re.search(r".*\.py$", val)
        and val != "__init__.py"
        and val not in ignored_files
    ]


def get_list_of_sub_directories(directory_path, ignored=[]):
    result = [
        os.path.join(directory_path, o)
        for o in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, o)) and o not in ignored
    ]
    return result


def get_list_of_files(directory_path, ignored=[]):
    result = [
        os.path.join(directory_path, o)
        for o in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, o)) and o not in ignored
    ]
    return result

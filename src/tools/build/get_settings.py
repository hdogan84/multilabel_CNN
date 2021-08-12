from inquirer import Text, Path, List, prompt
from tools.build.tools import is_int


def main():
    result = prompt(
        [
            Text(
                name="num_workers",
                message="Enter number of worker threads!",
                validate=is_int,
            ),
            Text(name="batch_size", message="Enter batch size!", validate=is_int),
        ]
    )
    result["num_workers"] = int(result["num_workers"])
    result["batch_size"] = int(result["batch_size"])
    return result

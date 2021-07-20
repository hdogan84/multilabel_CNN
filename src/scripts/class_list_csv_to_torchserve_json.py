import csv
import json
import pandas


def convert_csv_to_json(filepath, target_filepath):
    with open(filepath) as csvDataFile:
        data_frame = pandas.read_csv(csvDataFile, delimiter=";")
        output_dict = {}

        for index, row in enumerate(data_frame["latin_name"], start=0):
            output_dict[index] = row

        with open(target_filepath, "w+") as outfile:
            json.dump(output_dict, outfile)


if __name__ == "__main__":
    # Ask user an create build folder with contents
    convert_csv_to_json(
        "/home/bewr/projects/mfn/audio-classificator/logs/test_run/version_1/class-list.csv",
        "index_to_name.json",
    )

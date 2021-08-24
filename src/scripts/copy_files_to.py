

import argparse
import pandas
import shutil
import os
from pathlib import Path
def copy_files(csv_filepath, to_path,row_name='filepath',source_root ="./data/libro_animalis"):
    with open(csv_filepath) as csvDataFile:
        data_frame = pandas.read_csv(csvDataFile, delimiter=";")
        output_dict = {}

        for filepath in data_frame[row_name]:
            source = Path(source_root).joinpath(filepath)
            destination = Path(to_path).joinpath(filepath)
            if(destination.exists()):
                continue
            if(not destination.parents[0].exists()):
                os.makedirs(destination.parents[0])
            # print(destination)
            shutil.copy(source,destination)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("csv")
    parser.add_argument("to")
    args = parser.parse_args()
    copy_files( args.csv, args.to)
    


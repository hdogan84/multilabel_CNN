
from os import scandir
from pathlib import Path

def scan_file_list(file_list):
    if (isinstance(file_list, str)): #if string is presented convert to list
        file_list = [file_list]
    result = []
    for filepath in file_list:
        if (isinstance(filepath, str)):
            filepath = Path(filepath)
        if(filepath.is_dir()):
            result = result + scan_file_list(list(filepath.iterdir()))

        else:
            if filepath.suffix == '.wav' or filepath.suffix == '.mp3':
                print(filepath)
                result.append(filepath)

    
    return result

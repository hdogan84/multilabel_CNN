from tools.build.main import main
import sys

version = sys.version_info[0]+sys.version_info[1]*0.1
if(version < 3.9):
    raise Exception("Must be using Python >=3.9")
if __name__ == "__main__":
    # Ask user an create build folder with contents
    main()

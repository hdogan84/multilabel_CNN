#/bin/bash
export AMMOD_DATA_CLASS_LIST_FILEPATH="../libro_animalis/class-list.csv"
export AMMOD_DATA_DATA_LIST_FILEPATH="../libro_animalis/labels.csv"
export AMMOD_DATA_DATA_PATH="/mnt/tsa_transfer/TrainData/libro_animalis/"
python src/main.py --env AMMOD

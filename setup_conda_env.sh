#/bin/bash
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda install scikit-image -y
conda install pytorch-lightning -c conda-forge -y
conda install pandas -y
conda install -c conda-forge librosa -y
conda install --name base black -y
pip install typed-config
pip install soundfile
pip install audiomentations
pip install -U albumentations
pip install inflection
pip install colorednoise

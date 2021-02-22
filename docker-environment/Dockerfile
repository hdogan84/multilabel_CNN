#FROM gcr.io/kaggle-gpu-images/python
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
# ARG USERNAME=vscode
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
# ENV PATH="/home/$USERNAME/miniconda3/bin:${PATH}"
# ARG PATH="/home/$USERNAME/miniconda3/bin:${PATH}"

# RUN apt-get update && apt-get -y upgrade
# RUN apt-get install -y wget bash



# Create the user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

#Install Conda
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y libgl1-mesa-glx

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN conda install scikit-image
RUN conda install pytorch-lightning -c conda-forge
RUN conda install pandas
RUN conda install -c conda-forge librosa
RUN conda install --name base black -y
RUN pip install typed-config
RUN pip install soundfile 
RUN pip install audiomentations
RUN pip install -U albumentations
RUN pip install inflection
RUN pip install colorednoise
RUN useradd -ms /bin/bash vscode

#create user 
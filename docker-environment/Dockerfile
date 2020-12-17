#FROM gcr.io/kaggle-gpu-images/python
FROM ubuntu:21.04
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV PATH="/home/$USERNAME/miniconda3/bin:${PATH}"
ARG PATH="/home/$USERNAME/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y wget bash



# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME


RUN cd ~/
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN mkdir .conda 
RUN bash Miniconda3-latest-Linux-x86_64.sh -b 
RUN rm -f Miniconda3-latest-Linux-x86_64.sh 

# RUN apt-get upgrade
# RUN apt-get install -y apt-utils
# RUN apt-get install -y python3-dev libasound2-dev
# RUN pip install simpleaudio
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install pytorch-lightning -c conda-forge
RUN conda install pandas
RUN conda install -c conda-forge librosa
RUN pip install soundfile 
RUN conda install --name base black -y

RUN mkdir /mnt/data/
RUN mkdir /mnt/log/
RUN mkdir /tmp/.X11-unix/
#create user 
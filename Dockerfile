#FROM gcr.io/kaggle-gpu-images/python
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

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

#RUN conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -y
RUN conda install torchvision torchaudio -c pytorch -y
RUN conda install -c anaconda flask -y
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

RUN mkdir -p /service/build
RUN mkdir -p /service/src
RUN mkdir -p /service/data_directory

COPY ./src /service/src
COPY ./build /service/build

RUN useradd -ms /bin/bash vscode
EXPOSE 5000
ENV FLASK_ENV=production
ENV FLASK_APP=/service/src/service.py
ENV SERVICE_batch_size=32
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility 
ENV NVIDIA_VISIBLE_DEVICES=all
# start the appcd
WORKDIR /service
CMD [ "flask", "run","--host=0.0.0.0", "--port=5000" ]
#create user 
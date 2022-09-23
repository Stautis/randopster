FROM continuumio/miniconda3

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6 

COPY . /app

WORKDIR /app

RUN conda create --name randopenv --file env.txt \  
    
    && conda activate randopenv

RUN pip install coverpy
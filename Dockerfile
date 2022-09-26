FROM continuumio/miniconda3

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install ffmpeg libsm6 libxext6 

COPY . /app

WORKDIR /app

RUN conda create --name randopenv --file env.txt 

SHELL ["conda", "run", "-n", "randopenv", "/bin/bash", "-c"]

RUN pip install coverpy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "randopenv", "python", "docker_randopster.py"]
FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

RUN pip install --upgrade pip

COPY . /drbc
WORKDIR /drbc

RUN pip install .

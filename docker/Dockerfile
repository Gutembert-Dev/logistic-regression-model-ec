#!/usr/bin/env bash
FROM ubuntu:18.04

MAINTAINER Gutembert <nganpet2007@gmail.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    nginx \
    ca-certificates \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Install pip Requirements
COPY ./requirements_pip.txt ./requirements_pip.txt
RUN pip3 install pip --upgrade
RUN pip3 install --upgrade setuptools
RUN pip3 install -r ./requirements_pip.txt
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN mkdir -p /opt/ml/processing/input
RUN mkdir -p /opt/ml/processing/train
RUN mkdir -p /opt/ml/processing/validation
RUN mkdir -p /opt/ml/model

# Set up the program in the image
COPY explore_ai_demo /opt/program/explore_ai_demo
COPY explore_ai_demo/data/*.csv /opt/ml/processing/input

COPY scripts /opt/program/scripts/
WORKDIR /opt/program

RUN ln -s /opt/program/explore_ai_demo/serve /opt/program/serve
RUN ln -s /opt/program/explore_ai_demo/wsgi.py /opt/program/wsgi.py
RUN ln -s /opt/program/explore_ai_demo/nginx.conf /opt/program/nginx.conf
## Make the serve file executable
RUN chmod +x /opt/program/serve

EXPOSE 8080


#docker run --rm -v $(pwd)/models:/opt/ml/model -p 8080:8080 601385531861.dkr.ecr.eu-west-1.amazonaws.com/evolution_credit-ecr:1.0.1.dev173 serve
from ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt-get -y install \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    nano \
    vim \
    git \
    gcc \
    jags \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python-setuptools \
    tzdata \
    software-properties-common \
    libffi-dev \
    libgeos-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libspatialindex-dev \
    fonts-dejavu \
    gfortran \
    make \
    wget \
    curl

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.8-distutils

#add blobfuse dependencies
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN apt-get update && apt-get install -y blobfuse

RUN curl -o /tmp/get_pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.8 /tmp/get_pip.py

COPY Makefile ./
COPY requirements-minimal.txt ./

RUN python3.8 -m pip install -r requirements-minimal.txt

WORKDIR /code

CMD jupyter lab

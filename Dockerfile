FROM nvidia/cuda:10.0-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.6-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y wget
RUN apt-get install nano

RUN apt-get update -y && apt-get install -y --reinstall netbase

COPY requirements.txt .

RUN pip3 install -r requirements.txt

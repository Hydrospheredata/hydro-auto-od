FROM python:3.7.4-slim-stretch

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential \
                       python3-pip \
                       python3-numpy \
                       python3-scipy \
                       libatlas-dev \
                       libatlas3-base


RUN  update-alternatives --set libblas.so.3 \
    /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 \
    /usr/lib/atlas-base/atlas/liblapack.so.3

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app

COPY . /app
EXPOSE 5000


# use command: python app.py in docker-compose to launch
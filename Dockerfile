FROM python:3.8.2-slim-buster

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}

RUN apt-get update && \
    apt-get install -y -q build-essential git wget \
                          libatlas-base-dev libatlas3-base

# Download GRPC healthchecker and set the healthcheck param
RUN GRPC_HEALTH_PROBE_VERSION=v0.3.1 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe
HEALTHCHECK --start-period=10s CMD /bin/grpc_health_probe -addr=:${GRPC_PORT}

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app

COPY . /app
CMD python grpc_app.py
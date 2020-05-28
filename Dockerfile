FROM python:3.8.2-slim-buster AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libatlas-base-dev \
    libatlas3-base

RUN GRPC_HEALTH_PROBE_VERSION=v0.3.1 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --user -r ./requirements.txt


FROM python:3.8.2-slim-buster

RUN useradd --create-home --shell /bin/bash app
USER app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV PATH=/home/app/.local/bin:$PATH

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}
HEALTHCHECK --start-period=10s CMD /bin/grpc_health_probe -addr=:${GRPC_PORT}

COPY --from=build --chown=app:app /bin/grpc_health_probe /bin/grpc_health_probe
COPY --from=build --chown=app:app /root/.local /home/app/.local
COPY --chown=app:app version version
COPY --chown=app:app app /app

WORKDIR /app

CMD python3 grpc_app.py
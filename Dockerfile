FROM python:3.8.2-slim-buster AS base
ENV POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"

FROM base as build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libatlas-base-dev \
    libatlas3-base \
    curl

RUN GRPC_HEALTH_PROBE_VERSION=v0.3.1 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
RUN mv /root/.poetry $POETRY_PATH
RUN python -m venv $VENV_PATH
RUN poetry config virtualenvs.create false
RUN pip install --upgrade pip
COPY . ./

RUN poetry install --no-interaction --no-ansi -vvv


FROM base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app
WORKDIR /app

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
ENV UCF_FORCE_CONFOLD=1
ENV PYTHONUNBUFFERED=1

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}
HEALTHCHECK --start-period=10s CMD /bin/grpc_health_probe -addr=:${GRPC_PORT}

COPY --from=build --chown=app:app /bin/grpc_health_probe /bin/grpc_health_probe

COPY --from=build $VENV_PATH $VENV_PATH

COPY . ./

CMD . $VENV_PATH/bin/activate && python -m hydro_auto_od.server

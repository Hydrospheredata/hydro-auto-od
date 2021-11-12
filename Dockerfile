# syntax=docker/dockerfile:1
FROM python:3.8-slim-bullseye AS base
LABEL maintainer="support@hydrosphere.io"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_PATH=/opt/poetry \
    VENV_PATH=/opt/venv \
    POETRY_VERSION=1.1.6 
ENV PATH="$POETRY_PATH/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgssapi-krb5-2>=1.18.3-6+deb11u1 \
    libk5crypto3>=1.18.3-6+deb11u1 \
    libkrb5-3>=1.18.3-6+deb11u1 \
    libkrb5support0>=1.18.3-6+deb11u1 \
    libssl1.1>=1.1.1k-1+deb11u1 \
    openssl>=1.1.1k-1+deb11u1 && \
    rm -rf /var/lib/apt/lists/*


FROM base as build

# non-interactive env vars https://bugs.launchpad.net/ubuntu/+source/ansible/+bug/1833013
ENV DEBIAN_FRONTEND=noninteractive \
    DEBCONF_NONINTERACTIVE_SEEN=true \
    UCF_FORCE_CONFOLD=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libatlas-base-dev \
    libatlas3-base \
    wget && \
    \
    GRPC_HEALTH_PROBE_VERSION=v0.3.1 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe && \
    \
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python && \
    mv /root/.poetry $POETRY_PATH && \
    python -m venv $VENV_PATH && \
    poetry config virtualenvs.create false && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-ansi -vvv
ARG GIT_HEAD_COMMIT
ARG GIT_CURRENT_BRANCH
COPY . ./
RUN if [ -z "$GIT_HEAD_COMMIT" ] ; then \
    printf '{"name": "hydro-auto-od", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$(git rev-parse HEAD)" "$(git rev-parse --abbrev-ref HEAD)" "$(python --version)" >> buildinfo.json ; else \
    printf '{"name": "hydro-auto-od", "version":"%s", "gitHeadCommit":"%s","gitCurrentBranch":"%s", "pythonVersion":"%s"}\n' "$(cat version)" "$GIT_HEAD_COMMIT" "$GIT_CURRENT_BRANCH" "$(python --version)" >> buildinfo.json ; \
    fi

FROM base as runtime

RUN useradd -u 42069 --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

ENV GRPC_PORT=5000
EXPOSE ${GRPC_PORT}
HEALTHCHECK --start-period=10s CMD /bin/grpc_health_probe -addr=:${GRPC_PORT}

COPY --from=build --chown=app:app /bin/grpc_health_probe /bin/grpc_health_probe

COPY --from=build --chown=app:app $VENV_PATH $VENV_PATH

COPY --chown=app:app hydro_auto_od hydro_auto_od

RUN chmod -R 754 hydro_auto_od/resources

CMD python -m hydro_auto_od.server

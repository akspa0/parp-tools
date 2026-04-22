# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    V9_REPO_ROOT=/opt/parp-tools/gillijimproject_refactor

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${V9_REPO_ROOT}

COPY scripts/requirements_train_v7.txt /tmp/requirements_train_v7.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements_train_v7.txt \
    && python -m pip install \
        --index-url ${PYTORCH_INDEX_URL} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
    && python -c "import torch; print(f'TORCH={torch.__version__} CUDA={torch.version.cuda}')"

COPY src/WoWMapConverter/scripts/ ${V9_REPO_ROOT}/src/WoWMapConverter/scripts/
COPY docker/v9-trainer/entrypoint.sh ${V9_REPO_ROOT}/docker/v9-trainer/entrypoint.sh

RUN chmod +x ${V9_REPO_ROOT}/docker/v9-trainer/entrypoint.sh

WORKDIR ${V9_REPO_ROOT}/src/WoWMapConverter/scripts

ENTRYPOINT ["/opt/parp-tools/gillijimproject_refactor/docker/v9-trainer/entrypoint.sh"]
CMD []

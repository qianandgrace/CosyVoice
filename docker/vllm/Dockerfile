# CosyVoice3 inference image with vLLM-accelerated LLM decoding.
#
# Build from the repository root (the build context needs the whole repo):
#   docker build -f docker/vllm/Dockerfile -t cosyvoice-vllm:0.11.0 .
#
# vLLM 0.11.0 pins torch==2.8.0, well ahead of the torch==2.3.1 that
# requirements.txt targets for the rest of CosyVoice (flow matching, CausalHiFT,
# torchaudio). vLLM also brings its own matching torchaudio, so there is no
# torch/torchaudio mismatch. If the newer torch turns out to break the flow /
# vocoder path, rebuild with --build-arg INSTALL_VLLM=0 to get the exact
# upstream-tested torch 2.3.1 stack (without vLLM).

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG VLLM_VERSION=0.11.0
ARG INSTALL_VLLM=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Ubuntu 22.04 ships Python 3.10, which is what both torch 2.3.1 and vLLM 0.11 want.
# sox/libsox-dev are needed by torchaudio's sox backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        build-essential \
        git \
        git-lfs \
        curl \
        wget \
        ffmpeg \
        unzip \
        sed \
        sox \
        libsox-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# setuptools<81 keeps pkg_resources, which openai-whisper's sdist still imports.
RUN pip install --upgrade pip "setuptools<81" wheel

# Install the heavy ML stack first so pip resolves torch once. requirements.docker.txt
# deliberately carries no torch/transformers pins, so it will not touch these.
RUN if [ "$INSTALL_VLLM" = "1" ]; then \
        pip install "vllm==${VLLM_VERSION}" transformers==4.57.1 numpy==1.26.4; \
    else \
        pip install torch==2.3.1 torchaudio==2.3.1 transformers==4.51.3 numpy==1.26.4; \
    fi

COPY docker/vllm/requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install -r /tmp/requirements.docker.txt

# openai-whisper only ships an sdist whose setup.py imports pkg_resources, so
# build isolation has to be off. --no-deps matters just as much: whisper pins
# triton>=2.0.0,<3 while vLLM 0.11 ships triton 3.4, and letting pip reconcile
# that downgrades torch all the way back to 2.3.1 and silently breaks vLLM.
# Every runtime dep whisper declares (numba, numpy, torch, tqdm,
# more-itertools, tiktoken) is already present; triton is only used by the
# whisper CLI's timestamp alignment, which CosyVoice never touches.
RUN pip install --no-deps openai-whisper==20231117 --no-build-isolation \
    && python -c "import whisper; print('whisper', whisper.__file__)"

# ttsfrd ships as Linux wheels inside its modelscope repo. GIT_LFS_SKIP_SMUDGE
# avoids downloading the 676 MB resource archive, which is bind-mounted at run
# time from pretrained_models/CosyVoice-ttsfrd/resource instead.
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://www.modelscope.cn/models/iic/CosyVoice-ttsfrd.git /tmp/ttsfrd \
    && pip install /tmp/ttsfrd/ttsfrd_dependency-0.1-py3-none-any.whl \
                   /tmp/ttsfrd/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl \
    && rm -rf /tmp/ttsfrd

# Fail the build loudly if any install above dragged torch off vLLM's version.
# torchaudio is imported too on purpose: it is ABI-locked to torch, and a torch
# downgrade shows up here as 'undefined symbol' in libtorchaudio.so long before
# it shows up as a bad synthesis result.
RUN python -c "import torch, torchaudio, vllm, transformers; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| vllm', vllm.__version__, '| transformers', transformers.__version__)"

COPY . /workspace/CosyVoice
WORKDIR /workspace/CosyVoice

# Repo scripts append these at run time; setting PYTHONPATH means the container
# can be driven from any working directory.
ENV PYTHONPATH=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS

# Point wetext at the bind-mounted FST files instead of the modelscope API,
# which now returns 403 for anonymous downloads.
ENV COSYVOICE_WETEXT_DIR=/workspace/CosyVoice/pretrained_models/wetext

CMD ["/bin/bash"]

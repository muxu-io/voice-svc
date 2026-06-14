# voice-svc — standalone CUDA image: FastAPI + XTTS-v2 (Coqui TTS fork) + Piper.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip libpython3.12 \
    git ffmpeg \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Piper binary + the default voices (the caller sends these voice ids).
RUN curl -L -o /tmp/piper.tar.gz \
    https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz \
 && tar -C /opt -xzf /tmp/piper.tar.gz \
 && ln -s /opt/piper/piper /usr/local/bin/piper \
 && rm /tmp/piper.tar.gz \
 && mkdir -p /voices

# Default voices (the ids persona-side resolution falls back to).
RUN cd /voices && \
    curl -L -o en_GB-alba-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx && \
    curl -L -o en_GB-alba-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json && \
    curl -L -o en_GB-northern_english_male-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx && \
    curl -L -o en_GB-northern_english_male-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json && \
    curl -L -o en_US-libritts_r-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx && \
    curl -L -o en_US-libritts_r-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
COPY src /app/src

RUN python3.12 -m pip install 'poetry>=2,<3' && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main,engine

# Non-privileged runtime user (matches the muxu-io service hardening pattern).
RUN groupadd -g 10001 app && \
    useradd -u 10001 -g 10001 -m -d /home/app app && \
    mkdir -p /home/app/.cache/huggingface /home/app/.local/share/tts /home/app/.cache/voice-samples && \
    chown -R 10001:10001 /home/app

ENV VOICE_SVC_PORT=7000 \
    VOICE_SVC_PIPER_BIN=piper \
    VOICE_SVC_PIPER_MODEL_DIR=/voices \
    VOICE_SVC_SAMPLE_CACHE_DIR=/home/app/.cache/voice-samples \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    TTS_HOME=/home/app/.local/share/tts \
    COQUI_TOS_AGREED=1

USER 10001:10001

EXPOSE 7000
CMD ["python3.12", "-m", "voice_svc"]

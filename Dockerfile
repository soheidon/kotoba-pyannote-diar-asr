FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_VERBOSITY=error

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install -U pip setuptools wheel

WORKDIR /app

RUN python3.11 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchaudio==2.8.0
    
COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

COPY diar_asr.py /app/diar_asr.py
CMD ["python3.11", "/app/diar_asr.py"]

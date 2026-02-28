FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 必要パッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install -U pip setuptools wheel

WORKDIR /app

# Torch系を先行インストール（cu126指定）
RUN python3.11 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0+cu126 torchaudio==2.8.0+cu126

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY diar_asr.py .

CMD ["python3.11", "diar_asr.py"]

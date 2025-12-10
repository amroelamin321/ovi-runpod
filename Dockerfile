FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev build-essential git \
    wget curl ca-certificates libsndfile1 ffmpeg libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone Ovi code (no models)
RUN git clone https://github.com/character-ai/Ovi.git /workspace/ovi

# Install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Verify imports
RUN python -c "import torch; print('PyTorch OK')"
RUN python -c "from diffusers import FluxPipeline; print('Diffusers OK')"

# Copy handler
COPY handler.py /workspace/handler.py

RUN mkdir -p /tmp/video-output

ENV PYTHONPATH="/workspace/ovi:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]

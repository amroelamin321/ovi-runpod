FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/character-ai/Ovi.git /ovi

WORKDIR /ovi

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention skipped"

RUN pip install --no-cache-dir runpod cloudinary requests pillow python-dotenv

RUN mkdir -p /root/.cache/ovi_models /tmp/ovi_output

COPY handler.py /handler.py
COPY utils.py /utils.py

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "/handler.py"]

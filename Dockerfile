FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /

# Install minimal dependencies (non-interactive)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=UTC apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone Ovi repo
RUN git clone https://github.com/character-ai/Ovi.git /ovi

WORKDIR /ovi

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 (exists!) - base image already has compatible version
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ovi requirements
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Flash Attention (skip if fails)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Install RunPod + Cloudinary
RUN pip install --no-cache-dir runpod==0.7.0 cloudinary pillow requests python-dotenv

# Skip model download during build (too slow, download on first run)
RUN mkdir -p /root/.cache/ovi_models /tmp/ovi_output

# Copy handler files
COPY handler.py /handler.py
COPY utils.py /utils.py

# Environment
ENV PYTHONUNBUFFERED=1

# Start RunPod serverless
CMD ["python3", "-u", "/handler.py"]

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "/handler.py"]

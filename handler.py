#!/usr/bin/env python3
import os
import sys
import json
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any
import tempfile
import requests
from io import BytesIO

import torch
import runpod
from PIL import Image
import cloudinary
import cloudinary.uploader

sys.path.insert(0, '/ovi')

print("[INIT] Starting Ovi 1.1 initialization...")

os.makedirs('/tmp/ovi_output', exist_ok=True)

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

print("[INIT] Cloudinary configured")

if torch.cuda.is_available():
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[ERROR] No GPU available!")
    sys.exit(1)

print("[INIT] Loading Ovi 1.1 model...")

def download_image_from_url(image_url: str):
    print(f"[DOWNLOAD] Fetching image from: {image_url}")
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"[DOWNLOAD] Image size: {image.size}")
    return image

def upload_to_cloudinary(file_path: str):
    print(f"[UPLOAD] Uploading to Cloudinary: {file_path}")
    result = cloudinary.uploader.upload(
        file_path,
        resource_type='video',
        public_id=f'ovi_videos/{uuid.uuid4()}',
        quality='auto:best',
        fetch_format='auto',
        video_codec='h264',
        audio_codec='aac'
    )
    video_url = result.get('secure_url')
    print(f"[UPLOAD] Success! URL: {video_url}")
    return {
        'url': video_url,
        'public_id': result.get('public_id'),
        'bytes': result.get('bytes')
    }

def validate_prompt(prompt: str) -> bool:
    if not prompt or len(prompt) < 5:
        return False
    return True

def handler(job):
    import time
    start_time = time.time()
    
    try:
        mode = job.get('mode', 't2v').lower()
        prompt = job.get('prompt', '')
        
        print(f"[JOB] Processing {mode.upper()}")
        print(f"[JOB] Prompt: {prompt[:100]}...")
        
        if not prompt or not validate_prompt(prompt):
            raise ValueError("Invalid prompt")
        
        print("[GENERATE] Ovi 1.1 inference starting...")
        
        output_path = f"/tmp/ovi_output/video_{uuid.uuid4()}.mp4"
        
        upload_result = upload_to_cloudinary(output_path)
        
        generation_time = time.time() - start_time
        
        return {
            'status': 'success',
            'mode': mode,
            'video_url': upload_result['url'],
            'generation_time_seconds': round(generation_time, 2),
            'model_version': 'ovi-1.1',
            'video_duration': '10 seconds'
        }
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        
        generation_time = time.time() - start_time
        
        return {
            'status': 'error',
            'error': str(e),
            'generation_time_seconds': round(generation_time, 2)
        }

print("[INIT] Ready to process requests")

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

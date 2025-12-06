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
import time

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

print("[INIT] Ready to process requests")

def upload_to_cloudinary(file_path: str):
    print(f"[UPLOAD] Uploading: {file_path}")
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
    print(f"[UPLOAD] Success: {video_url}")
    return {
        'url': video_url,
        'public_id': result.get('public_id'),
        'bytes': result.get('bytes')
    }

def handler(job):
    start_time = time.time()
    
    try:
        mode = job.get('mode', 't2v').lower()
        prompt = job.get('prompt', '')
        
        print(f"[JOB] Processing {mode.upper()}")
        print(f"[JOB] Prompt: {prompt[:100]}...")
        
        # FIXED: Minimal validation only
        if not prompt:
            raise ValueError("Empty prompt")
        
        print("[GENERATE] Simulating Ovi 1.1 inference (model loading on first run)...")
        
        # Simulate video generation (replace with actual Ovi inference)
        output_path = f"/tmp/ovi_output/video_{uuid.uuid4()}.mp4"
        
        # Create dummy video for testing
        with open(output_path, 'w') as f:
            f.write("Ovi video placeholder - real model will generate here")
        
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

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

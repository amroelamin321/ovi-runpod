#!/usr/bin/env python3
import os
import sys
import json
import uuid
import traceback
import time
import subprocess
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

# Create output directory
os.makedirs('/tmp/ovi_output', exist_ok=True)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

print("[INIT] Cloudinary configured")

# Check GPU
if torch.cuda.is_available():
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[ERROR] No GPU available!")
    sys.exit(1)

print("[INIT] Ready to process requests")

def create_test_video(output_path: str, duration: int = 3):
    """Create a real test video using ffmpeg"""
    print(f"[VIDEO] Creating test video: {output_path}")
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=blue:s=1280x720:d={duration}',
        '-vf', f"drawtext=fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:text='Ovi Test Video'",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        '-y',
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[VIDEO] Created: {output_path} ({os.path.getsize(output_path)} bytes)")

def download_image_from_url(image_url: str) -> Image.Image:
    """Download and return PIL Image from URL"""
    print(f"[DOWNLOAD] Fetching image: {image_url}")
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"[DOWNLOAD] Image size: {image.size}")
    return image

def upload_to_cloudinary(file_path: str) -> Dict[str, Any]:
    """Upload video to Cloudinary and return URL"""
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

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler"""
    start_time = time.time()
    
    try:
        # REQUIRED FIELDS
        input_data = job.get('input', {})
        mode = input_data.get('mode', 't2v').lower()
        prompt = input_data.get('prompt', '').strip()
        
        # Print full job input for debugging
        print(f"[JOB] Full input: {json.dumps(input_data, indent=2)}")
        print(f"[JOB] Mode: {mode}")
        print(f"[JOB] Prompt length: {len(prompt)} chars")
        print(f"[JOB] Prompt preview: {prompt[:100]}...")
        
        # VALIDATION
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if len(prompt) < 3:
            raise ValueError("prompt must be at least 3 characters")
        
        print(f"[JOB] Processing {mode.upper()} - VALIDATED")
        
        # GENERATE VIDEO
        print("[GENERATE] Running Ovi inference (TEST MODE)...")
        
        # Create real test video with ffmpeg
        output_path = f"/tmp/ovi_output/video_{uuid.uuid4()}.mp4"
        create_test_video(output_path, duration=3)
        
        # Upload to Cloudinary
        upload_result = upload_to_cloudinary(output_path)
        
        # Clean up local file
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"[CLEANUP] Removed local file")
        
        generation_time = time.time() - start_time
        
        return {
            'status': 'success',
            'mode': mode,
            'video_url': upload_result['url'],
            'generation_time_seconds': round(generation_time, 2),
            'model_version': 'ovi-1.1-test',
            'video_duration': '3 seconds',
            'prompt_used': prompt,
            'cloudinary_public_id': upload_result['public_id']
        }
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        generation_time = time.time() - start_time
        
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'generation_time_seconds': round(generation_time, 2)
        }

# Start RunPod serverless worker
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

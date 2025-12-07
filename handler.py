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

# Add Ovi to Python path
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
if not torch.cuda.is_available():
    print("[ERROR] No GPU available!")
    sys.exit(1)
print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")

# TODO: Load Ovi model here
# Example:
# from ovi.model import OviPipeline
# ovi_pipeline = OviPipeline.from_pretrained("/root/.cache/ovi_models")
# ovi_pipeline = ovi_pipeline.to("cuda")
# print("[INIT] Ovi model loaded")

print("[INIT] Ready to process requests")

def download_image_from_url(image_url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    print(f"[DOWNLOAD] Fetching image: {image_url}")
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"[DOWNLOAD] Image size: {image.size}")
        return image
    except Exception as e:
        raise ValueError(f"Failed to download image from {image_url}: {str(e)}")

def create_test_video(output_path: str, duration: int = 3):
    """Create a test video using ffmpeg (FOR TESTING ONLY)"""
    print(f"[TEST] Creating test video: {output_path}")
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
    subprocess.run(cmd, check=True, capture_output=True, stderr=subprocess.PIPE)
    print(f"[TEST] Created: {output_path} ({os.path.getsize(output_path)} bytes)")

def generate_video_with_ovi(mode: str, prompt: str, image: Image.Image = None) -> str:
    """
    Generate video using Ovi model
    
    Args:
        mode: 't2v' or 'i2v'
        prompt: Text prompt
        image: PIL Image (required for i2v mode)
    
    Returns:
        Path to generated video file
    """
    output_path = f"/tmp/ovi_output/video_{uuid.uuid4()}.mp4"
    
    # TODO: Replace this with actual Ovi inference
    # Example for t2v:
    # video_frames = ovi_pipeline(
    #     prompt=prompt,
    #     num_frames=80,
    #     height=720,
    #     width=1280,
    #     num_inference_steps=50,
    #     guidance_scale=7.5
    # ).frames
    
    # Example for i2v:
    # video_frames = ovi_pipeline(
    #     prompt=prompt,
    #     image=image,
    #     num_frames=80,
    #     height=720,
    #     width=1280,
    #     num_inference_steps=50,
    #     guidance_scale=7.5
    # ).frames
    
    # Example: Save frames to video with ffmpeg
    # save_frames_to_video(video_frames, output_path, fps=24)
    
    # FOR NOW: Use test video
    print(f"[GENERATE] Running Ovi {mode.upper()} inference...")
    if mode == 'i2v' and image:
        print(f"[GENERATE] Using input image: {image.size}")
    
    create_test_video(output_path, duration=3)
    return output_path

def upload_to_cloudinary(file_path: str) -> Dict[str, Any]:
    """Upload video to Cloudinary and return details"""
    print(f"[UPLOAD] Uploading: {file_path} ({os.path.getsize(file_path)} bytes)")
    
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
        'bytes': result.get('bytes'),
        'duration': result.get('duration')
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler for Ovi video generation"""
    start_time = time.time()
    output_path = None
    
    try:
        # Extract input
        input_data = job.get('input', {})
        mode = input_data.get('mode', 't2v').lower()
        prompt = input_data.get('prompt', '').strip()
        image_url = input_data.get('image_url', '').strip()
        
        # Debug logging
        print(f"[JOB] Full input: {json.dumps(input_data, indent=2)}")
        print(f"[JOB] Mode: {mode}")
        print(f"[JOB] Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"[JOB] Prompt: {prompt}")
        if image_url:
            print(f"[JOB] Image URL: {image_url}")
        
        # Validate mode
        if mode not in ['t2v', 'i2v']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 't2v' or 'i2v'")
        
        # Validate prompt
        if not isinstance(prompt, str) or len(prompt) < 3:
            raise ValueError("prompt must be a string with at least 3 characters")
        
        # Handle i2v mode
        input_image = None
        if mode == 'i2v':
            if not image_url:
                raise ValueError("i2v mode requires 'image_url' parameter")
            input_image = download_image_from_url(image_url)
        
        print(f"[JOB] Processing {mode.upper()} - VALIDATED")
        
        # Generate video
        output_path = generate_video_with_ovi(
            mode=mode,
            prompt=prompt,
            image=input_image
        )
        
        # Upload to Cloudinary
        upload_result = upload_to_cloudinary(output_path)
        
        # Clean up local file
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
            print(f"[CLEANUP] Removed local file")
        
        generation_time = time.time() - start_time
        
        return {
            'status': 'success',
            'mode': mode,
            'video_url': upload_result['url'],
            'cloudinary_public_id': upload_result['public_id'],
            'generation_time_seconds': round(generation_time, 2),
            'model_version': 'ovi-1.1-test',
            'prompt_used': prompt,
            'video_duration_seconds': upload_result.get('duration'),
            'file_size_bytes': upload_result.get('bytes')
        }
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        
        # Clean up on error
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
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

#!/usr/bin/env python3
import os
import sys
import json
import uuid
import traceback

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import runpod
from PIL import Image
import cloudinary
import cloudinary.uploader
from omegaconf import OmegaConf

sys.path.insert(0, '/ovi')

from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video

print("[INIT] Starting Ovi initialization...")

os.makedirs('/tmp/ovi_output', exist_ok=True)

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)
print("[INIT] Cloudinary configured")

if not torch.cuda.is_available():
    print("[ERROR] No GPU available!")
    sys.exit(1)

device = 0
torch.cuda.set_device(device)
print(f"[INIT] GPU: {torch.cuda.get_device_name(device)}")

if os.path.ismount('/mnt/models') and os.path.exists('/mnt/models/ovi_models'):
    model_path = '/mnt/models/ovi_models'
    print("[INIT] Using Network Volume ✅")
else:
    model_path = '/root/.cache/ovi_models'
    print("[INIT] Using local cache ⚠️")

weights_exist = (
    os.path.exists(f'{model_path}/Ovi') and
    os.path.exists(f'{model_path}/Wan2.2-TI2V-5B') and
    os.path.exists(f'{model_path}/MMAudio')
)

if not weights_exist:
    print(f"[INIT] Downloading models...")
    result = os.system(f"cd /ovi && python3 download_weights.py --output-dir {model_path}")
    if result != 0:
        print(f"[ERROR] Download failed")
        sys.exit(1)
    print("[INIT] Models downloaded")
else:
    print("[INIT] Models cached ✅")

print("[INIT] Loading OVI config...")
# NOTE: DO NOT include video_frame_height_width in config - pass it to generate() instead
config = OmegaConf.create({
    'ckpt_dir': model_path,
    'output_dir': '/tmp/ovi_output',
    'solver_name': 'unipc',
    'shift': 5.0,
    'video_guidance_scale': 4.0,
    'audio_guidance_scale': 3.0,
    'slg_layer': 11,
    'video_negative_prompt': '',
    'audio_negative_prompt': '',
    'cpu_offload': True,
    'fp8': False,
    'seed': 100
})

print("[INIT] Loading OVI Engine...")
try:
    ovi_engine = OviFusionEngine(
        config=config,
        device=device,
        target_dtype=torch.bfloat16
    )
    print("[INIT] ✅ OVI loaded!")
except Exception as e:
    print(f"[ERROR] Load failed: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

print("[INIT] ✅ Ready!")

# ==================== HANDLER ====================

def handler(job):
    """RunPod handler - VERIFIED against official inference.py"""
    try:
        job_input = job['input']
        text_prompt = job_input.get('prompt', '')
        
        if not text_prompt:
            return {"status": "error", "message": "Prompt required"}
        
        print(f"[JOB] Text prompt: {text_prompt}")
        
        try:
            # EXACT parameters from inference.py lines 94-107
            generated_video, generated_audio, generated_image = ovi_engine.generate(
                text_prompt=text_prompt,                    # ✅ Verified
                image_path=None,                             # ✅ For T2V mode
                video_frame_height_width=[720, 720],        # ✅ Verified - pass here, not in config
                seed=100,                                    # ✅ Verified
                solver_name='unipc',                         # ✅ Verified
                sample_steps=50,                             # ✅ Verified (NOT num_steps)
                shift=5.0,                                   # ✅ Verified
                video_guidance_scale=4.0,                   # ✅ Verified
                audio_guidance_scale=3.0,                   # ✅ Verified
                slg_layer=11,                                # ✅ Verified
                video_negative_prompt='',                   # ✅ Verified
                audio_negative_prompt=''                    # ✅ Verified
            )
            
            print(f"[JOB] Video generated")
            
            # EXACT save_video call from inference.py line 109
            output_path = '/tmp/ovi_output/output.mp4'
            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            
            print(f"[JOB] Video saved to {output_path}")
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Generation: {str(e)}"}
        
        # Upload to Cloudinary
        try:
            print(f"[JOB] Uploading to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                output_path,
                resource_type="video",
                public_id=f"ovi_{uuid.uuid4()}"
            )
            
            print(f"[JOB] ✅ Success")
            return {
                "status": "success",
                "video_url": upload_result['secure_url'],
                "duration": upload_result.get('duration', 'N/A')
            }
        except Exception as e:
            print(f"[ERROR] Upload failed: {str(e)}")
            return {"status": "error", "message": f"Upload: {str(e)}"}
        
    except Exception as e:
        print(f"[ERROR] Handler exception: {str(e)}")
        print(traceback.format_exc())
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})

import runpod
import os
import sys
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cloudinary
    import cloudinary.uploader
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET')
    )
    CLOUDINARY_ENABLED = True
    logger.info("✓ Cloudinary configured")
except:
    CLOUDINARY_ENABLED = False
    logger.warning("⚠ Cloudinary not configured")


class OviVideoGenerator:
    def __init__(self):
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("="*80)
        logger.info("OVI 1.1 INITIALIZING")
        logger.info("="*80)
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"VRAM: {vram_gb:.2f} GB")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Model initialized")
        except Exception as e:
            logger.error(f"✗ Init failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_initialized = False

    def _initialize_model(self):
        """Initialize OviFusionEngine directly"""
        ovi_path = '/workspace/ovi'
        if ovi_path not in sys.path:
            sys.path.insert(0, ovi_path)
        
        if not os.path.exists(ovi_path):
            raise RuntimeError("Ovi repo not found")
        
        logger.info(f"✓ Ovi path: {ovi_path}")
        
        # Network volume mounts at /runpod-volume in serverless
        self.ckpt_dir = '/runpod-volume/ckpts'
        
        if not os.path.exists(self.ckpt_dir):
            raise RuntimeError(f"Models not found at {self.ckpt_dir}. Check network volume mount.")
        
        # Verify models
        size_mb = sum(f.stat().st_size for f in Path(self.ckpt_dir).rglob('*') if f.is_file()) / (1024**2)
        logger.info(f"✓ Models: {self.ckpt_dir} ({size_mb/1024:.1f} GB)")
        
        # Load base config with full path
        config_path = os.path.join(ovi_path, 'ovi/configs/inference/inference_fusion.yaml')
        if not os.path.exists(config_path):
            raise RuntimeError(f"Config not found: {config_path}")
        
        self.base_config = OmegaConf.load(config_path)
        
        # Override checkpoint directory in config
        self.base_config.ckpt_dir = self.ckpt_dir
        
        logger.info("✓ Config loaded")
        
        # Change to ovi directory before importing (module loads config on import)
        original_cwd = os.getcwd()
        os.chdir(ovi_path)
        
        try:
            # Import OviFusionEngine
            from ovi.ovi_fusion_engine import OviFusionEngine
            
            # Initialize engine
            target_dtype = torch.bfloat16
            self.ovi_engine = OviFusionEngine(
                config=self.base_config,
                device=self.device,
                target_dtype=target_dtype
            )
            
            logger.info("✓ OviFusionEngine loaded")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def generate_video(self, prompt: str, duration: int = 10, image_url: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate video using OviFusionEngine"""
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
        
        if not prompt:
            raise ValueError("Prompt required")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10")
        
        mode = 'i2v' if image_url else 't2v'
        logger.info(f"GENERATING {duration}s {mode.upper()}: {prompt[:60]}...")
        
        # Handle image download for i2v
        image_path = None
        if image_url:
            image_path = self._download_image(image_url)
        
        # Set video resolution based on duration
        # 5s uses 960x960, 10s uses 960x960 (both same resolution, different checkpoint)
        video_frame_height_width = [960, 960]
        
        # Determine seed
        if seed is None:
            seed = 100
        
        # Generation parameters (you can expose these in API later)
        solver_name = "unipc"
        sample_steps = 50
        shift = 5.0
        video_guidance_scale = 4.0
        audio_guidance_scale = 3.0
        slg_layer = 11
        video_negative_prompt = ""
        audio_negative_prompt = ""
        
        logger.info("⏳ Generating (60-90 seconds)...")
        
        try:
            generated_video, generated_audio, generated_image = self.ovi_engine.generate(
                text_prompt=prompt,
                image_path=image_path,
                video_frame_height_width=video_frame_height_width,
                seed=seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt
            )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
        
        # Save video to temp file
        os.makedirs("/tmp/video-output", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"/tmp/video-output/ovi_{mode}_{duration}s_{timestamp}.mp4"
        
        # Import save function
        from ovi.utils.io_utils import save_video
        
        save_video(output_file, generated_video, generated_audio, fps=24, sample_rate=16000)
        
        if not os.path.exists(output_file):
            raise RuntimeError("Output file not created")
        
        file_size = os.path.getsize(output_file)
        logger.info(f"✓ Video: {file_size/1024/1024:.2f} MB")
        
        # Save generated image if exists (for t2i2v mode)
        if generated_image is not None:
            image_output = output_file.replace('.mp4', '.png')
            generated_image.save(image_output)
            logger.info(f"✓ Image saved: {image_output}")
        
        return {
            'status': 'success',
            'video_path': output_file,
            'mode': mode,
            'duration': duration,
            'prompt': prompt,
            'file_size_mb': round(file_size/1024/1024, 2),
            'seed': seed
        }

    def _download_image(self, image_url: str) -> str:
        """Download image for i2v"""
        import urllib.request
        from PIL import Image
        
        logger.info(f"Downloading image: {image_url}")
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            urllib.request.urlretrieve(image_url, tmp.name)
            Image.open(tmp.name).verify()
            logger.info(f"✓ Image: {tmp.name}")
            return tmp.name

    def upload_to_cloudinary(self, video_path: str, prompt: str, duration: int, mode: str) -> Dict[str, Any]:
        """Upload to Cloudinary"""
        
        if not CLOUDINARY_ENABLED:
            logger.warning("Cloudinary not configured")
            return {
                'status': 'success',
                'url': f'file://{video_path}',
                'public_id': os.path.basename(video_path),
                'video_id': os.path.basename(video_path)
            }
        
        logger.info("Uploading to Cloudinary...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        response = cloudinary.uploader.upload(
            video_path,
            resource_type='video',
            public_id=f"ovi-{mode}-{duration}s-{timestamp}",
            folder='ovi-videos',
            tags=['ovi_1.1', mode, f'{duration}s'],
            timeout=600,
            chunk_size=6000000
        )
        
        logger.info(f"✓ Uploaded: {response['secure_url']}")
        
        return {
            'status': 'success',
            'url': response['secure_url'],
            'public_id': response['public_id'],
            'video_id': response['public_id']
        }

    def cleanup(self, video_path: str):
        """Clean up temp files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"✓ Cleaned: {video_path}")
            # Also clean up any .png files from t2i2v
            png_path = video_path.replace('.mp4', '.png')
            if os.path.exists(png_path):
                os.remove(png_path)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize generator
logger.info("Starting Ovi generator...")
try:
    generator = OviVideoGenerator()
    logger.info("✓ Generator ready")
except Exception as e:
    logger.error(f"✗ Generator init failed: {str(e)}")
    logger.error(traceback.format_exc())
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler
    
    Input:
    {
        "input": {
            "prompt": "A cat playing piano",
            "duration": 10,
            "image_url": "https://...",  // optional for i2v
            "seed": 42  // optional
        }
    }
    """
    
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        if generator is None or not generator.model_initialized:
            return {
                'status': 'error',
                'message': 'Model initialization failed. Check worker logs.'
            }
        
        # Extract params
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        
        if not prompt:
            return {'status': 'error', 'message': 'Prompt is required'}
        
        mode = 'i2v' if image_url else 't2v'
        logger.info(f"REQUEST: {mode.upper()} - {duration}s")
        
        # Generate video
        gen_result = generator.generate_video(
            prompt=prompt,
            duration=duration,
            image_url=image_url,
            seed=seed
        )
        
        # Upload to Cloudinary
        upload_result = generator.upload_to_cloudinary(
            video_path=gen_result['video_path'],
            prompt=prompt,
            duration=duration,
            mode=gen_result['mode']
        )
        
        # Cleanup
        generator.cleanup(gen_result['video_path'])
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ SUCCESS: {upload_result['url']} ({generation_time:.1f}s)")
        
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'prompt': prompt,
            'seed': gen_result['seed']
        }
        
    except Exception as e:
        logger.error(f"✗ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'status': 'error',
            'message': str(e)
        }


if __name__ == '__main__':
    logger.info("Starting RunPod serverless worker...")
    runpod.serverless.start({'handler': handler})

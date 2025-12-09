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
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
except Exception as e:
    CLOUDINARY_ENABLED = False
    logger.warning(f"⚠ Cloudinary not available: {e}")


class OviVideoGenerator:
    def __init__(self):
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("=" * 80)
        logger.info("INITIALIZING OVI 1.1 VIDEO GENERATOR (10-SECOND SUPPORT)")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"VRAM: {vram_gb:.2f} GB")
            
            if vram_gb < 70:
                logger.warning(f"⚠ WARNING: Ovi 1.1 (10s) requires ~80GB VRAM. You have {vram_gb:.1f}GB")
                logger.warning("⚠ Consider using cpu_offload or the 5-second model")
            
            capability = torch.cuda.get_device_capability(0)
            logger.info(f"CUDA Capability: sm_{capability[0]}{capability[1]}")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Ovi model initialized successfully")
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.model_initialized = False

    def _initialize_model(self):
        """Initialize Ovi with pre-downloaded models"""
        try:
            # Ovi repository path
            ovi_path = '/workspace/ovi'
            if ovi_path not in sys.path:
                sys.path.insert(0, ovi_path)
            
            if not os.path.exists(ovi_path):
                raise RuntimeError(f"Ovi repository not found at {ovi_path}")
            
            logger.info(f"✓ Ovi repository: {ovi_path}")
            
            # Checkpoint directory (from download_weights.py)
            self.ckpt_dir = '/workspace/ckpts'
            
            if not os.path.exists(self.ckpt_dir):
                raise RuntimeError(f"Checkpoints not found at {self.ckpt_dir}")
            
            # List downloaded models
            ckpt_contents = os.listdir(self.ckpt_dir)
            logger.info(f"✓ Checkpoints directory contains: {ckpt_contents}")
            
            # Verify inference script exists
            self.inference_script = os.path.join(ovi_path, 'inference.py')
            if not os.path.exists(self.inference_script):
                raise RuntimeError("inference.py not found")
            
            logger.info("✓ inference.py found")
            
            # Config file for 10-second generation
            self.config_file = os.path.join(ovi_path, 'ovi/configs/inference/inference_fusion.yaml')
            if not os.path.exists(self.config_file):
                raise RuntimeError("Config file not found")
            
            logger.info(f"✓ Config file: {self.config_file}")
            logger.info("✓ Model initialization complete - Ready for 10-second generation")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def generate_video(
        self,
        prompt: str,
        duration: int = 10,
        image_url: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate video using Ovi 1.1
        
        Args:
            prompt: Text description
            duration: Video length (5 or 10 seconds)
            image_url: Optional image for i2v mode
            seed: Random seed
            num_inference_steps: Diffusion steps (default: auto)
        
        Returns:
            Dict with status, video_path, mode, duration
        """
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10 seconds")
        
        try:
            mode = 'i2v' if image_url else 't2v'
            logger.info("=" * 80)
            logger.info(f"GENERATING {duration}s VIDEO ({mode.upper()})")
            logger.info(f"Prompt: {prompt}")
            if image_url:
                logger.info(f"Image: {image_url}")
            logger.info("=" * 80)
            
            # Output setup
            os.makedirs("/tmp/video-output", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"/tmp/video-output/ovi_{mode}_{duration}s_{timestamp}.mp4"
            
            # Build command
            cmd = [
                "python", self.inference_script,
                "--config-file", self.config_file,
                "--ckpt-dir", self.ckpt_dir,
                "--text-prompt", prompt,
                "--output-path", output_file,
            ]
            
            # Add seed if provided
            if seed is not None:
                cmd.extend(["--seed", str(seed)])
            
            # Handle i2v mode
            if image_url:
                image_path = self._download_image(image_url)
                cmd.extend(["--image-prompt", image_path])
                logger.info(f"✓ Image downloaded: {image_path}")
            
            # Add inference steps if provided
            if num_inference_steps:
                cmd.extend(["--num-inference-steps", str(num_inference_steps)])
            
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info("⏳ Generating (this takes ~60-90 seconds)...")
            
            # Run inference with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes max
                cwd='/workspace/ovi'
            )
            
            if result.returncode != 0:
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                raise RuntimeError(f"Generation failed: {result.stderr}")
            
            logger.info(f"Stdout: {result.stdout}")
            
            # Clean up downloaded image
            if image_url and 'image_path' in locals():
                try:
                    os.remove(image_path)
                except:
                    pass
            
            # Verify output
            if not os.path.exists(output_file):
                raise RuntimeError("Video file not created")
            
            file_size = os.path.getsize(output_file)
            logger.info("=" * 80)
            logger.info(f"✓ VIDEO GENERATED: {output_file}")
            logger.info(f"✓ Size: {file_size/1024/1024:.2f} MB")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'video_path': output_file,
                'mode': mode,
                'duration': duration,
                'prompt': prompt,
                'file_size_mb': round(file_size/1024/1024, 2)
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Generation timeout (>15min)")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _download_image(self, image_url: str) -> str:
        """Download image for i2v mode"""
        try:
            import urllib.request
            from PIL import Image
            
            logger.info(f"Downloading image: {image_url}")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_url, tmp.name)
                img = Image.open(tmp.name)
                img.verify()
                logger.info(f"✓ Image: {tmp.name} ({img.size[0]}x{img.size[1]})")
                return tmp.name
                
        except Exception as e:
            raise ValueError(f"Failed to download image: {str(e)}")

    def upload_to_cloudinary(
        self,
        video_path: str,
        prompt: str,
        duration: int,
        mode: str
    ) -> Dict[str, Any]:
        """Upload to Cloudinary"""
        
        if not CLOUDINARY_ENABLED:
            logger.warning("Cloudinary not configured")
            return {
                'status': 'success',
                'url': f'file://{video_path}',
                'public_id': os.path.basename(video_path),
                'video_id': os.path.basename(video_path)
            }
        
        try:
            logger.info(f"Uploading to Cloudinary: {video_path}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"ovi-{mode}-{duration}s-{timestamp}"
            
            response = cloudinary.uploader.upload(
                video_path,
                resource_type='video',
                public_id=public_id,
                folder='ovi-videos',
                tags=['ovi_1.1', f'{mode}', f'{duration}s'],
                timeout=600,
                chunk_size=6000000
            )
            
            logger.info(f"✓ Uploaded: {response['secure_url']}")
            
            return {
                'status': 'success',
                'url': response['secure_url'],
                'public_id': response['public_id'],
                'video_id': response['public_id'],
                'cloudinary_response': {
                    'size': response.get('bytes'),
                    'duration': response.get('duration'),
                    'format': response.get('format'),
                    'width': response.get('width'),
                    'height': response.get('height')
                }
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise

    def cleanup(self, video_path: str):
        """Clean up temp files"""
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"✓ Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


# Initialize
logger.info("=" * 80)
logger.info("STARTING OVI 1.1 VIDEO GENERATOR")
logger.info("=" * 80)

try:
    generator = OviVideoGenerator()
    logger.info("✓ Generator ready")
except Exception as e:
    logger.error(f"✗ Init failed: {str(e)}")
    logger.error(traceback.format_exc())
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler
    
    Input:
    {
        "input": {
            "prompt": "A cat playing piano",
            "duration": 10,  # 5 or 10 seconds
            "image_url": "https://...",  # optional for i2v
            "seed": 42  # optional
        }
    }
    
    Output:
    {
        "status": "success",
        "video_url": "https://cloudinary.com/...",
        "video_id": "ovi-t2v-10s-...",
        "mode": "t2v",
        "duration": 10,
        "generation_time_seconds": 85.3,
        "prompt": "..."
    }
    """
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        if generator is None or not generator.model_initialized:
            return {
                'status': 'error',
                'error_type': 'initialization_error',
                'message': 'Ovi not initialized. Check logs.'
            }
        
        # Extract params
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        
        if not prompt:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Prompt required'
            }
        
        mode = 'i2v' if image_url else 't2v'
        logger.info(f"REQUEST: {mode.upper()} - {duration}s - {prompt[:60]}...")
        
        # Generate
        gen_result = generator.generate_video(
            prompt=prompt,
            duration=duration,
            image_url=image_url,
            seed=seed
        )
        
        # Upload
        upload_result = generator.upload_to_cloudinary(
            video_path=gen_result['video_path'],
            prompt=prompt,
            duration=duration,
            mode=gen_result['mode']
        )
        
        # Cleanup
        generator.cleanup(gen_result['video_path'])
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info(f"✓ SUCCESS: {upload_result['url']}")
        logger.info(f"✓ Time: {generation_time:.1f}s")
        logger.info("=" * 80)
        
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'prompt': prompt,
            'file_size_mb': gen_result.get('file_size_mb')
        }
        
    except Exception as e:
        logger.error(f"✗ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'status': 'error',
            'error_type': 'generation_error',
            'message': f"{str(e)}"
        }


if __name__ == '__main__':
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    
    runpod.serverless.start({'handler': handler})

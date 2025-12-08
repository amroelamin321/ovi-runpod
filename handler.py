import runpod
import os
import sys
import torch
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
import cloudinary
import cloudinary.uploader
import tempfile
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

class OviVideoGenerator:
    def __init__(self):
        self.model_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Check CUDA capability
            capability = torch.cuda.get_device_capability(0)
            logger.info(f"CUDA Capability: sm_{capability[0]}{capability[1]}")
        
        try:
            self._initialize_model()
            self.model_initialized = True
            logger.info("✓ Ovi model initialized successfully")
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _initialize_model(self):
        """Load Ovi model using inference script from repository"""
        try:
            # Add Ovi repo to Python path
            ovi_path = '/workspace/ovi'
            if ovi_path not in sys.path:
                sys.path.insert(0, ovi_path)
            
            logger.info(f"Added {ovi_path} to Python path")
            logger.info(f"Ovi directory contents: {os.listdir(ovi_path)}")
            
            # Try to import Ovi's inference module
            try:
                # Check if there's an inference.py or run script
                inference_script = os.path.join(ovi_path, 'inference.py')
                if os.path.exists(inference_script):
                    logger.info(f"Found inference script: {inference_script}")
                else:
                    logger.warning(f"No inference.py found, checking for alternative entry points...")
                    
                # For now, use subprocess to run Ovi
                self.ovi_command = f"cd {ovi_path} && python inference.py"
                
            except ImportError as e:
                logger.error(f"Could not import Ovi modules: {e}")
                raise
            
            self.model_path = '/models/ovi-1-1'
            logger.info(f"Model path: {self.model_path}")
            
            # Verify model files exist
            if os.path.exists(self.model_path):
                logger.info(f"Model files found: {os.listdir(self.model_path)[:5]}")
            else:
                logger.warning(f"Model path does not exist: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Ovi: {str(e)}")
            raise

    def generate_video(
        self,
        prompt: str,
        duration: int = 10,
        image_url: Optional[str] = None,
        seed: Optional[int] = None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
        
        if not prompt or len(prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if duration not in [5, 10]:
            raise ValueError("Duration must be 5 or 10 seconds")
        
        try:
            logger.info(f"Generating {duration}s video: {prompt[:50]}...")
            
            # Create temporary output file
            output_file = f"/tmp/video-output/ovi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Build Ovi command
            import subprocess
            
            cmd = [
                "python", "/workspace/ovi/inference.py",
                "--prompt", prompt,
                "--output", output_file,
                "--duration", str(duration),
                "--model_path", self.model_path
            ]
            
            if seed is not None:
                cmd.extend(["--seed", str(seed)])
            
            if image_url:
                # Download image for I2V
                image_path = self._download_image(image_url)
                cmd.extend(["--image", image_path])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run Ovi inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode != 0:
                logger.error(f"Ovi stderr: {result.stderr}")
                raise RuntimeError(f"Ovi generation failed: {result.stderr}")
            
            logger.info("Video generation completed")
            
            return {
                'status': 'success',
                'video_path': output_file,
                'mode': 'i2v' if image_url else 't2v',
                'duration': duration,
                'prompt': prompt
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video generation timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            raise

    def _download_image(self, image_url: str) -> str:
        try:
            import urllib.request
            from PIL import Image
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                urllib.request.urlretrieve(image_url, tmp.name)
                Image.open(tmp.name)
                logger.info(f"Image downloaded: {tmp.name}")
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
        
        try:
            logger.info(f"Uploading to Cloudinary: {video_path}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"ovi-{mode}-{duration}s-{timestamp}"
            
            response = cloudinary.uploader.upload(
                video_path,
                resource_type='video',
                public_id=public_id,
                folder='ovi-videos',
                tags=['ovi_generated', f'ovi_{mode}', f'{duration}s'],
                timeout=300
            )
            
            logger.info(f"✓ Upload successful: {response['secure_url']}")
            
            return {
                'status': 'success',
                'url': response['secure_url'],
                'public_id': response['public_id'],
                'video_id': response['public_id'],
                'cloudinary_response': {
                    'size': response.get('bytes'),
                    'duration': response.get('duration'),
                    'format': response.get('format')
                }
            }
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            raise

    def cleanup(self, video_path: str):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up: {video_path}")
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")


try:
    generator = OviVideoGenerator()
except Exception as e:
    logger.error(f"Failed to initialize generator: {str(e)}")
    generator = None


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get('input', {})
    start_time = datetime.now()
    
    try:
        if generator is None:
            return {
                'status': 'error',
                'error': 'Model initialization failed',
                'message': 'Ovi model failed to initialize. Check worker logs.'
            }
        
        prompt = job_input.get('prompt', '').strip()
        duration = job_input.get('duration', 10)
        image_url = job_input.get('image_url')
        seed = job_input.get('seed')
        
        if not prompt:
            return {'status': 'error', 'error': 'Prompt is required'}
        
        logger.info(f"Processing: duration={duration}s, mode={'i2v' if image_url else 't2v'}")
        
        gen_result = generator.generate_video(
            prompt=prompt,
            duration=duration,
            image_url=image_url,
            seed=seed
        )
        
        upload_result = generator.upload_to_cloudinary(
            video_path=gen_result['video_path'],
            prompt=prompt,
            duration=duration,
            mode=gen_result['mode']
        )
        
        generator.cleanup(gen_result['video_path'])
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status': 'success',
            'video_url': upload_result['url'],
            'video_id': upload_result['video_id'],
            'mode': gen_result['mode'],
            'duration': duration,
            'generation_time_seconds': round(generation_time, 2),
            'prompt': prompt
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            'status': 'error',
            'error_type': 'generation_error',
            'message': str(e)
        }


if __name__ == '__main__':
    logger.info("Starting Ovi RunPod serverless worker...")
    runpod.serverless.start({'handler': handler})

import torch
from PIL import Image
import numpy as np

def load_image_and_resize(image_path: str, target_size: int = 960) -> Image.Image:
    image = Image.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if new_width != new_height:
        square = Image.new('RGB', (target_size, target_size), color='white')
        offset = ((target_size - new_width) // 2, (target_size - new_height) // 2)
        square.paste(image, offset)
        return square
    
    return image

def get_guidance_scales(video_scale: float = 4.0, audio_scale: float = 3.0):
    if not (0.0 < video_scale <= 10.0):
        raise ValueError(f"video_guidance_scale must be 0-10")
    
    if not (0.0 < audio_scale <= 10.0):
        raise ValueError(f"audio_guidance_scale must be 0-10")
    
    return {
        'video': video_scale,
        'audio': audio_scale
    }

def parse_seed(seed: int) -> int:
    if seed == -1:
        return torch.randint(0, 2**31, (1,)).item()
    return int(seed)

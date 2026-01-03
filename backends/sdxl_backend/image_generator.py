"""
Image Generation Logic for SDXL Backend
Handles text2img, img2img generation, and result encoding.
"""

import torch
import base64
import io
from PIL import Image
import logging

logger = logging.getLogger("sdxl_backend.image_generator")


def generate_text2img(pipeline, prompt, negative_prompt="", steps=4, guidance_scale=1.5, 
                      width=1024, height=1024, num_images=1):
    """
    Generate images from text prompts.
    Returns: list of PIL Images
    """
    logger.info(f"Generating Text2Image: '{prompt}' (Steps: {steps}, Size: {width}x{height})")
    
    with torch.inference_mode():
        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            num_images_per_prompt=int(num_images)
        ).images
    
    return images


def generate_img2img(img2img_pipeline, prompt, image, negative_prompt="", steps=4, 
                     guidance_scale=1.5, width=1024, height=1024, num_images=1, strength=0.75):
    """
    Generate images from image + text prompts.
    Returns: list of PIL Images
    """
    logger.info(f"Generating Img2Img: '{prompt}' (Steps: {steps}, Strength: {strength})")
    
    # Parse input image
    if isinstance(image, str) and "," in image:
        image = image.split(",")[1]  # Remove data:image/png;base64, prefix
    
    init_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    init_image = init_image.resize((int(width), int(height)))
    
    logger.info(f"Input Image Size: {init_image.size}")
    
    with torch.inference_mode():
        images = img2img_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            num_images_per_prompt=int(num_images)
        ).images
    
    return images


def encode_results(images):
    """
    Encode PIL Images to base64 strings.
    Returns: list of base64 strings
    """
    results = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        results.append(img_str)
    return results


def track_vram_usage():
    """
    Track peak VRAM usage and reset stats.
    Returns: dict with VRAM stats
    """
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    
    return {
        "peak_vram_mb": peak_vram / (1024 * 1024)
    }


def generate_image(pipeline, img2img_pipeline_getter, prompt, negative_prompt="", steps=4, 
                   guidance_scale=1.5, width=1024, height=1024, num_images=1, 
                   image=None, strength=0.75, scheduler_setter=None, scheduler="euler", **kwargs):
    """
    High-level generation function that handles both text2img and img2img.
    Returns: dict with images and stats
    """
    if not pipeline:
        raise RuntimeError("Pipeline not initialized")
    
    try:
        # Configure Scheduler
        if scheduler_setter:
            scheduler_setter(scheduler)
        
        # Generate images
        if image:
            # Img2Img mode
            img2img_pipeline = img2img_pipeline_getter()
            images = generate_img2img(
                img2img_pipeline, prompt, image, negative_prompt, 
                steps, guidance_scale, width, height, num_images, strength
            )
        else:
            # Text2Image mode
            images = generate_text2img(
                pipeline, prompt, negative_prompt, 
                steps, guidance_scale, width, height, num_images
            )
        
        # Encode results
        results = encode_results(images)
        
        # Track VRAM
        stats = track_vram_usage()
        
        return {
            "images": results,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        torch.cuda.empty_cache()
        raise e

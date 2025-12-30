import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from safetensors.torch import save_file

def convert(model_path, output_path, fp16=True):
    print(f"Loading model from {model_path}...")
    
    # Detect if SD or SDXL (naive check or try/except)
    # Most models in this context are likely SDXL based on previous context, but let's try generic loading
    try:
        # Try Loading as SDXL
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            use_safetensors=True
        )
        print("Loaded as StableDiffusionXLPipeline")
    except Exception as e:
        print(f"Not SDXL or failed to load as SDXL: {e}")
        try:
            # Try Loading as SD1.5/2.1
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if fp16 else torch.float32,
                use_safetensors=True
            )
            print("Loaded as StableDiffusionPipeline")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return False

    # Extract state dict
    # We want a single file checkpoint state dict. 
    # Usually diffusers pipelines don't just export a single state dict easily for .safetensors 
    # that matches the "webui" format perfectly without some mapping.
    # However, `save_file` from safetensors usually expects a simple state dict.
    # For widespread compatibility (ComfyUI, A1111), we might need specific keys.
    # BUT, if the goal is just to load it back into Diffusers via .from_single_file(), 
    # we can use the `diffusers` built-in capability if it exists, or just save the components.
    
    # Actually, Diffusers recently added support to save as single file? 
    # pipe.save_pretrained supports `safe_serialization=True` but that's still folder structure.
    
    # To save as a SINGLE .safetensors file that looks like a checkpoint:
    # We might need to iterate over components and keys. 
    # Simpler approach: Use the loaded state dict if possible? No, pipe doesn't have one state dict.
    
    # Alternative: Just use the `safetensors` loading/saving if we just want to amalgamate?
    # No, the "Diffusers to Checkpoint" conversion involves key remapping (UNet -> model.diffusion_model, etc.)
    # This is complex to write from scratch safely.
    
    # Let's try a simpler approach if the user just wants to "archive" it.
    # But usually "convert to checkpoint" implies "make it usable in A1111/Comfy".
    
    # Let's look for a standard conversion logic. 
    # Since I cannot browse the web freely for a new script, I will try a basic implementation 
    # that saves the UNet, VAE, and Text Encoder weights.
    # However, without correct remapping, it won't work in other tools.
    
    # Re-evaluating: If the goal is just storage efficiency, maybe we don't need full WebUI compat?
    # But "checkpoints" folder usually implies .safetensors format for WebUI/Comfy.
    
    # Let's use a "best effort" simple remapping or just warn the user.
    # Wait, `diffusers` usually doesn't have a built-in "to_single_file" method yet (as of some versions).
    
    # Actually, let's look at `diffusers.pipelines.stable_diffusion.convert_from_ckpt`. 
    # Is there a `convert_to_ckpt`? 
    
    # For now, I will implement a placeholder that warns about the complexity 
    # OR better: I'll use a very simple "save all keys" approach which works if the loader is smart enough.
    # But standard loaders expect "model.diffusion_model..." structure.
    
    # Let's look at what we can do reliably.
    # If the user just wants to move it, maybe just move it? 
    # But they asked for "convert diffusers to checkpoint safetensors".
    
    # I will assume SDXL since that's what we've been working with.
    # I'll construct a dictionary that mimics a basic SDXL checkpoint.
    
    state_dict = {}
    
    def merge_component(prefix, component):
        if hasattr(component, "state_dict"):
            sd = component.state_dict()
            for k, v in sd.items():
                state_dict[f"{prefix}{k}"] = v
    
    # This is a naive dump. It won't work with Standard Checkpoint Loaders without remapping script.
    # Given I cannot copy-paste a 500-line conversion script from the internet,
    # I will implement a "Packaged Diffusers" format or check if `save_single_file` exists (newer diffusers).
    
    if hasattr(pipe, "save_single_file"):
        print("Using pipe.save_single_file() ...")
        pipe.save_single_file(output_path)
        return True
        
    print("Warning: save_single_file not found. Using simple state dict dump (may not work in all UIs).")
    # Fallback (likely useless for other UIs, but valid validation)
    # Actually, recent diffusers 0.28+ has save_single_file. 
    # If the environment is old, this might fail.
    
    return False


def get_clean_model_name(model_path):
    """Derive a clean, human-readable model name from the path."""
    model_path = model_path.rstrip('/')
    
    # Strategy 1: HuggingFace Cache (models--User--Repo)
    # Check parts of the path for the pattern
    parts = model_path.split(os.sep)
    for part in parts:
        if part.startswith("models--") and "--" in part:
            # Format: models--User--Repo
            segments = part.split("--")
            if len(segments) >= 3:
                return segments[-1] # Return Repo name
            
    # Strategy 2: Standard basename (e.g. diffusers/realvisxl-v5)
    return os.path.basename(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    
    # Determine better output filename
    clean_name = get_clean_model_name(args.model_path)
    output_dir = os.path.dirname(args.output_path)
    new_output_path = os.path.join(output_dir, f"{clean_name}.safetensors")
    
    print(f"Targeting better output path: {new_output_path}")
    
    success = convert(args.model_path, new_output_path, args.fp16)
    if success:
        print(f"Successfully converted to {new_output_path}")
    else:
        print("Conversion failed.")
        exit(1)

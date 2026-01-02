import torch
from transformers import AutoModelForCausalLM, AutoProcessor


_backend_instance = None

def init_backend(**kwargs):
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = Backend()
    # If args passed, could configure here

def get_backend():
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = Backend()
    return _backend_instance

def caption(image_url=None, image=None, length="normal", stream=False, settings=None, **kwargs):
    # Support both image_url (from API) and image (direct)
    img_input = image if image is not None else image_url
    if not img_input:
        return {"error": "Image required"}
        
    image_obj = _load_image(img_input)
    return get_backend().caption(image_obj, length, stream, settings)

def query(image_url=None, image=None, question=None, stream=False, settings=None, reasoning=False, **kwargs):
    # Support both image_url (from API) and image (direct)
    img_input = image if image is not None else image_url
    if not img_input:
        return {"error": "Image required"}
        
    if not question:
         return {"error": "Question required"}
         
    image_obj = _load_image(img_input)
    return get_backend().query(image_obj, question, stream, settings, reasoning)

def unload():
    global _backend_instance
    if _backend_instance:
         # Clean up if needed
         if hasattr(_backend_instance, 'model'):
             del _backend_instance.model
         if hasattr(_backend_instance, 'processor'):
             del _backend_instance.processor
         import gc
         gc.collect()
         import torch
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
    _backend_instance = None


def _load_image(image_input):
    from PIL import Image
    import base64
    import io
    import os
    
    if isinstance(image_input, Image.Image):
        return image_input
        
    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            _, encoded = image_input.split(",", 1)
        else:
            # If it's a URL or file path that isn't data URI
            if os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            encoded = image_input
            
        try:
            raw_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except:
            # Fallback for plain file path if base64 failed
            if os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            raise ValueError("Invalid image input")
    return image_input


class Backend:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        print(f"Loading JoyCaption on {self.device}...")
        model_id = "mnenen/joy-caption-alpha-two"
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device).eval()
            
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return True
        except Exception as e:
            print(f"Error loading JoyCaption: {e}")
            raise e

    def caption(self, image, length="normal", stream=False, settings=None):
        if not self.model: self.load()
            
        prompt = "A descriptive caption for this image:\n"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        if self.device == 'cuda':
             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=300,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = generated_text.replace(prompt, "").strip()
        
        if stream:
            # Mock stream for now
            yield {"text": text}
        return {"text": text}

    def query(self, image, question, stream=False, settings=None, reasoning=False):
        if not self.model: self.load()
        
        prompt = question + "\n"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        if self.device == 'cuda':
             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=300
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = generated_text.replace(prompt, "").strip()
        
        if stream: yield {"text": text}
        return {"text": text}

    def detect(self, image, obj, settings=None, variant=None): return {"error": "Not supported"}
    def point(self, image, obj, settings=None, variant=None): return {"error": "Not supported"}

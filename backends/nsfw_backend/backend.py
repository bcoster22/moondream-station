import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import base64
import io
import logging

logger = logging.getLogger(__name__)

_model = None
_processor = None
_model_args = {}

def init_backend(**kwargs):
    global _model_args, _model, _processor
    _model_args = kwargs
    _model = None
    _processor = None
    logger.info(f"NSFW Backend initialized with args: {kwargs}")

def get_model():
    global _model, _processor
    if _model is None:
        model_id = _model_args.get("model_id", "Marqo/nsfw-image-detection-384")
        logger.info(f"Loading model: {model_id}")
        
        try:
            _processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
            _model = AutoModelForImageClassification.from_pretrained(model_id, trust_remote_code=True)
            
            if torch.cuda.is_available():
                _model = _model.cuda()
            
            _model.eval()
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise e
        
    return _model, _processor

def _load_base64_image(image_url: str) -> Image.Image:
    if image_url.startswith("data:image"):
        _, encoded = image_url.split(",", 1)
    else:
        encoded = image_url
    raw_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return image

def classify(image_url: str = None, **kwargs):
    if not image_url:
        return {"error": "image_url is required"}
        
    try:
        image = _load_base64_image(image_url)
        model, processor = get_model()
        
        inputs = processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get top prediction
        score, label_idx = torch.max(probs, dim=-1)
        label = model.config.id2label[label_idx.item()]
        score = score.item()
        
        return {
            "label": label,
            "score": score,
            "predictions": [
                {"label": model.config.id2label[i], "score": probs[0][i].item()}
                for i in range(len(probs[0]))
            ]
        }
    except Exception as e:
        logger.error(f"Error in classify: {e}")
        return {"error": str(e)}

def detect(image_url: str = None, **kwargs):
    result = classify(image_url, **kwargs)
    if "error" in result:
        return result
        
    return {
        "objects": [
            {"label": p["label"], "score": p["score"], "box": [0, 0, 1, 1]} 
            for p in result.get("predictions", [])
        ]
    }

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np

class Backend:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        print(f"Loading WD14 Tagger on {self.device}...")
        model_id = "SmilingWolf/wd-v1-4-vit-tagger-v2"
        self.model = ViTForImageClassification.from_pretrained(model_id).to(self.device).eval()
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        return True

    def caption(self, image, length="normal", stream=False, settings=None):
        if not self.model:
            self.load()
        
        # Check if input is list (Batch Mode)
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        
        # Prepare batch
        # ViTImageProcessor handles lists automatically
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Probs: [batch_size, num_labels]
        probs = torch.sigmoid(outputs.logits)
        
        # Get labels
        id2label = self.model.config.id2label
        probs_np = probs.cpu().numpy()
        
        batch_results = []
        
        for i in range(len(images)):
            # Filter for this image
            indices = np.where(probs_np[i] > 0.35)[0]
            
            results = []
            for idx in indices:
                tag = id2label[idx]
                # Skip rating tags generally
                if tag.startswith("rating:"): continue
                results.append(tag.replace("_", " "))
            
            text = ", ".join(results)
            batch_results.append({"text": text})
            
        # Return single dict if single input (backward compatibility)
        if not is_batch:
            if stream:
                yield batch_results[0]
            return batch_results[0]
            
        # Return list if batch input
        return batch_results

    def query(self, image, question, stream=False, settings=None, reasoning=False):
        return self.caption(image, stream=stream)

    def detect(self, image, obj, settings=None, variant=None):
        return {"error": "Not supported"}

    def point(self, image, obj, settings=None, variant=None):
        return {"error": "Not supported"}

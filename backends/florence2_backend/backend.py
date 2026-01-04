import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class Backend:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        print(f"Loading Florence-2 on {self.device}...")
        model_id = "microsoft/Florence-2-large"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return True

    def caption(self, image, length="normal", stream=False, settings=None):
        if not self.model: self.load()
        
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        
        if self.device == 'cuda':
             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        
        text = parsed_answer[task_prompt]
        
        if stream: yield {"text": text}
        return {"text": text}

    def query(self, image, question, stream=False, settings=None, reasoning=False):
        if not self.model: self.load()
        
        # Maps query to VQA task
        task_prompt = "<VQA>"
        prompt = f"{task_prompt}{question}"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        if self.device == 'cuda':
             inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        with torch.no_grad():
             generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        text = parsed_answer[task_prompt]
        
        if stream: yield {"text": text}
        return {"text": text}
    
    def detect(self, image, obj, settings=None, variant=None):
        return {"error": "Not implemented yet"}

    def point(self, image, obj, settings=None, variant=None):
        return {"error": "Not supported"}

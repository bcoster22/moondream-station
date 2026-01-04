import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import logging
import base64
import io

logger = logging.getLogger(__name__)

_model_service = None
_model_args = {}


def init_backend(**kwargs):
    """Initialize the backend with model arguments from manifest"""
    global _model_args, _model_service
    _model_args = kwargs
    _model_service = None  # Reset service to force recreation with new args
    logger.info(f"Backend initialized with args: {kwargs}")


def get_model_service():
    global _model_service
    if _model_service is None:
        model_id = _model_args.get("model_id", "vikhyatk/moondream2")
        revision_id = _model_args.get("revision_id", None)
        _model_service = ModelService(model_id, revision_id)
    return _model_service


class ModelService:
    def __init__(self, model_name: str, revision: str):
        self.model_name = model_name
        self.revision = revision
        self.device = self._get_best_device()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.model = self.model.to("mps")
        else:
            self.model = self.model.cpu()
        logger.info(f"Model commit hash: {self.model.config._commit_hash}")

    @staticmethod
    def _get_best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def caption(
        self,
        image: Image.Image,
        length: str,
        stream: bool = False,
        settings: dict = {},
        variant: str = None,
    ) -> dict:
        settings["variant"] = variant
        return self.model.caption(
            image, length=length, stream=stream, settings=settings
        )

    def query(
        self,
        image: Image.Image,
        question: str,
        stream: bool = False,
        settings: dict = {},
        variant: str = None,
        reasoning: bool = False,
    ) -> dict:
        settings["variant"] = variant
        return self.model.query(
            image, question, stream=stream, settings=settings, reasoning=reasoning
        )

    def detect(
        self,
        image: Image.Image,
        obj: str,
        settings: dict = {},
        variant=None,
    ) -> dict:
        settings["variant"] = variant
        return self.model.detect(image, obj, settings)

    def point(
        self, image: Image.Image, obj: str, settings: dict = {}, variant=None
    ) -> dict:
        settings["variant"] = variant
        return self.model.point(image, obj, settings)


def _load_base64_image(image_url: str) -> Image.Image:
    if image_url.startswith("data:image"):
        _, encoded = image_url.split(",", 1)
    else:
        encoded = image_url
    raw_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return image


def caption(
    image_url: str = None, length: str = "normal", stream: bool = False, **kwargs
):
    if not image_url:
        return {"error": "image_url is required"}

    try:
        image = _load_base64_image(image_url)
        service = get_model_service()
        result = service.caption(image, length, stream=stream)

        # For streaming, return the generator directly in the result
        if stream:
            return result  # This contains {"caption": <generator>}
        else:
            return {"caption": result.get("caption", "")}
    except Exception as e:
        return {"error": str(e)}


def query(
    image_url: str = None,
    question: str = None,
    stream: bool = False,
    reasoning: bool = False,
    **kwargs,
):
    if not image_url or not question:
        return {"error": "image_url and question are required"}

    try:
        image = _load_base64_image(image_url)
        service = get_model_service()
        result = service.query(image, question, stream=stream, reasoning=reasoning)

        # For streaming, return the generator directly in the result
        if stream:
            return result  # This contains {"answer": <generator>}
        else:
            return {"answer": result.get("answer", "")}
    except Exception as e:
        return {"error": str(e)}


def detect(image_url: str = None, object: str = None, obj: str = None, **kwargs):
    target_obj = object or obj
    if not image_url or not target_obj:
        return {"error": "image_url and object are required"}

    try:
        image = _load_base64_image(image_url)
        service = get_model_service()
        result = service.detect(image, target_obj)
        return {"objects": result.get("objects", [])}
    except Exception as e:
        return {"error": str(e)}


def point(image_url: str = None, object: str = None, obj: str = None, **kwargs):
    target_obj = object or obj
    if not image_url or not target_obj:
        return {"error": "image_url and object are required"}

    try:
        image = _load_base64_image(image_url)
        service = get_model_service()
        result = service.point(image, target_obj)
        return {
            "points": result.get("points", []),
            "count": len(result.get("points", [])),
        }
    except Exception as e:
        return {"error": str(e)}

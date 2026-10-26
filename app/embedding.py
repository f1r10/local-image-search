"""Pluggable image embeddings; each index is bound to a provider/model version."""
import io
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps


def unit(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if not len(arr) or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must contain finite, nonzero values")
    return arr / norm


class Lite:
    """Deterministic offline smoke-test baseline, NOT fine-grained visual AI."""
    key = "lite-rgb-v1"

    def embed(self, jpeg):
        with Image.open(io.BytesIO(jpeg)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            arr = np.asarray(image.resize((32, 32)), dtype=np.float32) / 255
        hist = [np.histogram(arr[:, :, i], bins=16, range=(0, 1))[0].astype("float32") for i in range(3)]
        thumbnail = np.asarray(Image.fromarray((arr * 255).astype("uint8")).resize((8, 8)), dtype=np.float32).reshape(-1) / 255
        return unit(np.concatenate([*(h / max(h.sum(), 1) * 3 for h in hist), thumbnail]))


class Gemini:
    key = "gemini-embedding-2-768"

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for IMAGE_PROVIDER=gemini")
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def embed(self, jpeg):
        from google.genai import types
        response = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")],
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("Gemini returned no embedding")
        return unit(response.embeddings[0].values)


class Dino:
    key = "facebook/dinov2-small-384"

    def __init__(self):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install requirements-dinov2.txt for DINOv2") from exc
        torch.set_num_threads(max(1, min(2, int(os.environ.get("TORCH_NUM_THREADS", "2")))))
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").eval().to("cpu")

    def embed(self, jpeg):
        with Image.open(io.BytesIO(jpeg)) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.inference_mode():
            features = self.model(**inputs).last_hidden_state[:, 0, :][0]
        return unit(features.cpu().numpy())


@lru_cache(maxsize=3)
def provider(name):
    if name == "lite":
        return Lite()
    if name == "gemini":
        return Gemini()
    if name == "dinov2":
        return Dino()
    if name == "dinov2-yolo-torch":
        from .vision_pipeline import DinoYolo
        return DinoYolo(backend="torch")
    if name == "dinov2-yolo-onnx":
        from .vision_pipeline import DinoYolo
        return DinoYolo(backend="onnx")
    raise ValueError("IMAGE_PROVIDER must be lite, gemini, dinov2, dinov2-yolo-torch, or dinov2-yolo-onnx")


def active_provider():
    return provider(os.environ.get("IMAGE_PROVIDER", "lite").strip().lower())

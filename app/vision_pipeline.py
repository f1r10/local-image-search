"""Colab-compatible YOLOv8n + DINOv2 Small local retrieval, CPU only.

Heavy dependencies are imported lazily. Both catalog and query use identical
preprocessing. COCO has no wristwatch class: do not crop person detections.
"""
import hashlib
import io
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .embedding import unit

MODEL_ID = "facebook/dinov2-small"
# COCO80 supports these products; notably NOT wristwatches.
PRODUCT_CLASSES = frozenset({
    "backpack", "handbag", "suitcase", "bottle", "cup", "bowl",
    "chair", "couch", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "book", "clock", "vase", "scissors", "teddy bear",
})


def settings():
    mode = os.getenv("YOLO_CROP_MODE", "auto").strip().lower()
    if mode not in {"auto", "off"}:
        raise ValueError("YOLO_CROP_MODE must be auto or off")
    raw = os.getenv("YOLO_CROP_CLASSES", "")
    classes = frozenset(x.strip().lower() for x in raw.split(",") if x.strip()) if raw else PRODUCT_CLASSES
    # Disallow person even if a custom class list accidentally includes it.
    classes = classes - {"person"}
    min_conf = float(os.getenv("YOLO_MIN_CONF", "0.35"))
    min_area = float(os.getenv("YOLO_MIN_AREA", "0.02"))
    if not 0 <= min_conf <= 1 or not 0 <= min_area < 1:
        raise ValueError("Invalid YOLO confidence/area settings")
    model = os.getenv("YOLO_MODEL", "yolov8n.pt")
    fingerprint = hashlib.sha256(
        repr((mode, sorted(classes), min_conf, min_area, Path(model).name)).encode()
    ).hexdigest()[:10]
    return mode, classes, min_conf, min_area, model, fingerprint


def crop_main_object(image, predictions, classes, min_conf, min_area):
    """Return the most reliable eligible product crop, or the original image.

    The raw YOLO 'most confident box' can target a person rather than a watch.
    No wristwatch class exists in standard COCO80; for watches this safely
    falls back to the whole image (not an assertion of object detection).
    """
    if not predictions:
        return image
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return image
    w, h = image.size
    chosen = None
    best_score = -1.0
    for box in boxes:
        label_id = int(box.cls[0])
        names = result.names
        label = names[label_id] if isinstance(names, (dict, list, tuple)) else ""
        if str(label).lower() not in classes:
            continue
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        area = max(0., x2-x1) * max(0., y2-y1) / (w*h)
        if area < min_area or area > 0.97:
            continue
        # Prefer higher confidence, with a small bonus for a visible object.
        score = conf + 0.05 * min(area, 0.5)
        if score > best_score:
            chosen, best_score = (x1, y1, x2, y2), score
    if chosen is None:
        return image
    x1, y1, x2, y2 = chosen
    pad = 0.04 * max(x2-x1, y2-y1)
    rect = (max(0, int(x1-pad)), max(0, int(y1-pad)),
            min(w, int(x2+pad)), min(h, int(y2+pad)))
    return image.crop(rect) if rect[2] > rect[0] and rect[3] > rect[1] else image


def mean_pool_embedding(last_hidden_state):
    """Match the working Colab notebook: mean all tokens, then unit normalize."""
    tensor = last_hidden_state
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    values = np.asarray(tensor, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError("Expected [1, tokens, features] from DINOv2")
    return unit(values.mean(axis=1)[0])


class DinoYolo:
    """Optional provider: PyTorch or ONNX DINOv2, optionally YOLOv8n cropping."""
    def __init__(self, backend="torch"):
        if backend not in {"torch", "onnx"}:
            raise ValueError("Unknown DINO backend")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install requirements-vision-cpu.txt and CPU-only PyTorch") from exc
        self.torch = torch
        torch.set_num_threads(max(1, min(4, int(os.getenv("TORCH_NUM_THREADS", "2")))))
        self.backend = backend
        mode, classes, conf, area, yolo_model, fingerprint = settings()
        self.mode, self.classes = mode, classes
        self.min_conf, self.min_area = conf, area
        self.key = "dinov2-yolo-mean-" + backend + ("-v2-" if backend == "onnx" else "-v1-") + fingerprint
        self._lock = threading.Lock()
        self.detector = None
        if mode == "auto":
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics dependency missing") from exc
            self.detector = YOLO(yolo_model)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        if backend == "torch":
            self.model = AutoModel.from_pretrained(MODEL_ID).eval().to("cpu")
        else:
            # Use ONNX Runtime directly to avoid Optimum's torch.int4 compatibility
            # failure with the 2.5 CPU build. Torch is used only for one-time export.
            try:
                import onnxruntime as ort
                import onnx  # required by torch.onnx.export
            except ImportError as exc:
                raise RuntimeError("Install requirements-vision-onnx.txt for ONNX") from exc
            cache = Path(os.getenv("DINO_ONNX_DIR", "model-cache/dinov2-onnx"))
            cache.mkdir(parents=True, exist_ok=True)
            graph = cache / "model-colab-v2.onnx"
            if not graph.is_file():
                class LastHidden(torch.nn.Module):
                    def __init__(self, encoder):
                        super().__init__()
                        self.encoder = encoder

                    def forward(self, pixel_values):
                        return self.encoder(pixel_values=pixel_values).last_hidden_state

                encoder = AutoModel.from_pretrained(MODEL_ID).eval().to("cpu")
                example = self.processor(
                    images=Image.new("RGB", (224, 224)), return_tensors="pt"
                )["pixel_values"]
                tmp = graph.with_suffix(".tmp.onnx")
                try:
                    torch.onnx.export(
                        LastHidden(encoder).eval(), (example,), str(tmp),
                        opset_version=17,
                        input_names=["pixel_values"],
                        output_names=["last_hidden_state"],
                        dynamic_axes={
                            "pixel_values": {0: "batch"},
                            "last_hidden_state": {0: "batch"},
                        },
                    )
                    tmp.replace(graph)
                finally:
                    tmp.unlink(missing_ok=True)
                del encoder
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(4, int(os.getenv("TORCH_NUM_THREADS", "2"))))
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(graph), sess_options=options, providers=["CPUExecutionProvider"]
            )

    def embed(self, jpeg):
        with self._lock:
            with Image.open(io.BytesIO(jpeg)) as image:
                prepared = image.convert("RGB")
            if self.detector is not None:
                result = self.detector.predict(prepared, device="cpu", verbose=False)
                prepared = crop_main_object(
                    prepared, result, self.classes, self.min_conf, self.min_area)
            inputs = self.processor(images=prepared, return_tensors="pt")
            if self.backend == "torch":
                with self.torch.inference_mode():
                    hidden = self.model(**inputs).last_hidden_state
            else:
                hidden = self.session.run(
                    ["last_hidden_state"],
                    {"pixel_values": inputs["pixel_values"].cpu().numpy()},
                )[0]
            return mean_pool_embedding(hidden)

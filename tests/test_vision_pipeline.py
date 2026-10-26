"""Dependency-light regression tests for Colab-compatible extraction rules."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.vision_pipeline import (
    PRODUCT_CLASSES, crop_main_object, mean_pool_embedding, settings,
)


def detection(label, confidence=0.9, rect=(20, 30, 120, 130)):
    box = SimpleNamespace(
        cls=np.array([0]), conf=np.array([confidence]),
        xyxy=np.array([rect], dtype=np.float32))
    return [SimpleNamespace(names={0: label}, boxes=[box])]


def test_person_is_never_cropped_as_a_product():
    img = Image.new("RGB", (200, 200), "white")
    assert crop_main_object(img, detection("person"), PRODUCT_CLASSES, 0.35, 0.02) is img


def test_unknown_watch_class_uses_original_image():
    img = Image.new("RGB", (200, 200), "white")
    assert crop_main_object(img, detection("watch"), PRODUCT_CLASSES, 0.35, 0.02) is img


def test_product_detection_can_crop_with_padding():
    img = Image.new("RGB", (200, 200), "white")
    cropped = crop_main_object(img, detection("cell phone"), PRODUCT_CLASSES, 0.35, 0.02)
    assert 100 < cropped.width < 200
    assert 100 < cropped.height < 200


def test_low_confidence_and_tiny_boxes_use_original():
    img = Image.new("RGB", (200, 200), "white")
    assert crop_main_object(img, detection("clock", 0.1), PRODUCT_CLASSES, 0.35, 0.02) is img
    assert crop_main_object(img, detection("clock", 0.9, (1, 1, 5, 5)), PRODUCT_CLASSES, 0.35, 0.02) is img


def test_mean_pool_matches_colab_and_l2_normalizes():
    hidden = np.array([[[3, 0], [3, 0], [0, 3]]], dtype=np.float32)
    result = mean_pool_embedding(hidden)
    expected = np.array([2, 1], dtype=np.float32) / np.sqrt(5)
    assert np.allclose(result, expected)
    assert np.isclose(np.linalg.norm(result), 1)


def test_bad_embedding_shape_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        mean_pool_embedding(np.zeros((384,), dtype=np.float32))


def test_crop_config_changes_index_identity(monkeypatch):
    monkeypatch.setenv("YOLO_CROP_MODE", "auto")
    a = settings()[-1]
    monkeypatch.setenv("YOLO_CROP_MODE", "off")
    b = settings()[-1]
    assert a != b


def test_config_never_enables_person_crop(monkeypatch):
    monkeypatch.setenv("YOLO_CROP_CLASSES", "person,cell phone")
    assert "person" not in settings()[1]
    assert "cell phone" in settings()[1]


def test_unknown_provider_rejected():
    import pytest
    from app.embedding import provider
    with pytest.raises(ValueError):
        provider("unknown-provider")

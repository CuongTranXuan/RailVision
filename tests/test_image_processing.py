"""Tests for the image processing utilities."""
import pytest
import numpy as np
import cv2
from src.utils.image_processing import (
    resize_image,
    normalize_image,
    draw_detections,
    crop_roi
)

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def detection_results():
    """Create sample detection results."""
    return {
        'boxes': np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'scores': np.array([0.95, 0.85]),
        'classes': np.array([0, 1])  # 0: speed sign, 1: traffic light
    }

def test_resize_image(test_image):
    """Test image resizing functionality."""
    target_size = (320, 240)
    resized = resize_image(test_image, target_size)

    assert resized.shape[:2] == target_size
    assert resized.dtype == test_image.dtype
    assert len(resized.shape) == 3

def test_resize_image_invalid_input():
    """Test resize_image with invalid inputs."""
    with pytest.raises(ValueError):
        resize_image(None, (320, 240))

    with pytest.raises(ValueError):
        resize_image(np.zeros((100, 100)), (320, 240))  # 2D array

    with pytest.raises(ValueError):
        resize_image(np.zeros((100, 100, 3)), (0, 0))  # Invalid target size

def test_normalize_image(test_image):
    """Test image normalization."""
    normalized = normalize_image(test_image)

    assert normalized.dtype == np.float32
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0
    assert normalized.shape == test_image.shape

def test_normalize_image_invalid_input():
    """Test normalize_image with invalid inputs."""
    with pytest.raises(ValueError):
        normalize_image(None)

    with pytest.raises(ValueError):
        normalize_image(np.zeros((100, 100)))  # 2D array

def test_draw_detections(test_image, detection_results):
    """Test drawing detection boxes and labels."""
    annotated = draw_detections(
        test_image.copy(),
        detection_results,
        class_names=['speed_sign', 'traffic_light']
    )

    assert annotated.shape == test_image.shape
    assert annotated.dtype == test_image.dtype
    # The annotated image should be different from the original
    assert not np.array_equal(annotated, test_image)

def test_draw_detections_no_detections(test_image):
    """Test drawing with no detections."""
    empty_results = {
        'boxes': np.array([]),
        'scores': np.array([]),
        'classes': np.array([])
    }

    annotated = draw_detections(
        test_image.copy(),
        empty_results,
        class_names=['speed_sign', 'traffic_light']
    )

    # Should return unchanged image
    assert np.array_equal(annotated, test_image)

def test_crop_roi(test_image):
    """Test region of interest cropping."""
    bbox = [100, 100, 200, 200]  # x1, y1, x2, y2
    roi = crop_roi(test_image, bbox)

    assert roi.shape == (100, 100, 3)  # Height and width should match bbox size
    assert np.array_equal(roi, test_image[100:200, 100:200])

def test_crop_roi_invalid_input(test_image):
    """Test crop_roi with invalid inputs."""
    with pytest.raises(ValueError):
        crop_roi(None, [100, 100, 200, 200])

    with pytest.raises(ValueError):
        crop_roi(test_image, [100, 100, 90, 90])  # Invalid bbox (x2 < x1)

    with pytest.raises(ValueError):
        crop_roi(test_image, [100, 100, 1000, 1000])  # bbox outside image

def test_crop_roi_edge_cases(test_image):
    """Test crop_roi with edge cases."""
    # Test cropping at image boundaries
    roi = crop_roi(test_image, [0, 0, 100, 100])
    assert roi.shape == (100, 100, 3)

    roi = crop_roi(test_image, [540, 380, 640, 480])
    assert roi.shape == (100, 100, 3)
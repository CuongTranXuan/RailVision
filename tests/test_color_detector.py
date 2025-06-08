"""Tests for the color detector module."""
import pytest
import numpy as np
import cv2
from src.models.color_detector import ColorDetector

@pytest.fixture
def color_detector():
    """Create a ColorDetector instance."""
    return ColorDetector()

@pytest.fixture
def red_light_image():
    """Create a test image with a red traffic light."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a red circle (traffic light)
    cv2.circle(img, (50, 50), 20, (0, 0, 255), -1)
    return img

@pytest.fixture
def green_light_image():
    """Create a test image with a green traffic light."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a green circle (traffic light)
    cv2.circle(img, (50, 50), 20, (0, 255, 0), -1)
    return img

def test_color_detector_initialization(color_detector):
    """Test ColorDetector initialization."""
    assert color_detector is not None
    assert hasattr(color_detector, 'detect_color')

def test_red_light_detection(color_detector, red_light_image):
    """Test detection of red traffic light."""
    roi = red_light_image[30:70, 30:70]  # Region of interest around the light
    color = color_detector.detect_color(roi)
    assert color == "red"

def test_green_light_detection(color_detector, green_light_image):
    """Test detection of green traffic light."""
    roi = green_light_image[30:70, 30:70]  # Region of interest around the light
    color = color_detector.detect_color(roi)
    assert color == "green"

def test_no_light_detection(color_detector):
    """Test detection with no clear traffic light."""
    # Create black image (no light)
    black_image = np.zeros((40, 40, 3), dtype=np.uint8)
    color = color_detector.detect_color(black_image)
    assert color == "unknown"

def test_invalid_input(color_detector):
    """Test color detection with invalid input."""
    with pytest.raises(ValueError):
        color_detector.detect_color(None)

    with pytest.raises(ValueError):
        color_detector.detect_color(np.array([]))  # Empty array

    with pytest.raises(ValueError):
        color_detector.detect_color(np.zeros((100, 100)))  # 2D array instead of 3D

def test_color_thresholds(color_detector):
    """Test color detection with different threshold values."""
    # Create an image with values just at the threshold boundaries
    test_image = np.zeros((40, 40, 3), dtype=np.uint8)

    # Test red threshold
    test_image[:, :] = [0, 0, 255]  # Pure red
    assert color_detector.detect_color(test_image) == "red"

    # Test green threshold
    test_image[:, :] = [0, 255, 0]  # Pure green
    assert color_detector.detect_color(test_image) == "green"

    # Test yellow threshold (if implemented)
    test_image[:, :] = [0, 255, 255]  # Pure yellow
    result = color_detector.detect_color(test_image)
    assert result in ["yellow", "unknown"]  # Depending on implementation
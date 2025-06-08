"""Shared test fixtures and configuration."""
import os
import pytest
import numpy as np
import cv2

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a 640x640 test image with a red rectangle (simulating a traffic sign)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)  # Red rectangle
    return img

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')

@pytest.fixture
def sample_yolo_result():
    """Create a sample YOLO detection result."""
    return {
        'boxes': np.array([[100, 100, 200, 200]]),  # x1, y1, x2, y2
        'scores': np.array([0.95]),
        'classes': np.array([0]),  # Assuming 0 is speed sign class
    }

@pytest.fixture
def mock_video_path(tmp_path):
    """Create a temporary mock video file."""
    video_path = tmp_path / "test_video.mp4"
    # Create a blank video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for _ in range(10):  # 10 frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(video_path)
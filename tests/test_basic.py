"""Basic integration tests for the railway detection system."""

import pytest
import numpy as np
import cv2
from src.models.yolo_detector import YOLODetector
from src.models.color_detector import ColorDetector
from src.utils.video_utils import VideoProcessor
from src.utils.image_processing import resize_image, draw_detections

@pytest.fixture
def test_video(mock_video_path):
    """Create a test video processor."""
    return VideoProcessor(mock_video_path)

@pytest.fixture
def detection_system():
    """Create instances of detection components."""
    yolo = YOLODetector(model_path="models/sign_yolov8_best_saved_model")
    color = ColorDetector()
    return yolo, color

def test_basic_imports():
    """Test that all required modules can be imported."""
    from src.models import yolo_detector, color_detector, ocr_processor
    from src.utils import image_processing, video_utils
    assert True  # If imports succeed, test passes

def test_video_frame_processing(test_video):
    """Test basic video frame processing pipeline."""
    ret, frame = test_video.read_frame()
    assert ret is True
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)

    # Test frame preprocessing
    processed_frame = resize_image(frame, (416, 416))
    assert processed_frame.shape == (416, 416, 3)

def test_detection_pipeline(test_video, detection_system, sample_image):
    """Test the complete detection pipeline."""
    yolo_detector, color_detector = detection_system

    # Process a frame
    ret, frame = test_video.read_frame()
    assert ret is True

    # Detect objects
    detections = yolo_detector.detect(frame)
    assert isinstance(detections, dict)
    assert all(key in detections for key in ['boxes', 'scores', 'classes'])

    # Process detections
    if len(detections['boxes']) > 0:
        for box in detections['boxes']:
            roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if roi.size > 0:  # If ROI is valid
                color = color_detector.detect_color(roi)
                assert color in ['red', 'green', 'yellow', 'unknown']

    # Draw results
    annotated_frame = draw_detections(
        frame,
        detections,
        class_names=['speed_sign', 'traffic_light']
    )
    assert annotated_frame.shape == frame.shape

def test_error_handling():
    """Test error handling in the pipeline."""
    # Test with invalid video path
    with pytest.raises(ValueError):
        VideoProcessor("nonexistent.mp4")

    # Test with invalid model path
    with pytest.raises(Exception):
        YOLODetector(model_path="nonexistent_model")

    # Test with invalid image
    color_detector = ColorDetector()
    with pytest.raises(ValueError):
        color_detector.detect_color(None)

def test_performance_basic(test_video, detection_system):
    """Basic performance test."""
    yolo_detector, _ = detection_system

    # Process 5 frames and measure time
    import time
    times = []

    for _ in range(5):
        ret, frame = test_video.read_frame()
        if not ret:
            break

        start_time = time.time()
        _ = yolo_detector.detect(frame)
        times.append(time.time() - start_time)

    # Check if average processing time is reasonable (adjust threshold as needed)
    avg_time = np.mean(times)
    assert avg_time < 5.0  # Should process a frame in less than 5 seconds

def test_system_configuration():
    """Test system configuration and settings."""
    from src.config.settings import get_config

    config = get_config()
    assert isinstance(config, dict)
    assert 'model_path' in config
    assert 'confidence_threshold' in config
    assert isinstance(config['confidence_threshold'], float)
    assert 0 <= config['confidence_threshold'] <= 1

def test_import_config():
    """Test that configuration can be imported."""
    from src.config.settings import settings
    assert settings is not None
    assert hasattr(settings, 'detection')
    assert hasattr(settings, 'models')

def test_import_detectors():
    """Test that detector classes can be imported."""
    try:
        from src.models.yolo_detector import YOLODetector
        from src.models.color_detector import ColorDetector
        # Note: We don't initialize them as they need model files
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import detector classes: {e}")

def test_import_utils():
    """Test that utility functions can be imported."""
    from src.utils.image_processing import optimal_font_scale

    # Test the function works
    scale = optimal_font_scale("Test", 100)
    assert isinstance(scale, float)
    assert scale > 0

def test_color_detector_legacy():
    """Test the legacy color detector function."""
    from src.models.color_detector import ColorDetector_legacy

    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :, 1] = 255  # Green image

    # This should not crash (though it might not detect anything)
    result = ColorDetector_legacy(test_image)
    # Result can be None or a color string
    assert result is None or isinstance(result, str)
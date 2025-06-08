"""Tests for the YOLO detector module."""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.models.yolo_detector import YOLODetector

@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    mock_model = Mock()
    mock_model.predict.return_value = [Mock(
        boxes=Mock(
            data=torch.tensor([[100, 100, 200, 200]]),
            cpu=lambda: [[100, 100, 200, 200]]
        ),
        conf=torch.tensor([0.95]),
        cls=torch.tensor([0])
    )]
    return mock_model

def test_yolo_detector_initialization():
    """Test YOLODetector initialization."""
    with patch('src.models.yolo_detector.YOLO') as mock_yolo:
        detector = YOLODetector(model_path="dummy_path")
        assert detector is not None
        mock_yolo.assert_called_once_with("dummy_path")

def test_detect_objects(mock_yolo_model, sample_image):
    """Test object detection on a sample image."""
    with patch('src.models.yolo_detector.YOLO', return_value=mock_yolo_model):
        detector = YOLODetector(model_path="dummy_path")
        results = detector.detect(sample_image)

        assert isinstance(results, dict)
        assert 'boxes' in results
        assert 'scores' in results
        assert 'classes' in results

        # Check shapes and types
        assert isinstance(results['boxes'], np.ndarray)
        assert isinstance(results['scores'], np.ndarray)
        assert isinstance(results['classes'], np.ndarray)

        # Check values
        assert len(results['boxes']) > 0
        assert len(results['scores']) > 0
        assert len(results['classes']) > 0
        assert results['scores'][0] >= 0 and results['scores'][0] <= 1

def test_empty_detection(mock_yolo_model, sample_image):
    """Test detection with no objects found."""
    mock_yolo_model.predict.return_value = [Mock(
        boxes=Mock(
            data=torch.tensor([]),
            cpu=lambda: []
        ),
        conf=torch.tensor([]),
        cls=torch.tensor([])
    )]

    with patch('src.models.yolo_detector.YOLO', return_value=mock_yolo_model):
        detector = YOLODetector(model_path="dummy_path")
        results = detector.detect(sample_image)

        assert isinstance(results, dict)
        assert len(results['boxes']) == 0
        assert len(results['scores']) == 0
        assert len(results['classes']) == 0

def test_invalid_input():
    """Test detection with invalid input."""
    with patch('src.models.yolo_detector.YOLO') as mock_yolo:
        detector = YOLODetector(model_path="dummy_path")

        with pytest.raises(ValueError):
            detector.detect(None)

        with pytest.raises(ValueError):
            detector.detect(np.array([]))  # Empty array

        with pytest.raises(ValueError):
            detector.detect(np.zeros((100, 100)))  # 2D array instead of 3D
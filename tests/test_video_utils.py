"""Tests for the video utilities module."""
import pytest
import numpy as np
import cv2
from src.utils.video_utils import VideoProcessor

@pytest.fixture
def video_processor(mock_video_path):
    """Create a VideoProcessor instance."""
    return VideoProcessor(mock_video_path)

def test_video_processor_initialization(mock_video_path):
    """Test VideoProcessor initialization."""
    processor = VideoProcessor(mock_video_path)
    assert processor is not None
    assert processor.video_path == mock_video_path
    assert processor.cap is not None
    assert processor.cap.isOpened()

def test_video_properties(video_processor):
    """Test video properties retrieval."""
    width = video_processor.get_width()
    height = video_processor.get_height()
    fps = video_processor.get_fps()

    assert width == 640
    assert height == 480
    assert fps == 30.0

def test_frame_reading(video_processor):
    """Test frame reading functionality."""
    ret, frame = video_processor.read_frame()

    assert ret is True
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)

def test_end_of_video(video_processor):
    """Test behavior at the end of video."""
    # Read all frames
    frames = []
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break
        frames.append(frame)

    assert len(frames) == 10  # As created in mock_video_path fixture

    # Try reading after end
    ret, frame = video_processor.read_frame()
    assert ret is False
    assert frame is None

def test_video_reset(video_processor):
    """Test video reset functionality."""
    # Read a few frames
    for _ in range(5):
        video_processor.read_frame()

    # Reset video
    video_processor.reset()

    # Should be able to read from start
    ret, frame = video_processor.read_frame()
    assert ret is True
    assert isinstance(frame, np.ndarray)

def test_invalid_video_path():
    """Test initialization with invalid video path."""
    with pytest.raises(ValueError):
        VideoProcessor("nonexistent_video.mp4")

def test_video_release(video_processor):
    """Test video release functionality."""
    video_processor.release()
    assert not video_processor.cap.isOpened()

    # Trying to read after release should fail gracefully
    ret, frame = video_processor.read_frame()
    assert ret is False
    assert frame is None

def test_frame_processing(video_processor):
    """Test frame processing with a simple operation."""
    ret, frame = video_processor.read_frame()
    assert ret is True

    # Test basic frame processing (e.g., resize)
    processed_frame = cv2.resize(frame, (320, 240))
    assert processed_frame.shape == (240, 320, 3)

def test_context_manager(mock_video_path):
    """Test VideoProcessor as context manager."""
    with VideoProcessor(mock_video_path) as processor:
        ret, frame = processor.read_frame()
        assert ret is True
        assert isinstance(frame, np.ndarray)

    # After context manager, video should be released
    assert not processor.cap.isOpened()
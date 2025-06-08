"""Video processing utilities for railway detection system."""

from pathlib import Path
from typing import Tuple, Optional, Generator, List
import math
import random
import cv2
import numpy as np

from src.config.settings import settings


class VideoProcessor:
    """Video processing utilities for railway detection."""

    def __init__(self, video_path: Path):
        """Initialize video processor."""
        self.video_path = Path(video_path)
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0

    def open(self):
        """Open video file and initialize properties."""
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def close(self):
        """Close video file."""
        if self.cap:
            self.cap.release()
            self.cap = None

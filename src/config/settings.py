"""Configuration settings for the railway detection system."""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class ColorRanges:
    """Color range definitions for traffic light detection."""

    # HSV color ranges for traffic lights
    red_lower: np.ndarray = np.array([136, 87, 111], np.uint8)
    red_upper: np.ndarray = np.array([180, 255, 255], np.uint8)

    green_lower: np.ndarray = np.array([52, 0, 55], np.uint8)
    green_upper: np.ndarray = np.array([104, 255, 255], np.uint8)

    yellow_lower: np.ndarray = np.array([20, 100, 100], np.uint8)
    yellow_upper: np.ndarray = np.array([30, 255, 255], np.uint8)


@dataclass
class ModelPaths:
    """Model path configurations."""

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "data" / "models"

    # YOLO models
    TRAFFIC_LIGHT_MODEL: Path = MODEL_DIR / "yolov8n.pt"
    TRAFFIC_SIGN_MODEL: Path = MODEL_DIR / "sign_yolov8_best.pt"
    CUSTOM_YOLO_MODEL: Path = BASE_DIR / "runs" / "detect" / "train8" / "weights" / "best.pt"

    # OCR model
    OCR_MODEL_DIR: Path = BASE_DIR / "models" / "en_PP-OCRv3_rec_infer"


@dataclass
class DetectionSettings:
    """Detection and inference parameters."""

    # YOLO detection parameters
    CONFIDENCE_THRESHOLD: float = 0.5
    TRAFFIC_LIGHT_CONF: float = 0.4
    TRAFFIC_SIGN_CONF: float = 0.5
    IOU_THRESHOLD: float = 0.45

    # Image processing
    IMAGE_SIZE: Tuple[int, int] = (640, 640)
    MIN_CONTOUR_AREA: int = 50
    MORPHOLOGY_KERNEL_SIZE: Tuple[int, int] = (5, 5)

    # Video processing
    DEFAULT_FPS: int = 12
    ROI_WIDTH_FACTOR: float = 0.2  # Start of ROI as fraction of frame width
    ROI_WIDTH_VARIATION: Tuple[float, float] = (0.05, 0.07)  # Random variation range


@dataclass
class SpeedSignLabels:
    """Speed sign classification labels."""

    LABELS: List[str] = None

    def __post_init__(self):
        if self.LABELS is None:
            self.LABELS = [
                '.65', '.75', '.90', '100', '105', '115', '50', '55', '60',
                '65', '70', '75', '80', '85', 'X25', 'X30', 'X40'
            ]


@dataclass
class Settings:
    """Main settings class combining all configurations."""

    colors: ColorRanges = ColorRanges()
    models: ModelPaths = ModelPaths()
    detection: DetectionSettings = DetectionSettings()
    speed_signs: SpeedSignLabels = SpeedSignLabels()

    def __post_init__(self):
        # Ensure model directories exist
        self.models.MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
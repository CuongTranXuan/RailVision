"""YOLO-based object detection for railway traffic signs and lights."""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

from src.config.settings import settings


class YOLODetector:
    """YOLO model wrapper for object detection."""

    def __init__(self, model_path: Union[str, Path], conf_threshold: float = 0.5):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Perform object detection on an image.

        Args:
            image: Input image as numpy array
            classes: List of class IDs to detect (None for all)
            conf: Confidence threshold (uses instance default if None)
            verbose: Whether to print verbose output

        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        conf = conf or self.conf_threshold

        results = self.model.predict(
            image,
            classes=classes,
            conf=conf,
            verbose=verbose
        )

        return self._process_results(results[0])

    def detect_traffic_lights(
        self,
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect traffic lights specifically (class 9 in COCO).

        Args:
            image: Input image
            conf: Confidence threshold

        Returns:
            List of traffic light detections
        """
        conf = conf or settings.detection.TRAFFIC_LIGHT_CONF
        return self.detect(image, classes=[9], conf=conf)

    def get_annotated_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get image with detection annotations.

        Args:
            image: Input image
            **kwargs: Additional arguments for detect method

        Returns:
            Annotated image
        """
        results = self.model.predict(image, **kwargs)
        return results[0].plot()

    def _process_results(self, results) -> List[Dict]:
        """
        Process YOLO results into standardized format.

        Args:
            results: YOLO results object

        Returns:
            List of detection dictionaries
        """
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2, conf, cls = box.data.tolist()[0]

                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                }
                detections.append(detection)

        return detections


class TrafficLightDetector(YOLODetector):
    """Specialized detector for traffic lights."""

    def __init__(self):
        super().__init__(
            settings.models.TRAFFIC_LIGHT_MODEL,
            settings.detection.TRAFFIC_LIGHT_CONF
        )


class TrafficSignDetector(YOLODetector):
    """Specialized detector for traffic signs."""

    def __init__(self):
        super().__init__(
            settings.models.TRAFFIC_SIGN_MODEL,
            settings.detection.TRAFFIC_SIGN_CONF
        )

    def detect_with_labels(self, image: np.ndarray) -> List[Dict]:
        """
        Detect traffic signs and include speed limit labels.

        Args:
            image: Input image

        Returns:
            List of detections with speed limit labels
        """
        detections = self.detect(image)

        for detection in detections:
            class_id = detection['class_id']
            if class_id < len(settings.speed_signs.LABELS):
                detection['label'] = settings.speed_signs.LABELS[class_id]
            else:
                detection['label'] = f"Unknown_{class_id}"

        return detections


class CustomYOLODetector(YOLODetector):
    """Custom trained YOLO detector."""

    def __init__(self):
        super().__init__(
            settings.models.CUSTOM_YOLO_MODEL,
            settings.detection.CONFIDENCE_THRESHOLD
        )
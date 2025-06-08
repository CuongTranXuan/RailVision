"""OCR processing for text recognition in traffic signs."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("Warning: PaddleOCR not available. OCR functionality will be limited.")

from src.config.settings import settings


class OCRProcessor:
    """OCR processor for text recognition in images."""

    def __init__(self, model_dir: Optional[Path] = None, use_gpu: bool = False):
        """
        Initialize OCR processor.

        Args:
            model_dir: Path to OCR model directory
            use_gpu: Whether to use GPU acceleration
        """
        self.model_dir = model_dir or settings.models.OCR_MODEL_DIR
        self.use_gpu = use_gpu
        self.ocr = None

        if PADDLEOCR_AVAILABLE:
            self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize PaddleOCR model."""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                rec_model_dir=str(self.model_dir),
                use_gpu=self.use_gpu
            )
        except Exception as e:
            print(f"Warning: Failed to initialize PaddleOCR: {e}")
            self.ocr = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply threshold for better text recognition
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Convert back to BGR for PaddleOCR
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def extract_text(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        cls: bool = True
    ) -> List[Dict]:
        """
        Extract text from image using OCR.

        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            cls: Whether to use classification

        Returns:
            List of text detection results
        """
        if not self.ocr:
            return []

        if preprocess:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image

        try:
            results = self.ocr.ocr(processed_image, cls=cls)
            return self._process_ocr_results(results)
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return []

    def extract_text_simple(self, image: np.ndarray) -> str:
        """
        Extract text as a simple string.

        Args:
            image: Input image

        Returns:
            Extracted text as string
        """
        results = self.extract_text(image)
        text_parts = []

        for result in results:
            if result.get('text'):
                text_parts.append(result['text'])

        return ' '.join(text_parts)

    def _process_ocr_results(self, results: List) -> List[Dict]:
        """
        Process OCR results into standardized format.

        Args:
            results: Raw OCR results

        Returns:
            Processed results
        """
        processed_results = []

        if not results or not results[0]:
            return processed_results

        for line in results[0]:
            if line and len(line) >= 2:
                bbox = line[0]  # Bounding box coordinates
                text_info = line[1]  # Text and confidence

                if text_info and len(text_info) >= 2:
                    text = text_info[0]
                    confidence = text_info[1]

                    result = {
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    processed_results.append(result)

        return processed_results

    def is_available(self) -> bool:
        """Check if OCR is available and properly initialized."""
        return self.ocr is not None


class SpeedSignOCR(OCRProcessor):
    """Specialized OCR for speed sign recognition."""

    def extract_speed_limit(self, image: np.ndarray) -> Optional[str]:
        """
        Extract speed limit from traffic sign.

        Args:
            image: Traffic sign image

        Returns:
            Speed limit as string or None
        """
        results = self.extract_text(image, preprocess=True, cls=False)

        # Filter for numeric results that look like speed limits
        for result in results:
            text = result.get('text', '').strip()
            confidence = result.get('confidence', 0)

            # Simple heuristic for speed limit detection
            if confidence > 0.5 and self._is_speed_limit(text):
                return text

        return None

    def _is_speed_limit(self, text: str) -> bool:
        """
        Check if text looks like a speed limit.

        Args:
            text: Text to check

        Returns:
            True if text looks like speed limit
        """
        # Remove common non-numeric characters
        cleaned = text.replace('O', '0').replace('l', '1').replace('I', '1')

        # Check if it's a reasonable speed limit number
        try:
            speed = int(cleaned)
            return 20 <= speed <= 200  # Reasonable speed range
        except ValueError:
            return False
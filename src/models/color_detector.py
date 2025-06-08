"""Color detection for traffic light recognition."""

from typing import Optional, Tuple, List
import cv2
import numpy as np

from src.config.settings import settings


class ColorDetector:
    """Color detector for traffic light recognition."""

    def __init__(self):
        """Initialize color detector with predefined color ranges."""
        self.color_ranges = settings.colors
        self.kernel = np.ones(settings.detection.MORPHOLOGY_KERNEL_SIZE, "uint8")
        self.min_area = settings.detection.MIN_CONTOUR_AREA

    def detect_color(self, image: np.ndarray) -> Optional[str]:
        """
        Detect the dominant color in an image (red, green, yellow).

        Args:
            image: Input image (BGR format)

        Returns:
            Detected color name or None
        """
        # Convert to HSV color space for better color detection
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for each color
        masks = self._create_color_masks(hsv_frame)

        # Check each color and return the first one with significant area
        for color_name, mask in masks.items():
            if self._has_significant_color_area(mask):
                return color_name

        return None

    def detect_color_with_confidence(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Detect color with confidence score.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (color_name, confidence_score)
        """
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = self._create_color_masks(hsv_frame)

        color_areas = {}
        total_pixels = image.shape[0] * image.shape[1]

        for color_name, mask in masks.items():
            # Calculate the area of detected color
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_area = sum(cv2.contourArea(contour) for contour in contours)

            # Calculate confidence as percentage of total image area
            confidence = total_area / total_pixels
            color_areas[color_name] = confidence

        # Return the color with highest confidence
        if color_areas:
            best_color = max(color_areas, key=color_areas.get)
            best_confidence = color_areas[best_color]

            # Only return if confidence is above minimum threshold
            if best_confidence > 0.01:  # 1% of image area
                return best_color, best_confidence

        return None, 0.0

    def get_color_mask(self, image: np.ndarray, color: str) -> np.ndarray:
        """
        Get mask for specific color.

        Args:
            image: Input image (BGR format)
            color: Color name ('red', 'green', 'yellow')

        Returns:
            Binary mask for the specified color
        """
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color.lower() == 'red':
            mask = cv2.inRange(hsv_frame, self.color_ranges.red_lower, self.color_ranges.red_upper)
        elif color.lower() == 'green':
            mask = cv2.inRange(hsv_frame, self.color_ranges.green_lower, self.color_ranges.green_upper)
        elif color.lower() == 'yellow':
            mask = cv2.inRange(hsv_frame, self.color_ranges.yellow_lower, self.color_ranges.yellow_upper)
        else:
            raise ValueError(f"Unsupported color: {color}")

        # Apply morphological operations to clean up the mask
        mask = cv2.dilate(mask, self.kernel)
        return mask

    def visualize_color_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Create visualization showing detected colors.

        Args:
            image: Input image

        Returns:
            Image with color detection visualization
        """
        result_image = image.copy()
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = self._create_color_masks(hsv_frame)

        # Define colors for visualization (BGR format)
        viz_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }

        for color_name, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    # Draw contour
                    cv2.drawContours(result_image, [contour], -1, viz_colors[color_name], 2)

                    # Add label
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.putText(
                        result_image,
                        color_name.upper(),
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        viz_colors[color_name],
                        2
                    )

        return result_image

    def _create_color_masks(self, hsv_image: np.ndarray) -> dict:
        """
        Create masks for all supported colors.

        Args:
            hsv_image: Image in HSV color space

        Returns:
            Dictionary mapping color names to their masks
        """
        masks = {}

        # Red mask
        red_mask = cv2.inRange(hsv_image, self.color_ranges.red_lower, self.color_ranges.red_upper)
        masks['red'] = cv2.dilate(red_mask, self.kernel)

        # Green mask
        green_mask = cv2.inRange(hsv_image, self.color_ranges.green_lower, self.color_ranges.green_upper)
        masks['green'] = cv2.dilate(green_mask, self.kernel)

        # Yellow mask
        yellow_mask = cv2.inRange(hsv_image, self.color_ranges.yellow_lower, self.color_ranges.yellow_upper)
        masks['yellow'] = cv2.dilate(yellow_mask, self.kernel)

        return masks

    def _has_significant_color_area(self, mask: np.ndarray) -> bool:
        """
        Check if mask has significant color area.

        Args:
            mask: Binary mask

        Returns:
            True if significant area is detected
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                return True

        return False


# Legacy function for backward compatibility
def ColorDetector_legacy(imageFrame: np.ndarray) -> Optional[str]:
    """
    Legacy color detection function for backward compatibility.

    Args:
        imageFrame: Input image

    Returns:
        Detected color name or None
    """
    detector = ColorDetector()
    return detector.detect_color(imageFrame)
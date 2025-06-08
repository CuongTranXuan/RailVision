"""Image processing utilities for railway detection system."""

from typing import Tuple, Optional
import cv2
import numpy as np


def optimal_font_scale(text: str, width: int, max_scale: int = 60) -> float:
    """
    Calculate optimal font scale for text to fit within given width.

    Args:
        text: Text to be displayed
        width: Available width in pixels
        max_scale: Maximum scale to try (divided by 10)

    Returns:
        Optimal font scale
    """
    for scale in reversed(range(0, max_scale, 1)):
        font_scale = scale / 10
        text_size = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=font_scale,
            thickness=1
        )
        text_width = text_size[0][0]

        if text_width <= width * 1.1:  # Allow 10% overflow
            return font_scale

    return 1.0


def calculate_thickness(image_shape: Tuple[int, ...]) -> int:
    """
    Calculate appropriate line thickness based on image size.

    Args:
        image_shape: Shape of the image (height, width, ...)

    Returns:
        Recommended thickness for drawing operations
    """
    height, width = image_shape[:2]
    return max(1, int(min(width, height) * 0.002))


def crop_region(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop region from image using bounding box.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Cropped image region
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    return image[y1:y2, x1:x2]


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio.

    Args:
        image: Input image
        target_size: Target (width, height)

    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate aspect ratios
    aspect_ratio = width / height
    target_aspect = target_width / target_height

    if aspect_ratio > target_aspect:
        # Image is wider, fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller, fit to height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized = cv2.resize(image, (new_width, new_height))

    # Create canvas with target size and center the resized image
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate position to center the image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return canvas


def enhance_image_for_detection(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better detection results.

    Args:
        image: Input image

    Returns:
        Enhanced image
    """
    # Convert to LAB color space for better luminance processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image specifically for OCR.

    Args:
        image: Input image

    Returns:
        Preprocessed image optimized for OCR
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive threshold for better text extraction
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Convert back to BGR for consistency
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)


def create_roi_mask(image_shape: Tuple[int, int], roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Create a binary mask for region of interest.

    Args:
        image_shape: Shape of the image (height, width)
        roi_coords: ROI coordinates (x1, y1, x2, y2)

    Returns:
        Binary mask with ROI area set to 255
    """
    height, width = image_shape
    x1, y1, x2, y2 = roi_coords

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    return mask


def apply_roi_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply ROI mask to image.

    Args:
        image: Input image
        mask: Binary mask

    Returns:
        Masked image
    """
    if len(image.shape) == 3:
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(image, mask_3d)
    else:
        return cv2.bitwise_and(image, mask)


def draw_detection_box(
    image: np.ndarray,
    bbox: list,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: Optional[int] = None
) -> np.ndarray:
    """
    Draw detection bounding box with label on image.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Text label
        confidence: Detection confidence
        color: Box color in BGR format
        thickness: Line thickness (auto-calculated if None)

    Returns:
        Image with detection box drawn
    """
    if thickness is None:
        thickness = calculate_thickness(image.shape)

    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label text
    text = f"{label} {confidence:.2f}" if confidence is not None else label

    # Calculate optimal font size
    box_width = x2 - x1
    font_scale = optimal_font_scale(text, box_width)

    # Draw label background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    cv2.rectangle(
        image,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0], y1),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        image,
        text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (255, 255, 255),  # White text
        thickness
    )

    return image


# Legacy function for backward compatibility
def OptimalFontScale(text: str, width: int) -> float:
    """
    Legacy function for backward compatibility.

    Args:
        text: Text to be displayed
        width: Available width

    Returns:
        Optimal font scale
    """
    return optimal_font_scale(text, width)
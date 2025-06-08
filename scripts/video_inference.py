#!/usr/bin/env python3
"""Main video inference script for railway traffic detection."""

import argparse
import math
import random
from pathlib import Path
import cv2

from src.models.yolo_detector import TrafficLightDetector, TrafficSignDetector
from src.models.color_detector import ColorDetector
from src.utils.image_processing import optimal_font_scale, calculate_thickness
from src.config.settings import settings


def main():
    """Main function for video inference."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Railway Traffic Detection')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--save', nargs='?', help='Save processed video to specified path')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')

    # Parse arguments
    args = parser.parse_args()

    # Initialize detectors
    print("Initializing detectors...")
    try:
        light_detector = TrafficLightDetector()
        sign_detector = TrafficSignDetector()
        color_detector = ColorDetector()
        print("✓ All detectors initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing detectors: {e}")
        return

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file: {args.video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"✓ Video opened: {width}x{height} @ {fps:.1f} FPS")

    # Calculate thickness for drawing
    thickness = calculate_thickness((height, width))

    # Setup video writer if saving
    video_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.save, fourcc, 12, (width, height))
        print(f"✓ Video writer initialized: {args.save}")

    frame_count = 0

    try:
        print("Starting video processing... Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Define ROI with random variation
            roi_start = int(width * settings.detection.ROI_WIDTH_FACTOR)
            variation = random.uniform(*settings.detection.ROI_WIDTH_VARIATION)
            roi_end = math.floor(width * (0.5 + variation))

            # Extract ROI
            roi = frame[0:height, roi_start:roi_end]

            # Traffic light detection
            light_detections = light_detector.detect_traffic_lights(roi)

            for detection in light_detections:
                # Adjust coordinates for full frame
                bbox = detection['bbox']
                bbox[0] += roi_start  # Adjust x1
                bbox[2] += roi_start  # Adjust x2

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),  # Green
                    thickness
                )

                # Detect color
                light_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                detected_color = color_detector.detect_color(light_crop)

                if detected_color:
                    # Calculate font size
                    box_width = bbox[2] - bbox[0]
                    font_size = optimal_font_scale(detected_color, box_width)

                    # Draw color label
                    cv2.putText(
                        frame,
                        detected_color.upper(),
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size,
                        (0, 255, 0),
                        thickness
                    )

            # Traffic sign detection
            sign_detections = sign_detector.detect_with_labels(roi)

            for detection in sign_detections:
                # Adjust coordinates for full frame
                bbox = detection['bbox']
                bbox[0] += roi_start  # Adjust x1
                bbox[2] += roi_start  # Adjust x2

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 255),  # Yellow
                    thickness
                )

                # Draw speed limit label
                if 'label' in detection:
                    box_width = bbox[2] - bbox[0]
                    font_size = optimal_font_scale(detection['label'], box_width)

                    cv2.putText(
                        frame,
                        detection['label'],
                        (bbox[2], bbox[3]),
                        cv2.FONT_HERSHEY_DUPLEX,
                        font_size * 2.2,
                        (255, 0, 255),
                        thickness
                    )

            # Draw ROI boundary
            cv2.rectangle(
                frame,
                (roi_start, 0),
                (roi_end, height),
                (0, 0, 255),  # Red
                thickness
            )

            # Display frame
            if not args.no_display:
                cv2.imshow("Railway Traffic Detection", frame)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save frame if writer is available
            if video_writer:
                video_writer.write(frame)

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("\n✓ Processing interrupted by user")

    except Exception as e:
        print(f"✗ Error during processing: {e}")

    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print(f"✓ Processing completed. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
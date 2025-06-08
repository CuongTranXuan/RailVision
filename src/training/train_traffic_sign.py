#!/usr/bin/env python3
"""Training script for traffic sign detection."""

import argparse
from pathlib import Path
from ultralytics import YOLO

from src.config.settings import settings


class TrafficSignTrainer:
    """Trainer for traffic sign detection models."""

    def __init__(self, base_model_path: str = None):
        """
        Initialize trainer.

        Args:
            base_model_path: Path to base YOLO model
        """
        self.base_model_path = base_model_path or str(settings.models.TRAFFIC_LIGHT_MODEL)
        self.model = None

    def load_model(self):
        """Load YOLO model for training."""
        print(f"Loading base model: {self.base_model_path}")
        self.model = YOLO(self.base_model_path)
        print("✓ Model loaded successfully")

    def train(
        self,
        data_config: str,
        epochs: int = 20,
        batch_size: int = 12,
        image_size: int = 640,
        save_period: int = 5,
        **kwargs
    ):
        """
        Train the traffic sign detection model.

        Args:
            data_config: Path to data configuration YAML file
            epochs: Number of training epochs
            batch_size: Training batch size
            image_size: Training image size
            save_period: Save model every N epochs
            **kwargs: Additional training parameters
        """
        if not self.model:
            self.load_model()

        print(f"Starting training with:")
        print(f"  Data config: {data_config}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}")

        # Train the model
        results = self.model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            save=True,
            save_period=save_period,
            **kwargs
        )

        print("✓ Training completed")
        return results

    def validate(self):
        """Validate the trained model."""
        if not self.model:
            raise ValueError("Model not loaded. Train or load a model first.")

        print("Starting validation...")
        results = self.model.val()
        print("✓ Validation completed")
        return results

    def test_inference(self, test_image: str):
        """
        Test inference on a single image.

        Args:
            test_image: Path to test image
        """
        if not self.model:
            raise ValueError("Model not loaded. Train or load a model first.")

        print(f"Testing inference on: {test_image}")
        results = self.model(test_image)
        print("✓ Inference test completed")
        return results

    def export_model(self, format: str = 'onnx'):
        """
        Export model to specified format.

        Args:
            format: Export format (onnx, tflite, etc.)
        """
        if not self.model:
            raise ValueError("Model not loaded. Train or load a model first.")

        print(f"Exporting model to {format} format...")
        success = self.model.export(format=format)

        if success:
            print(f"✓ Model exported successfully to {format}")
        else:
            print(f"✗ Failed to export model to {format}")

        return success


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Traffic Sign Detection Model')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data configuration YAML file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=12,
        help='Training batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Training image size'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        help='Path to base YOLO model'
    )
    parser.add_argument(
        '--test-image',
        type=str,
        help='Path to test image for inference validation'
    )
    parser.add_argument(
        '--export',
        type=str,
        choices=['onnx', 'tflite', 'tensorrt'],
        help='Export model to specified format after training'
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = TrafficSignTrainer(args.base_model)

    try:
        # Train model
        trainer.train(
            data_config=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz
        )

        # Validate model
        trainer.validate()

        # Test inference if test image provided
        if args.test_image:
            trainer.test_inference(args.test_image)

        # Export model if format specified
        if args.export:
            trainer.export_model(args.export)

    except Exception as e:
        print(f"✗ Training failed: {e}")
        return 1

    print("✓ Training pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
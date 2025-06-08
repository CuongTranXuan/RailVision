# Railway Traffic Detection System (DemoAusrail2023)

A modern Python-based system for detecting traffic lights and speed signs on railways using computer vision and deep learning, optimized for Jetson devices.

## 🏗️ Project Structure

```
DemoAusrail2023/
├── src/                          # Main source code
│   ├── models/                   # Detection models
│   │   ├── yolo_detector.py     # YOLO-based object detection
│   │   ├── ocr_processor.py     # OCR text recognition
│   │   └── color_detector.py    # Traffic light color detection
│   ├── utils/                   # Utility functions
│   │   ├── image_processing.py  # Image processing utilities
│   │   └── video_utils.py       # Video processing utilities
│   ├── config/                  # Configuration management
│   │   └── settings.py          # Centralized settings
│   └── training/                # Training scripts
│       └── train_traffic_sign.py # Traffic sign training
├── scripts/                     # Executable scripts
│   ├── video_inference.py       # Main inference script
│   ├── convert_tflite.py        # Model conversion utilities
│   └── convert_xml_yolo.py      # Data format conversion
├── tests/                       # Unit tests
├── data/                        # Data directory
│   ├── raw/                     # Original datasets
│   ├── processed/               # Processed data
│   └── models/                  # Trained model weights
├── requirements/                # Dependency management
│   ├── base.txt                # Base requirements
│   ├── dev.txt                 # Development requirements
│   └── jetson.txt              # Jetson-specific notes
├── notebooks/                   # Jupyter notebooks for analysis
├── docs/                        # Documentation
├── pyproject.toml              # Modern Python project configuration
└── README.md
```

## 🚀 Features

- **Multi-modal Detection**: Combines YOLO object detection, color recognition, and OCR
- **Traffic Light Detection**: Detects and classifies traffic light colors (red, green, yellow)
- **Speed Sign Recognition**: Identifies and reads speed limit signs
- **Real-time Processing**: Optimized for real-time video processing
- **Jetson Optimized**: Designed for edge deployment on NVIDIA Jetson devices
- **Modular Architecture**: Clean, maintainable, and extensible codebase
- **Modern Python**: Uses type hints, dataclasses, and modern Python features

## 📦 Installation

### Using uv (Recommended)

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
# Install base dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"

# Install OCR support (optional)
uv pip install -e ".[ocr]"
```

### Using pip (Alternative)

If you prefer using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

pip install -r requirements/base.txt
pip install -r requirements/dev.txt  # Optional: for development
```

### Jetson Installation

For Jetson devices, follow the specific installation guide in `requirements/jetson.txt`:

```bash
# 1. Install JetPack 5.0.2
sudo apt update && sudo apt upgrade
sudo apt install nvidia-jetpack python3.8-venv

# 2. Install PyTorch for Jetson (see requirements/jetson.txt for details)
# 3. Build TorchVision from source
# 4. Install project requirements
pip install -r requirements/jetson.txt
```

### Development Installation

```bash
# Install development dependencies
pip install -r requirements/dev.txt

# Run tests
pytest

# Code formatting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Type checking
mypy src/
```

## 🎯 Usage

### Video Inference

Process a video file with traffic detection:

```bash
python scripts/video_inference.py path/to/video.mp4

# Save output video
python scripts/video_inference.py path/to/video.mp4 --save output.mp4

# Run without display (useful for headless systems)
python scripts/video_inference.py path/to/video.mp4 --no-display
```

### Training

Train a custom traffic sign detection model:

```bash
python src/training/train_traffic_sign.py \
    --data datasets/speed.v1i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16 \
    --test-image images/test.jpg
```

### Using as a Library

```python
from src.models.yolo_detector import TrafficLightDetector, TrafficSignDetector
from src.models.color_detector import ColorDetector
import cv2

# Initialize detectors
light_detector = TrafficLightDetector()
sign_detector = TrafficSignDetector()
color_detector = ColorDetector()

# Process an image
image = cv2.imread("test_image.jpg")
light_detections = light_detector.detect_traffic_lights(image)
sign_detections = sign_detector.detect_with_labels(image)

# Detect colors in traffic lights
for detection in light_detections:
    bbox = detection['bbox']
    light_crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    color = color_detector.detect_color(light_crop)
    print(f"Traffic light color: {color}")
```

## ⚙️ Configuration

All configuration is centralized in `src/config/settings.py`. Key settings include:

- **Model paths**: Locations of YOLO and OCR models
- **Detection parameters**: Confidence thresholds, image sizes
- **Color ranges**: HSV ranges for traffic light colors
- **Processing parameters**: ROI settings, morphology parameters

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_basic.py
```

## 📊 Model Performance

The system uses multiple specialized models:

- **Traffic Lights**: YOLOv8n trained on COCO dataset (class 9)
- **Traffic Signs**: Custom YOLOv8 model trained on speed sign dataset
- **Color Detection**: HSV-based color classification
- **OCR**: PaddleOCR for text recognition (optional)

## 🤝 Development

### Code Style

The project follows modern Python best practices:

- **Type hints** for better code clarity
- **Dataclasses** for configuration management
- **Black** for code formatting
- **isort** for import sorting
- **Pytest** for testing

### Adding New Features

1. Create feature branch
2. Add your module in appropriate `src/` subdirectory
3. Write tests in `tests/`
4. Update configuration if needed
5. Add documentation
6. Submit pull request

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**: Ensure model weights are in the correct directory
2. **CUDA/GPU issues**: Check CUDA installation and compatibility
3. **PaddleOCR on Jetson**: OCR may not work on Jetson; system will gracefully degrade
4. **Permission errors**: Ensure proper file permissions for model directories

### Performance Optimization

- Use appropriate batch sizes for your hardware
- Adjust confidence thresholds for speed vs accuracy trade-off
- Consider model pruning for edge deployment
- Use TensorRT for optimized inference on Jetson

## 📄 License

[Specify your license here]

## 👥 Contributors

[Add contributor information]

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- PaddlePaddle for PaddleOCR
- OpenCV community
- NVIDIA for Jetson platform support

## Development Workflow

### Git Setup

The project uses Git for version control with pre-commit hooks to ensure code quality. The following checks are performed before each commit:

- Code formatting (ruff)
- Type checking (mypy)
- Code quality checks (various pre-commit hooks)
- Unit tests (pytest)

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd DemoAusrail2023
```

2. Install dependencies with uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install the project and development dependencies:
```bash
uv pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
uv pip install pre-commit
pre-commit install
```

### Pre-commit Hooks

The project uses several pre-commit hooks to maintain code quality:

1. **Code Formatting**
   - Ruff for fast Python linting and formatting
   - Trailing whitespace removal
   - End of file fixing
   - Debug statement checks

2. **Code Quality**
   - MyPy for static type checking
   - AST checking
   - Large file checks
   - Merge conflict detection
   - Private key detection

3. **Testing**
   - Pytest runs with coverage reporting
   - All tests must pass before commit

To manually run all pre-commit hooks:
```bash
pre-commit run --all-files
```

To skip pre-commit hooks in emergency situations:
```bash
git commit -m "Your message" --no-verify
```

### Commit Guidelines

1. Write clear, descriptive commit messages
2. Use present tense ("Add feature" not "Added feature")
3. First line should be 50 characters or less
4. Reference issues and pull requests liberally

Example commit message:
```
Add traffic light color detection

- Implement HSV color space conversion
- Add red/yellow/green detection thresholds
- Include unit tests for color detection
- Update documentation with color detection details

Fixes #123
```

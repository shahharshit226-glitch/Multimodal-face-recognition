# Multimodal Biometric System - Ear and Profile Face Recognition

A deep learning-based multimodal biometric authentication system that combines ear and profile face recognition for enhanced person identification accuracy.

## Features

- **Dual Modality Recognition**: Combines ear and profile face biometrics
- **Deep Learning Models**: CNN-based feature extraction using PyTorch
- **Multiple Fusion Methods**: Score-level and feature-level fusion
- **Automatic Detection**: MediaPipe and contour-based detection for ears and faces
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Flexible Architecture**: Easily extensible for additional biometric modalities

## System Architecture

### Components

1. **Ear Recognition Module**
   - Custom CNN architecture optimized for ear features
   - Contour-based ear detection
   - CLAHE enhancement for better feature extraction

2. **Face Recognition Module**
   - ResNet50-based backbone with pretrained weights
   - MediaPipe face detection
   - Profile face specialization

3. **Multimodal Fusion**
   - Score-level fusion with weighted combination
   - Feature-level fusion with learned weights
   - Multiple fusion strategies (weighted sum, max, product)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository or navigate to the project directory:
```bash
cd "D:\mini project"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset in the following structure:

```
data/
├── raw/
│   ├── ear/
│   │   ├── person_001/
│   │   │   ├── ear_001.jpg
│   │   │   ├── ear_002.jpg
│   │   │   └── ...
│   │   ├── person_002/
│   │   └── ...
│   └── face/
│       ├── person_001/
│       │   ├── face_001.jpg
│       │   ├── face_002.jpg
│       │   └── ...
│       ├── person_002/
│       └── ...
└── processed/
```

**Notes:**
- Each person should have a unique ID folder
- Ear and face images should be paired (same person ID in both directories)
- Supported formats: JPG, JPEG, PNG
- For profile images, both ear and face can be extracted from the same image

## Configuration

Edit `config.py` to customize:

- **Image sizes**: Adjust input dimensions for ear and face
- **Training parameters**: Batch size, learning rate, epochs
- **Fusion weights**: Balance between ear and face contributions
- **Model architecture**: Embedding dimensions, number of classes

## Usage

### Training

Train both ear and face models:

```bash
python train.py
```

The script will:
- Load and preprocess the dataset
- Split into train/validation/test sets
- Train both models with early stopping
- Save best checkpoints to `checkpoints/`

### Inference

Run inference on new images:

**Option 1: Using a profile image (contains both ear and face)**
```bash
python inference.py --profile path/to/profile_image.jpg --num_classes 100
```

**Option 2: Using separate ear and face images**
```bash
python inference.py --ear path/to/ear.jpg --face path/to/face.jpg --num_classes 100
```

**Option 3: Using GPU**
```bash
python inference.py --profile path/to/profile.jpg --device cuda --num_classes 100
```

### Output

The system provides three predictions:
- **Ear Model**: Individual ear recognition result
- **Face Model**: Individual face recognition result
- **Multimodal System**: Fused prediction combining both modalities

Example output:
```
============================================================
BIOMETRIC RECOGNITION RESULTS
============================================================
Ear Model:
  Prediction: Person 42
  Confidence: 0.8234

Face Model:
  Prediction: Person 42
  Confidence: 0.9156

Multimodal System:
  Prediction: Person 42
  Confidence: 0.8812
============================================================
```

## Project Structure

```
mini project/
├── config.py                 # Configuration parameters
├── train.py                  # Training script
├── inference.py              # Inference script
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── ear_model.py         # Ear CNN models
│   ├── face_model.py        # Face CNN models
│   └── fusion.py            # Fusion modules
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py     # Image preprocessing
│   └── dataset.py           # Dataset loader
├── data/                     # Dataset directory
│   ├── raw/
│   │   ├── ear/
│   │   └── face/
│   └── processed/
└── checkpoints/              # Saved model checkpoints
    ├── ear_model_best.pth
    └── face_model_best.pth
```

## Model Details

### Ear Recognition Model (EarCNN)

- **Input**: 128x128 RGB images
- **Architecture**: 4 convolutional blocks with batch normalization
- **Features**: 512-dimensional embedding
- **Output**: Person identity classification

### Face Recognition Model (FaceCNN)

- **Input**: 224x224 RGB images
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Features**: 512-dimensional embedding
- **Output**: Person identity classification

### Fusion Strategies

1. **Score-Level Fusion** (Default)
   - Weighted sum: `Score = w_ear * P_ear + w_face * P_face`
   - Product rule: `Score = P_ear^w_ear * P_face^w_face`
   - Max rule: `Score = max(P_ear, P_face)`

2. **Feature-Level Fusion**
   - Concatenates embeddings from both models
   - Learns optimal combination through training

## Performance Optimization

### For Better Accuracy:

1. **Increase dataset size**: More training samples per person
2. **Adjust fusion weights**: Tune `EAR_WEIGHT` and `FACE_WEIGHT` in config
3. **Data augmentation**: Enable and customize in `utils/dataset.py`
4. **Fine-tuning**: Train for more epochs with lower learning rate

### For Faster Training:

1. **Use GPU**: Set device to 'cuda' in config
2. **Reduce batch size**: Lower memory usage
3. **Use smaller models**: Switch to EarResNet or SimpleFaceCNN
4. **Reduce image size**: Lower resolution inputs

## Troubleshooting

### "Dataset is empty"
- Ensure data is in correct directory structure
- Check file extensions (.jpg, .jpeg, .png)
- Verify person IDs match in both ear and face directories

### "Could not detect ear/face"
- Ensure input images show clear profile view
- Adjust detection thresholds in `utils/preprocessing.py`
- Try using pre-cropped images

### Out of Memory Errors
- Reduce batch size in `config.py`
- Lower image resolution
- Use CPU instead of GPU for inference

## Future Enhancements

- [ ] Add more biometric modalities (iris, fingerprint)
- [ ] Implement attention mechanisms
- [ ] Add real-time webcam recognition
- [ ] Support for video-based recognition
- [ ] Mobile deployment optimization
- [ ] Database enrollment system
- [ ] Anti-spoofing measures

## References

- MediaPipe Face Detection: https://google.github.io/mediapipe/
- PyTorch Deep Learning Framework: https://pytorch.org/
- ResNet Architecture: He et al., "Deep Residual Learning for Image Recognition"

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This system is designed for research and educational purposes. For production deployment in security-critical applications, additional validation, testing, and security measures are required.

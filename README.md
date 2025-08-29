# Medical Image Classification

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.0+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> An AI-powered system for classifying medical images into key organ categories using deep learning and transfer learning techniques.
<img src="https://github.com/abdullah-gamil05/ImageBasedAnatomy_Tasks/blob/main/assets/prediction1.png" alt="Result 1" width="600" />

## 🎯 Overview

This project implements a robust AI model capable of classifying medical images into categories of key organs including **breast**, **brain**, **kidney**, and **lungs**. The system leverages transfer learning with VGG16 architecture and provides a user-friendly GUI for seamless interaction.

The project Uses transfer learning with VGG16, the model achieves robust performance with an organized dataset and effective preprocessing.

### Key Highlights
- 🧠 **Deep Learning**: Utilizes VGG16 pre-trained model with transfer learning
- 📊 **High Accuracy**: Robust performance with comprehensive evaluation metrics
- 🖥️ **User-Friendly GUI**: Intuitive interface for image upload and prediction
- 📈 **Scalable Architecture**: Designed for easy expansion to additional organ categories.

## 🚀 Features

### 🤖 Model Training & Evaluation
- **Pre-trained Architecture**: VGG16 model fine-tuned for medical image classification
- **Transfer Learning**: Leverages ImageNet weights for enhanced feature extraction
- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score evaluation
- **Visualization Tools**: Training progress monitoring and performance charts

### 🎨 GUI Interface
- **Drag & Drop Upload**: Easy image upload functionality
- **Real-time Predictions**: Instant classification with confidence scores
- **Results Display**: Clear visualization of predicted organ categories
- **User Experience**: Intuitive and responsive design.

### 📊 Performance & Scalability
- **Robust Classification**: Handles various medical image formats
- **Extensible Design**: Easy addition of new organ categories
- **Batch Processing**: Support for multiple image analysis

## 📁 Project Structure

```
medical-image-classification/
├── 📁 data/
│   ├── 📁 raw/                 # Raw medical images
│   ├── 📁 processed/           # Preprocessed datasets
│   └── 📁 splits/              # Train/validation/test splits
├── 📁 models/
│   ├── 📁 saved_models/        # Trained model files
│   ├── 📁 checkpoints/         # Training checkpoints
│   └── model_architecture.py   # Model definition
├── 📁 src/
│   ├── 📁 data/                # Data processing utilities
│   ├── 📁 models/              # Model training/evaluation
│   ├── 📁 visualization/       # Plotting and visualization
│   └── 📁 gui/                 # GUI implementation
├── 📁 notebooks/               # Jupyter notebooks
├── 📁 scripts/                 # Utility scripts
├── 📁 tests/                   # Unit tests
├── 📁 docs/                    # Documentation
├── requirements.txt
├── setup.py
└── README.md
```

## Application Interface

### 1. GUI Components
- **Upload Button**: Load a medical image for classification.
- **Prediction Output**: Displays the predicted organ category with confidence.

### 2. Training Process
- Command-line-based interface for monitoring model training progress.
- Generates performance metrics and visualizations of model accuracy/loss.

## 🧠 Technical Details

### Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet
- **Adaptation**: Custom classification head for medical images
- **Input Processing**: Grayscale to RGB conversion for VGG16 compatibility
- **Output**: Multi-class classification (brain, breast, kidney, lungs)

### Dataset Information
```
| Organ Category | Training Samples | Validation Samples | Test Samples |
|----------------|------------------|-------------------|--------------|
| Brain          | 2,500            | 500               | 300          |
| Breast         | 2,200            | 450               | 280          |
| Kidney         | 1,800            | 400               | 250          |
| Lungs          | 2,100            | 420               | 270          |
```
### Training Pipeline

- **Image Preprocessing**: Resizing (224×224), normalization, data augmentation
- **Optimizer**: Adam with learning rate scheduling (initial: 1e-4)
- **Loss Function**: Categorical cross-entropy
- **Regularization**: Dropout (0.5) and early stopping
- **Training Time**: ~2-3 hours on GPU

### Performance Metrics

```
Overall Accuracy: 94.2%
Per-class Performance:
├── Brain:  Precision: 95.1%, Recall: 93.8%, F1: 94.4%
├── Breast: Precision: 92.9%, Recall: 94.7%, F1: 93.8%
├── Kidney: Precision: 93.8%, Recall: 92.1%, F1: 92.9%
└── Lungs:  Precision: 95.2%, Recall: 96.1%, F1: 95.6%
```

## 🔬 Data Sources

This project utilizes publicly available medical imaging datasets:

- **Brain Tumor MRI Dataset**: High-resolution brain MRI scans
- **Breast Cancer MRI Dataset**: Breast tissue imaging data
- **Kidney Function Dataset**: Kidney-related medical images
- **Lung X-ray Dataset**: Chest X-ray images for lung classification

## 🙏 Acknowledgments

- **VGG Team**: For the foundational VGG16 architecture
- **TensorFlow/Keras**: For the deep learning framework
- **Medical Imaging Community**: For providing open datasets
- **Contributors**: All developers who contributed to this project




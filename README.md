# Final-Project
# Multi-Dataset Image Classification with Classic and Modern CNNs

## Overview

This project focuses on building a robust deep learning image classification system capable of classifying images from multiple datasets using both classic and modern Convolutional Neural Networks (CNNs). The project compares the performance of several state-of-the-art CNN architectures through transfer learning and fine-tuning techniques.

The implementation is developed using **PyTorch** and provides a modular, scalable, and production-ready deep learning pipeline for image classification.

---

# Problem Statement

Develop a multi-class image classification system capable of recognizing images from three different datasets:

* Animals Dataset
* Butterflies Dataset
* ImageNet-10 Dataset

The project evaluates multiple CNN architectures and compares their performance based on accuracy, robustness, calibration, computational efficiency, and inference speed.

---

# Business Use Cases

* Wildlife Monitoring and Species Identification
* Biodiversity Conservation
* Image Auto-Tagging Systems
* Educational Applications
* Citizen Science Platforms
* Visual Search Engines
* Edge AI Image Classification
* Mobile Image Recognition Applications

---

# Features

* Multi-Dataset Image Classification
* Transfer Learning using ImageNet Pretrained Weights
* Fine-Tuning Strategies
* Multiple CNN Model Comparison
* Automatic Dataset Loading
* Data Augmentation Pipeline
* Mixed Precision Training (AMP)
* Early Stopping
* Learning Rate Scheduling
* Model Checkpointing
* Confusion Matrix Generation
* Expected Calibration Error (ECE)
* Reliability Diagram
* Model Performance Comparison
* Streamlit Web Application for Inference

---

# Project Structure

```
Multi_Dataset_Image_Classification/

│── configs/
│── datasets/
│── models/
│── trainer/
│── utils/
│── saved_models/
│── logs/
│── results/

│── train.py
│── evaluate.py
│── inference.py
│── app.py
│── requirements.txt
│── README.md
```

---

# Datasets

## Animals Dataset

* Multi-class animal image dataset
* RGB Images
* Folder Structure

```
Animals/

├── Cat/
├── Dog/
├── Elephant/
├── Horse/
└── ...
```

---

## Butterflies Dataset

Contains butterfly species images with varying backgrounds.

```
Butterflies/

├── Class1/
├── Class2/
├── Class3/
└── ...
```

---

## ImageNet-10 Dataset

Subset of ImageNet containing 10 image classes.

```
ImageNet10/

├── Class1/
├── Class2/
└── ...
```

---

# Data Preprocessing

The preprocessing pipeline includes:

* Resize images
* Random Resized Crop
* Center Crop
* Horizontal Flip
* Color Jitter
* Image Normalization
* ImageNet Mean & Standard Deviation
* Corrupted Image Removal
* RGB Conversion

---

# Data Augmentation

Training augmentation includes:

* Random Horizontal Flip
* Random Resized Crop
* Color Jitter
* Random Rotation
* Optional MixUp
* Optional CutMix
* Optional RandAugment

---

# CNN Models Implemented

### Classic CNNs

* AlexNet
* VGG16
* VGG19

### Residual Networks

* ResNet18
* ResNet34

### Attention-Based CNN

* SE-ResNet

### Modern CNN

* ConvNeXt-Tiny

---

# Transfer Learning Strategy

The project uses ImageNet pretrained weights whenever available.

### Phase 1

* Freeze backbone
* Train classifier head

### Phase 2

* Unfreeze last layers
* Fine-tune

### Phase 3

* Unfreeze complete network
* Train with lower learning rate

---

# Training Configuration

| Parameter       | Value            |
| --------------- | ---------------- |
| Image Size      | 224 × 224        |
| Batch Size      | 32 / 64          |
| Epochs          | 30               |
| Optimizer       | AdamW            |
| Learning Rate   | 1e-4             |
| Weight Decay    | 1e-4             |
| Scheduler       | Cosine Annealing |
| Loss            | Cross Entropy    |
| Label Smoothing | 0.1              |
| Mixed Precision | Yes              |

---

# Evaluation Metrics

## Classification Metrics

* Top-1 Accuracy
* Macro F1 Score
* Precision
* Recall
* Confusion Matrix

## Calibration Metrics

* Expected Calibration Error (ECE)
* Maximum Calibration Error (MCE)
* Reliability Diagram

## Efficiency Metrics

* Model Parameters
* FLOPs
* CPU Latency
* GPU Throughput

---

# Training Pipeline

1. Load Dataset
2. Apply Transformations
3. Build DataLoader
4. Initialize CNN Model
5. Load Pretrained Weights
6. Freeze Backbone
7. Train Classifier
8. Fine-Tune Model
9. Validate
10. Save Best Checkpoint
11. Evaluate on Test Set

---

# Model Comparison

The project compares all implemented CNN architectures based on:

* Accuracy
* Macro F1 Score
* Precision
* Recall
* Calibration Error
* Training Time
* Inference Time
* Model Size
* FLOPs

---

# Streamlit Application

The project includes an interactive Streamlit application that allows users to:

* Upload an image
* Select a trained CNN model
* Predict the image class
* Display prediction confidence
* View top predicted classes

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Multi_Dataset_Image_Classification.git

cd Multi_Dataset_Image_Classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python train.py
```

---

# Evaluate the Model

```bash
python evaluate.py
```

---

# Run Inference

```bash
python inference.py
```

---

# Launch the Streamlit Application

```bash
streamlit run app.py
```

---

# Expected Results

The project produces:

* Trained Model Checkpoints
* Training Logs
* Accuracy Curves
* Loss Curves
* Confusion Matrices
* Reliability Diagrams
* Performance Comparison Tables
* Model Evaluation Reports

---

# Technologies Used

* Python
* PyTorch
* TorchVision
* TIMM
* NumPy
* Pandas
* Scikit-learn
* OpenCV
* Matplotlib
* Plotly
* Streamlit
* TensorBoard

---

# Future Improvements

* Vision Transformers (ViT)
* EfficientNet Models
* MobileNetV3
* Knowledge Distillation
* ONNX Export
* TensorRT Optimization
* Docker Deployment
* REST API Integration
* Edge Device Deployment

---

# License

This project is intended for educational and research purposes.

---

# Acknowledgements

* PyTorch
* TorchVision
* TIMM
* ImageNet
* Animals Dataset Contributors
* Butterflies Dataset Contributors
* Open Source Community

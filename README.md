# BreastMNIST - Automated Classification AI

## Overview
This repository contains a deep learning project focused on the binary classification of breast ultrasound images (benign vs. malignant) using the BreastMNIST dataset. The objective is to improve upon baseline models for medical image analysis using a customized convolutional neural network.

## Architecture Modifications
The model is based on the **ResNet-18** architecture, with specific modifications to handle 28x28 grayscale ultrasound images:
- Replaced the initial 7x7 convolutional layer with a smaller 3x3 kernel (stride=1, padding=1) to preserve fine-grained spatial resolution.
- Removed the initial max-pooling layer entirely to prevent aggressive downsampling of the small input images.
- Adapted the final fully connected layer to output two classes using the 512-unit feature extractor.

## Performance
The modified architecture successfully outperformed the baseline ResNet-18 model across key evaluation metrics:
- **AUC-ROC**: 0.915 (Baseline: 0.901)
- **Accuracy**: 87.2% (Baseline: 86.3%)
- **AUPR**: 0.908
- **Recall**: 90.1%

## Repository Structure
- `AI_CourseWork_2025.ipynb`: Main Jupyter notebook containing data loading, preprocessing, model definition, training loop, and evaluation.
- `docs/Report-AI-CW.pdf`: Comprehensive academic project report detailing the methodology, visual performance analysis (ROC & Precision-Recall curves), and cross-validation challenges.

## Requirements
To run the notebook, install the following dependencies:
```bash
pip install torch torchvision medmnist tqdm scikit-learn
```

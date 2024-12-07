# Image Forgery Detection

## Project Overview

This project implements an advanced image forgery detection system using Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs) to identify manipulated images.

## Dataset

- **Source**: National Laboratory of Pattern Recognition (NLPR) CASIA Dataset
- **Total Images**: 5,123
  - Authentic Images: 3,683
  - Spliced Images: 1,440

## Methodology

### Preprocessing
- Error Level Analysis (ELA) technique used to convert images
- Image resizing to 128x128 pixels
- Normalized pixel values

### Model Architectures
1. CNN with Adam Optimizer
2. CNN with RMSProp Optimizer
3. Random Forest Classifier

## Key Features
- Detects image splicing and tampering
- Uses deep learning for robust feature extraction
- Supports various image types and scenarios

## Performance Metrics

### Model Accuracies
- CNN (Adam Optimizer): 94.05%
- CNN (RMSProp Optimizer): 92.36%
- Random Forest: 89.21%

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn

## Author
JUMANA.B

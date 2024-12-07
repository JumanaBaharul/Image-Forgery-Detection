# Image Forgery Detection

## Project Overview

This project implements an advanced image forgery detection system using Error Level Analysis (ELA) and Convolutional Neural Networks (CNNs) to identify manipulated images with high accuracy.

## Dataset

- **Source**: National Laboratory of Pattern Recognition (NLPR) CASIA Dataset
- **Total Images**: 5,123
  - Authentic Images: 3,683
  - Spliced Images: 1,440

## Dataset Characteristics

- Wide range of image types
- Covers various challenging scenarios:
  - Object insertion
  - Background replacement
  - Composite scenes
- Ground truth masks for each image
- Carefully crafted tampered images to simulate realistic forgeries

## Model Architectures

### 1. Convolutional Neural Network with Adam Optimizer

#### Architecture Details:
- Input Layer: 128x128x3 (image dimensions)
- Convolutional Layers:
  - 1st Block: 
    - 64 filters, 5x5 kernel size
    - ReLU activation
    - Valid padding
  - 2nd Block:
    - 64 filters, 5x5 kernel size
    - ReLU activation
    - Valid padding
  - Followed by MaxPooling (2x2)
- Total Convolutional Layers: 6
- Pooling Layers: 4 MaxPooling layers (2x2)
- Global Average Pooling layer
- Final Dense Layer: 
  - Single neuron with sigmoid activation for binary classification

#### Optimizer Characteristics:
- Adam Optimizer
- Learning Rate: 1e-4
- Learning rate decay based on epochs
- Loss Function: Binary Crossentropy

### 2. Convolutional Neural Network with RMSProp Optimizer

#### Architecture Details:
- Input Layer: 128x128x3 (image dimensions)
- Convolutional Layers:
  - 1st Layer: 32 filters, 3x3 kernel
  - 2nd Layer: 64 filters, 3x3 kernel
  - 3rd Layer: 128 filters, 3x3 kernel
- Pooling Layers: 3 MaxPooling layers (2x2)
- Flatten Layer
- Dense Layers:
  - Hidden Layer: 128 neurons with ReLU activation
  - Output Layer: Single neuron with sigmoid activation

#### Optimizer Characteristics:
- RMSProp Optimizer
- Learning Rate: 0.001
- Loss Function: Binary Crossentropy

### Training Configuration
- Epochs: 15
- Batch Size: 32
- Early Stopping: 
  - Monitored Metric: Validation Accuracy
  - Patience: 10 epochs

## Preprocessing Technique

### Error Level Analysis (ELA)
- Converts images to highlight potential forgery regions
- Helps in feature extraction for model training
- Key steps:
  - Resave image at specific quality
  - Calculate pixel differences
  - Enhance image to brighten suspicious regions

## Performance Metrics

### Model Accuracies
- CNN (Adam Optimizer): 94.05%
- CNN (RMSProp Optimizer): 92.36%

### Comparative Analysis
- Adam Optimizer showed slightly better performance
- Deep learning models captured more intricate image features

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn

## Future Improvements
- Experiment with more advanced architectures
- Integrate additional forgery detection techniques
- Enhance model generalization
- Develop real-time forgery detection system

## References
- [IEEE Paper](https://ieeexplore.ieee.org/document/10429021)
- [Springer Article](https://link.springer.com/article/10.1007/s11063-024-11448-9)

## Author
JUMANA.B

# ASL Alphabet Recognition using Convolutional Neural Networks

This project aims to recognize American Sign Language (ASL) alphabet gestures using Convolutional Neural Networks (CNNs). The model is trained to classify images of hand gestures corresponding to the 26 letters of the English alphabet and 3 additional symbols (space, delete, nothing).

## Overview

The American Sign Language (ASL) is a visual-gestural language used by deaf and hard-of-hearing individuals to communicate. This project focuses on recognizing ASL alphabet gestures using deep learning techniques.

## Dataset

The ASL Alphabet dataset used in this project consists of images representing hand gestures for each letter of the English alphabet and additional symbols. The dataset is divided into training, validation, and test sets.

## Model Architecture

The CNN model architecture consists of multiple convolutional layers followed by batch normalization, max-pooling, and fully connected layers. Dropout regularization is applied to prevent overfitting. The final layer uses a softmax activation function to output the probabilities of each class.
```python

Input (64, 64, 3)
       |
   Conv2D(64)
       |
BatchNormalization
       |
 MaxPooling2D
       |
   Conv2D(128)
       |
BatchNormalization
       |
 MaxPooling2D
       |
   Conv2D(256)
       |
BatchNormalization
       |
 MaxPooling2D
       |
   Conv2D(512)
       |
BatchNormalization
       |
 MaxPooling2D
       |
    Flatten
       |
    Dense(256)
       |
   Dropout(0.5)
       |
BatchNormalization
       |
    Dense(128)
       |
   Dropout(0.5)
       |
BatchNormalization
       |
    Dense(29)
       |
  Softmax Output
```
## Training

The model is trained using the training dataset and evaluated on the validation set. Training parameters such as batch size, learning rate, and optimizer are tuned to optimize performance. Data augmentation techniques may be applied to increase the robustness of the model.

## Evaluation

The trained model is evaluated on the test dataset to assess its performance in real-world scenarios. Metrics such as accuracy, precision, recall, and confusion matrix are computed to measure the model's effectiveness.

## Visualization

Feature maps are visualized to understand the learned representations at different layers of the network. This helps in interpreting how the model extracts and processes information from the input images.

## Dependencies

- Python 3
- TensorFlow
- Keras
- Matplotlib
- NumPy

## Usage

1. Clone the repository:


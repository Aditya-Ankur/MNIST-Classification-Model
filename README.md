# MNIST Digit Classification
This is a Simple Neural Network model that classifies the MNIST dataset into `10 classes (digits)` with accuracy of `0.9862142857142857` on the testing set.

## Dataset
- 56,000 training images and 14,000 testing images
- Grayscale images of size 28×28 pixels
- Each image contains a centered handwritten digit (0-9)
- Images are normalized to fit into a 28×28 pixel bounding box and anti-aliased, introducing grayscale levels

## Model Architecture
- Input layer: 28×28×1 (grayscale image)
- Two hidden layers each containing `512` neurons
- ReLU activation used on the hidden layers
- Output contains 10 classes and a `softmax` layer
- Total neurons are `1034`  

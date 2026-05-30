# Deep Learning-Based Handwritten Digit Recognition Using MNIST

A repository dedicated to implementing deep learning models for handwritten digit recognition (0–9) using Keras and TensorFlow. The project focuses on solving practical automation challenges—such as digital form processing in banking and automated sorting in postal systems—where traditional OCR tools fail due to variance in writing styles, noise, and geometric distortion.

## Project Overview

This repository demonstrates multiple approaches to training neural networks on the baseline MNIST dataset:
1. **Convolutional Neural Networks (CNNs):** Implemented for high-precision recognition, leveraging alternating convolutional and max-pooling layers to capture robust spatial hierarchies in stroke patterns.
2. **Artificial Neural Networks / Multi-Layer Perceptrons (ANN/MLP):** Deep fully connected networks featuring flat pixel-intensity vector evaluation, dense layer activation tuning, and comparative histogram analysis of normalized vs. raw pixel intensities.

### Real-World Use Cases Covered
* **Fintech Applications:** Automated remote cheque deposit systems and digital form-filling pipelines reading account details and transaction amounts.
* **Postal Automation:** National postal mail-sorting systems using zip-code scans for high-throughput automated categorization and routing.

## Repository Contents

* **CNN Digit Classifier:** Contains data preprocessing architectures utilizing data reshaped to standard multi-channel imagery (28x28x1) with categorical cross-entropy loss tracking, achieving over 99% test accuracy.
* **MLP Intensity Analytics:** Includes code visualizing training shapes (60000, 28, 28) and mapping individual array values directly alongside localized pixel frequency distribution histograms.

## Key Technical Specifications

* **Frameworks:** Keras, TensorFlow, NumPy, Matplotlib
* **Pre-processing:** Min-Max scaling normalization (pixel / 255.0), structural reshaping, and label encoding (to_categorical / sparse_categorical_crossentropy).
* **Optimizers:** Adam Optimizer
* **Output Functions:** Softmax activation mapping probabilities across 10 dense target nodes.

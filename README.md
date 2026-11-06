# Handwritten Digit Recognition using CNN (MNIST)

## Overview
A Convolutional Neural Network (CNN) that classifies handwritten digit images (0–9) using the MNIST dataset.

## Dataset
MNIST contains 70,000 grayscale 28×28 images: 60,000 training images and 10,000 test images. TensorFlow downloads the dataset automatically the first time the training script runs.

## Model architecture
- Input: 28×28×1
- Conv2D (32 filters, ReLU) + Batch Normalization + Max Pooling
- Conv2D (64 filters, ReLU) + Batch Normalization + Max Pooling
- Dropout, Flatten
- Dense (128, ReLU) + Dropout
- Dense (10, Softmax)

Optimizer: Adam. Loss: sparse categorical cross-entropy.

## Setup
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate
```bash
python train.py
```

This trains the CNN, evaluates it on the held-out test set, and saves the model as `mnist_cnn.keras`. Plots and metrics are saved in `outputs/`. Use the measured test accuracy printed by the script in your report; do not assume a fixed score.

## Predict a custom image
After training:

```bash
python predict.py path/to/digit.png
```

Use a centered, high-contrast image containing one digit on a plain background.

## Project report sections
1. Abstract and introduction
2. Problem statement and objectives
3. Dataset description
4. Preprocessing
5. CNN architecture and methodology
6. Training configuration
7. Results and evaluation
8. Conclusion and future work

## References
- https://www.tensorflow.org/tutorials/quickstart/beginner
- https://keras.io/examples/vision/

Note: Real-world handwriting may differ from MNIST. Custom images may require careful cropping, centering, and preprocessing.

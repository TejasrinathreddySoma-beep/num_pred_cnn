# Handwritten Digit Recognition using CNN (MNIST) — PyTorch

A Convolutional Neural Network built with **PyTorch** to classify handwritten digits (0–9) from the MNIST dataset.

## Dataset
MNIST contains 70,000 grayscale images of size 28×28: 60,000 training images and 10,000 test images. The torchvision dataset loader downloads the data automatically.

## CNN architecture
- Input: 1 × 28 × 28
- Conv2D: 32 filters, 3×3 + ReLU
- MaxPool2D: 2×2
- Conv2D: 64 filters, 3×3 + ReLU
- MaxPool2D: 2×2
- Flatten: 64 × 7 × 7 = 3,136 features
- Fully connected hidden layer: 128 neurons + ReLU + Dropout
- Output layer: 10 neurons (digits 0–9)

There are **3 learnable hidden layers** (two convolutional layers and one fully connected layer), excluding pooling, dropout, and flatten operations.

## Setup
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

## Train and evaluate
```bash
python train.py
```

The script downloads MNIST, trains the CNN, evaluates it on the test set, and saves:
- `mnist_cnn.pth` — model weights
- `outputs/metrics.json` — test loss and accuracy
- Accuracy and loss curves, confusion matrix, and prediction gallery in `outputs/`

The actual test accuracy is printed after training; it can vary slightly by environment and random seed.

## Predict a custom image
After training:

```bash
python predict.py path/to/digit.png
```

Provide a single digit on a plain background. The prediction script converts it to grayscale, inverts it to match MNIST's white-digit-on-black-background convention, and resizes it to 28×28.

## Project report sections
1. Abstract and introduction
2. Problem statement and objectives
3. Dataset description
4. Image preprocessing
5. CNN architecture and methodology
6. Training configuration
7. Results and evaluation
8. Conclusion and future work

## References
- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- https://pytorch.org/vision/stable/datasets.html#mnist

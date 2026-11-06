# Handwritten Digit Recognition using CNN (MNIST) — PyTorch + Streamlit

A PyTorch CNN that recognizes handwritten digits (0–9) from MNIST. Draw a digit in the Streamlit app and get the model's top 3 predictions with confidence scores.

## Features

- MNIST CNN with two convolutional layers
- Intel XPU support, then NVIDIA CUDA, then CPU fallback
- Training-time data augmentation (small rotations, shifts, scaling and shear)
- Drawing canvas or image upload for multiple handwritten digits
- Connected-component segmentation that splits separated digits and processes them left-to-right
- Per-digit top 3 predictions with confidence percentages
- Separate **Predict all digits** and **Clear / reset** buttons
- MNIST-style preprocessing: invert, threshold, crop, preserve aspect ratio, center and convert to 28×28
- Top 3 predicted digits and confidence percentages

## CNN architecture

| Layer | Configuration |
|---|---|
| Input | 1 × 28 × 28 grayscale |
| Convolution 1 | 32 filters, 3×3, ReLU |
| Max pooling | 2×2 |
| Convolution 2 | 64 filters, 3×3, ReLU |
| Max pooling | 2×2 |
| Flatten | 3,136 features |
| Dense hidden | 128 neurons, ReLU, Dropout |
| Output | 10 digit classes |

## Windows setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch. For a supported Intel GPU, install the XPU wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

For CPU-only instead, use:

```powershell
python -m pip install torch torchvision torchaudio
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

**Intel GPU note:** PyTorch XPU support depends on the exact processor/GPU and driver. The scripts automatically fall back to CPU if no supported accelerator is detected.

## Train the improved model

Run this after pulling the latest code. It downloads MNIST, applies augmentation to training images only, trains the CNN, evaluates on untouched test images, and saves the best validation checkpoint as `mnist_cnn.pth`.

```powershell
python train.py
```

Training can take some time. Wait until it prints the test accuracy and confirms the model weights have been saved. It also writes metrics and plots into `outputs/`.

**Important:** The retrained `mnist_cnn.pth` file is created on your computer and is ignored by Git. You must run `python train.py` locally before starting the app; the previous weights should not be treated as the augmented model.

## Launch the drawing app

After training completes:

```powershell
streamlit run streamlit_app.py
```

Choose **Draw digits** or **Upload image**. Enter an image with two or more separated, non-touching digits, then click **Predict all digits**. The app uses connected-component segmentation to find each digit, sorts them from left to right, normalizes each crop to 28×28, and predicts each one independently. It displays the recognized sequence and top 3 alternatives for every digit. Click **Clear / reset** to start again.

For best results, use dark digits on a light background, keep visible gaps between digits, and avoid touching or overlapping strokes. Connected-component segmentation cannot reliably split digits that touch, and a disconnected stroke within one digit may be treated as a separate component.

## Predict from an image file (optional)

```powershell
python predict.py path\to\digit.png
```

## Project files

- `train.py` — CNN architecture, augmented training, evaluation and plots
- `predict.py` — command-line prediction from an image file
- `streamlit_app.py` — drawing UI, preprocessing, Predict/Clear buttons and top-3 predictions
- `requirements.txt` — dependencies (PyTorch installed separately as above)
- `outputs/` — metrics and evaluation plots generated during training
- `mnist_cnn.pth` — trained model weights generated locally

## Recorded previous baseline

The earlier committed `outputs/metrics.json` reported 99.27% test accuracy before the new augmentation changes. The augmented model will produce new metrics when you run `train.py`; do not assume the old score applies to the new weights.

## References

- [MNIST dataset](https://pytorch.org/vision/stable/datasets.html#mnist)
- [PyTorch Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Streamlit documentation](https://docs.streamlit.io/)

# Handwritten Digit and Multi-Digit Sequence Recognition

A PyTorch CNN trained on MNIST with a Streamlit interface for recognizing one digit or a sequence of separated handwritten digits. You can draw digits on the canvas or upload an image. The app segments the image from left to right, preprocesses each digit, and predicts each one independently.

## Features

- Recognizes individual digits (0–9) and multi-digit sequences such as `12345`
- Draw handwritten digits on a large Streamlit canvas or upload a PNG/JPG image
- Segments separate, non-touching digits and orders them from left to right
- Applies MNIST-style preprocessing to every digit: grayscale conversion, thresholding, cropping, aspect-ratio-preserving resize, centering and 28×28 normalization
- Shows the complete predicted sequence
- Shows the top 3 digit predictions and confidence percentages for each position
- Separate **Predict all digits** and **Clear / reset** controls
- CNN training with data augmentation (rotation, translation, scaling and shear)
- Intel XPU support, then NVIDIA CUDA, then CPU fallback

## How multi-digit recognition works

1. Draw or upload an image containing multiple digits.
2. The app converts the image into a foreground mask and detects connected components.
3. Each detected component is cropped, resized proportionally and centered on a 28×28 grayscale image.
4. The CNN predicts each crop independently.
5. Predictions are sorted left-to-right and combined into a sequence.

For example, an input containing five clearly separated digits may produce a sequence such as `12345`. This is an illustration of the output format, not a guaranteed prediction.

**Segmentation limitation:** The current approach relies on connected components. Digits that touch or overlap can be treated as one component. Disconnected marks belonging to one digit may be treated as separate components. Leave clear gaps between digits for best results.

## Model architecture

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

## Accuracy and evaluation

The repository's recorded baseline metrics in `outputs/metrics.json` are:

| Evaluation | Result |
|---|---:|
| MNIST test accuracy (previous baseline) | **99.27%** |
| MNIST test loss (previous baseline) | **0.02213** |

**How to interpret this score:** 99.27% is the previously recorded accuracy for classifying individual MNIST test images. It is not a measured accuracy for complete multi-digit sequences. Sequence-level accuracy (the percentage of entire strings where every digit is correct) has not yet been measured. The augmentation changes also need to be trained locally before a new accuracy can be reported.

After training, `train.py` writes updated test metrics to `outputs/metrics.json`, along with an accuracy curve, loss curve and confusion matrix. Use those new metrics rather than assuming the previous baseline still applies.

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

PyTorch XPU support depends on the exact processor/GPU and driver. The scripts automatically fall back to CPU if no supported accelerator is detected.

## Train the model

Run this after pulling the latest code. The script downloads MNIST, applies augmentation to training images only, trains the CNN, evaluates it on untouched test images, and saves the best validation checkpoint as `mnist_cnn.pth`.

```powershell
python train.py
```

Wait for training to finish. The script prints test accuracy and saves model weights and evaluation outputs. Training can take some time.

**Important:** `mnist_cnn.pth` is generated on your computer and is ignored by Git. Run `python train.py` locally before starting the app if you need to create the model weights. The updated augmented-model accuracy cannot be known until this training run finishes.

## Launch the Streamlit app

```powershell
streamlit run streamlit_app.py
```

Select **Draw digits** or **Upload image**. Write two or more digits with visible gaps, then click **Predict all digits**. The app displays the recognized sequence and the top 3 alternatives with confidence percentages for each detected digit. Click **Clear / reset** to start again.

For best results:
- Use dark, solid strokes on a light background.
- Keep digits separated and avoid touching or overlapping strokes.
- Make digits reasonably large and fully visible in the input area.
- Avoid stray marks or detached strokes.

## Command-line prediction (single image)

```powershell
python predict.py path\to\digit.png
```

## Project files

- `train.py` — CNN architecture, augmented training, evaluation and plots
- `predict.py` — command-line prediction from an image file
- `streamlit_app.py` — drawing/upload UI, multi-digit segmentation, preprocessing and top-3 predictions
- `requirements.txt` — Python dependencies (install PyTorch separately as described above)
- `outputs/` — training metrics and evaluation plots generated locally
- `mnist_cnn.pth` — trained model weights generated locally

## References

- [MNIST dataset](https://pytorch.org/vision/stable/datasets.html#mnist)
- [PyTorch Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Streamlit documentation](https://docs.streamlit.io/)

# Handwritten Digit Recognition using CNN (MNIST) — PyTorch + Intel Arc XPU

A Convolutional Neural Network built with **PyTorch** to classify handwritten digits (0–9) from MNIST. The project automatically prefers an Intel GPU through PyTorch XPU, then NVIDIA CUDA, and finally CPU.

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

## Windows 11 setup — Intel Arc integrated graphics

**Important compatibility note:** PyTorch XPU does not support every Intel integrated GPU. Current official validation lists specific Intel Core Ultra processor generations with Intel Arc graphics, as well as Arc A-Series and B-Series cards. A generic Intel UHD/Iris Xe device may not be supported. If your processor/GPU model is unknown, check it in Windows Device Manager before assuming XPU will work.

### 1. Update the Intel graphics driver
Install the latest Windows 11 graphics driver for your exact PC/processor from your PC manufacturer's support page or Intel Driver & Support Assistant. Restart Windows after installation.

### 2. Create and activate a virtual environment
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install the PyTorch Intel XPU build
Use the official PyTorch XPU wheel index (not the regular CPU/CUDA install):

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 4. Install the remaining project dependencies
```powershell
python -m pip install -r requirements.txt
```

The requirements file intentionally does not list torch/torchvision, so a later requirements install will not replace your XPU-enabled PyTorch build with a default wheel.

### 5. Verify Intel GPU detection
Run:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('XPU available:', torch.xpu.is_available()); print('XPU devices:', torch.xpu.device_count() if torch.xpu.is_available() else 0); print('GPU:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'Not detected')"
```

You want `XPU available: True`. If it says `False`, confirm that your exact Intel GPU is in the supported hardware list, update the driver, restart, and reinstall the XPU wheels in the active virtual environment. The project will fall back to CPU rather than crash if XPU is unavailable.

## Train and evaluate
```powershell
python train.py
```

At startup, the script prints the selected device. For an available Intel GPU, it should report `Training device: xpu` and the GPU name. It downloads MNIST, trains the CNN, evaluates it on the test set, and saves:
- `mnist_cnn.pth` — model weights
- `outputs/metrics.json` — test loss and accuracy
- Accuracy and loss curves, confusion matrix, and prediction gallery in `outputs/`

## Results

The recorded evaluation metrics in `outputs/metrics.json` are:

| Metric | Result |
|---|---:|
| Test loss | 0.02213 |
| Test accuracy | **99.27%** |

These values are taken from the committed `outputs/metrics.json` file. The metrics file does not record which device performed training, so GPU usage should be confirmed separately from the `Training device` line printed by `train.py`.

## Predict a custom image
After training:

```powershell
python predict.py path\to\digit.png
```

Provide a single digit on a plain background. The prediction script converts it to grayscale, inverts it to match MNIST's white-digit-on-black-background convention, and resizes it to 28×28. Inference uses the same automatic XPU/CUDA/CPU device selection.

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
- PyTorch Intel GPU getting-started guide: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
- Intel GPU prerequisites: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html
- MNIST dataset: https://pytorch.org/vision/stable/datasets.html#mnist

 # Handwritten Digit & Alphabet Recognition (EMNIST ByClass)

A PyTorch CNN and Streamlit application for recognizing handwritten digits and English letters. The model is trained on **EMNIST ByClass**, which has 62 classes: digits 0–9, uppercase A–Z, and lowercase a–z. The app supports drawing or uploading a sequence of separated characters and predicts each character from left to right.

## Features

- 62 output classes: 10 digits + 26 uppercase letters + 26 lowercase letters
- Multi-character sequences such as `A7b2` or `hello` (characters should be separated)
- Streamlit drawing canvas and image upload
- Connected-component segmentation and left-to-right character ordering
- Per-character top 3 predictions and confidence percentages
- Character preprocessing to grayscale 28×28 inputs
- Training-time augmentation (small rotations, shifts, scaling and shear)
- Intel XPU, NVIDIA CUDA, or CPU device selection

## Dataset and class mapping

This project uses the torchvision **EMNIST ByClass** split. Its 62 labels are mapped in this order:

| Class IDs | Characters |
|---|---|
| 0–9 | `0 1 2 3 4 5 6 7 8 9` |
| 10–35 | `A` through `Z` |
| 36–61 | `a` through `z` |

Uppercase and lowercase letters are separate classes. Some characters look very similar (for example, `C/c`, `O/o`, and `P/p`), so per-character predictions can be uncertain. The EMNIST images are transposed into upright orientation consistently during training and evaluation.

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
| Output | 62 classes |

## Accuracy

The old **99.27%** result was measured on MNIST digits only. It does **not** apply to the new EMNIST model, which recognizes 62 digit/letter classes. No EMNIST accuracy is claimed before you run the new training script.

After training, the script prints the EMNIST ByClass test accuracy and writes the measured results to `outputs/emnist_metrics.json`. This is per-character test accuracy, not complete-sequence accuracy.

## Windows setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For a supported Intel GPU, install the PyTorch XPU wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

For CPU-only PyTorch, use:

```powershell
python -m pip install torch torchvision torchaudio
```

Install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
```

PyTorch XPU support depends on your exact Intel processor/GPU and driver. The training script falls back to CPU if no supported accelerator is available.

## Train the new EMNIST model

Pull the latest repository changes, then run:

```powershell
python train.py
```

The first run downloads EMNIST ByClass. It trains with augmentation, uses a validation split for early stopping, evaluates on the official test split, and saves:

- `emnist_byclass_cnn.pth` — new 62-class model weights
- `outputs/emnist_metrics.json` — measured test loss and accuracy
- `outputs/emnist_accuracy_curve.png` and `outputs/emnist_loss_curve.png` — training curves

**Training note:** EMNIST ByClass is much larger than MNIST (nearly 700,000 training examples), so training can take considerably longer. Keep the terminal open until it reports the test accuracy and saved model. The model file is generated locally and is ignored by Git.

## Run the Streamlit app

After training completes:

```powershell
streamlit run streamlit_app.py
```

Choose **Draw characters** or **Upload image**. Write digits and/or letters with visible gaps, then click **Predict sequence**. The app shows the combined predicted string and the top 3 alternatives for every detected character. Use **Clear / reset** to start again.

### Segmentation limitations

The current sequence reader uses connected components. Characters that touch or overlap may be grouped into one region. Dots on lowercase `i` and `j`, or detached strokes, may be treated as separate components. Leave a visible gap between characters. This first version is intended for separated handwritten characters, not cursive handwriting or joined-up words.

## Command-line prediction

For one character per image:

```powershell
python predict.py path\to\character.png
```

## Project files

- `train.py` — EMNIST ByClass dataset, 62-class CNN, augmentation, training and evaluation
- `predict.py` — command-line prediction for one character
- `streamlit_app.py` — drawing/upload interface, segmentation and sequence prediction
- `requirements.txt` — Python dependencies (install PyTorch separately)
- `outputs/emnist_metrics.json` — generated evaluation metrics
- `emnist_byclass_cnn.pth` — locally generated trained model weights

## References

- [EMNIST dataset paper and information](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Torchvision EMNIST documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)
- [PyTorch Intel GPU guide](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [Streamlit documentation](https://docs.streamlit.io/)


## Experimental: IAM handwritten line recognition (CNN + BiLSTM + CTC)

A separate line-level recognizer is being added without replacing the existing EMNIST character model. It converts line images to grayscale, extracts visual features with a CNN, models the left-to-right character sequence with a bidirectional LSTM, and uses CTC loss so characters do not need to be segmented individually.

### IAM dataset setup

1. Register for and download the **IAM Handwriting Database — Lines** images and the ASCII annotation/split files from the official IAM website: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Extract the files so the project has this structure (the script accepts either root-level annotation files or the common `ascii/` subfolder):

```text
data/iam/
  lines/
    a01/a01-000u/a01-000u-00.png
  lines.txt                 # or ascii/lines.txt
  trainset.txt              # or ascii/sets/trainset.txt
  validationset1.txt        # or ascii/sets/validationset1.txt
  testset.txt               # or ascii/sets/testset.txt
```

Keep the IAM images and annotations out of Git; the dataset is large and has its own access terms. The training script expects the IAM line-image IDs and transcription fields from the official files.

### Train the IAM line model

```powershell
python train_iam.py
```

The script uses grayscale conversion and contrast normalization, trains a CNN-BiLSTM with CTC loss, selects the best checkpoint using validation character error rate (CER), and evaluates the held-out test split. It saves `iam_crnn_ctc.pth`. This is a separate model from `emnist_byclass_cnn.pth`.

**Hardware note:** Start with the default small batch size. Sequence models use more memory than the old character classifier. Intel XPU operator support can vary by PyTorch version; if an operation is unsupported on XPU, run this experimental training script on CPU or use a compatible PyTorch XPU build.

### New files

- `iam_model.py` — CNN + bidirectional LSTM + CTC model and greedy decoder
- `train_iam.py` — IAM line dataset loader, grayscale preprocessing, training and CER evaluation

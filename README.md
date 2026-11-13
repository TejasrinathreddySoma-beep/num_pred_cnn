# Handwritten Digit & Character Recognition — EMNIST ByClass

A PyTorch convolutional neural network (CNN) that recognizes handwritten digits and English letters from the **EMNIST ByClass** dataset, with a Streamlit interface for drawing or uploading short sequences of separated characters.

This repository contains the character-level classifier. It is separate from the line-level IAM handwriting recognizer in the companion repository: [iam_handwriting_recognition](https://github.com/TejasrinathreddySoma-beep/iam_handwriting_recognition).

## What it does

- Classifies **62 character classes**: digits `0–9`, uppercase `A–Z`, and lowercase `a–z`.
- Lets a user draw characters on a canvas or upload an image.
- Segments an image into connected components and orders detected components from left to right.
- Displays a predicted sequence and the top three class probabilities for each detected component.
- Supports NVIDIA CUDA, Intel XPU (where the installed PyTorch build supports it), and CPU inference/training.

> **Scope:** this is a separated-character recognizer, not a cursive word or full-page handwriting recognizer. Leave visible gaps between characters.

## Model

The classifier accepts a single-channel 28 × 28 image and uses:

1. Two convolution blocks (32 and 64 filters), each followed by ReLU and 2 × 2 max pooling.
2. A 128-unit fully connected layer with ReLU and dropout.
3. A 62-class output layer.

The label order is fixed as:

| Class IDs | Labels |
|---|---|
| 0–9 | Digits `0` through `9` |
| 10–35 | Uppercase `A` through `Z` |
| 36–61 | Lowercase `a` through `z` |

The training and inference code applies the same EMNIST orientation correction so the characters are upright.

## Dataset and measured results

Training uses the torchvision **EMNIST ByClass** split. The script creates a reproducible 90/10 train/validation split from the training partition and evaluates the selected checkpoint on the official test partition.

<badge-placeholder/>

| Metric | Recorded result |
|---|---:|
| Test accuracy | **86.06%** |
| Test loss | **0.3824** |
| Best validation accuracy | **85.92%** |

These are per-character classification metrics from the committed run, not complete-sequence accuracy. The detailed values are in [outputs/emnist_metrics.json](outputs/emnist_metrics.json), with training curves in [outputs/emnist_accuracy_curve.png](outputs/emnist_accuracy_curve.png) and [outputs/emnist_loss_curve.png](outputs/emnist_loss_curve.png).

## Challenges addressed

- **EMNIST orientation:** raw EMNIST samples are not presented in the same orientation as ordinary handwritten images. A consistent transpose is applied in both training and evaluation.
- **62-way classification:** uppercase and lowercase letters are separate labels, which is more difficult than digit-only MNIST classification. The class mapping is explicit and shared with prediction.
- **Dataset memory:** training and validation views share one loaded training dataset instead of loading duplicate image arrays.
- **Input variation:** random rotation, translation, scaling, and shear augmentation help the classifier handle modest differences in handwriting.
- **Hardware differences:** device selection prefers a supported Intel XPU, then CUDA, then CPU; CPU-only execution remains available.
- **Character segmentation:** connected-component analysis gives a simple left-to-right sequence workflow. It works best when characters do not touch and detached strokes are limited.

## Repository contents

| Path | Purpose |
|---|---|
| `train.py` | EMNIST loading, CNN definition, augmentation, training, validation, and test evaluation |
| `predict.py` | Command-line prediction for one character image |
| `streamlit_app.py` | Drawing/upload UI, connected-component segmentation, and sequence predictions |
| `emnist_byclass_cnn.pth` | Trained 62-class model weights |
| `outputs/emnist_metrics.json` | Recorded test and validation metrics |
| `outputs/emnist_*_curve.png` | Training and validation curves |
| `requirements.txt` | Python dependencies other than PyTorch |

## Implementation guide

### 1. Get the project

```bash
git clone https://github.com/TejasrinathreddySoma-beep/num_pred_cnn.git
cd num_pred_cnn
```

### 2. Create an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install PyTorch and dependencies

Install a PyTorch build appropriate for your hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). For a supported Intel GPU, follow the [PyTorch XPU installation guide](https://pytorch.org/docs/stable/notes/get_start_xpu.html). For CPU-only use, install the CPU build.

Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

### 4. Train (optional)

The repository includes the trained weights, so you can run the app without retraining. To reproduce training:

```bash
python train.py
```

The first run downloads EMNIST ByClass automatically into `data/`. Training saves `emnist_byclass_cnn.pth` and refreshes the metrics and curve files under `outputs/`. On Windows or a memory-limited machine, set `EMNIST_WORKERS=0` before running if DataLoader workers cause resource problems.

### 5. Launch the application

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal. Select **Draw characters** or **Upload image**, provide clearly separated characters, and click **Predict sequence**. The app shows the combined prediction and top-three alternatives with confidence values.

### 6. Run a single-character prediction

```bash
python predict.py path/to/character.png
```

Use an image containing one centered character for the command-line predictor.

## Known limitations and next improvements

- Touching characters may merge into one component; detached dots or strokes (notably `i` and `j`) may be detected separately.
- The current segmentation assumes a mostly horizontal sequence and does not infer word spacing reliably.
- Similar-looking classes (for example, uppercase/lowercase pairs) remain challenging.
- The reported metric is per-character accuracy; evaluate sequence-level accuracy separately before using this in a production workflow.

Potential next steps are improved character grouping, word-level spacing, a sequence model, and evaluation on handwriting collected outside EMNIST.

## References

- [NIST — EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Torchvision EMNIST documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.EMNIST.html)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Streamlit documentation](https://docs.streamlit.io/)

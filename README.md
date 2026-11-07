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
  a01/a01-000u/a01-000u-00.png  # or lines/a01/a01-000u/... 
  lines.txt                 # or ascii/lines.txt
  trainset.txt              # or ascii/sets/trainset.txt
  validationset1.txt        # or ascii/sets/validationset1.txt
  testset.txt               # or ascii/sets/testset.txt
```

Keep the IAM images and annotations out of Git; the dataset is large and has its own access terms. The training script expects the IAM line-image IDs and transcription fields from the official files.

### macOS setup (Apple Silicon: M1/M2/M3/M4)

Create an ARM-native Python environment (Python 3.11 or 3.12 is a good choice):

```bash
xcode-select --install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

Verify that PyTorch can see the Apple GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

The expected MPS result is `True`. The IAM training script automatically selects Apple MPS when it is available.

The IAM archive may extract images directly as `data/iam/a01/a01-000u/a01-000u-00.png` (as in the Zenodo archive), or under `data/iam/lines/a01/a01-000u/`. The script supports both layouts.

### Continue training from a Windows checkpoint on a Mac

Copy the extracted `data/iam/` directory and your saved `iam_crnn_ctc.pth` file from the Windows computer to the same paths in this project on the Mac. Do not put the dataset or checkpoint in Git.

The first checkpoint created by older versions of this script contains model weights only. It can still be resumed, but the optimizer state cannot be restored, so training restarts with a fresh optimizer from the next epoch. New checkpoints also save the optimizer state and epoch number.

Back up the original checkpoint before resuming, then run:

```bash
cp iam_crnn_ctc.pth iam_crnn_ctc_epoch1_backup.pth
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_iam.py --resume
```

The fallback environment variable lets unsupported MPS operations run on CPU where possible. If an operation still fails, share the complete error rather than switching devices blindly. With the legacy epoch-1 checkpoint, training resumes at epoch 2.

### Train the IAM line model with an NVIDIA RTX GPU (Windows)

Use this section on the Windows/Linux computer that has the NVIDIA RTX card. The MacBook Air M4 cannot run CUDA.

1. Install or update the NVIDIA driver from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).
2. Open PowerShell in the project folder. Check that Windows sees the GPU:

```powershell
nvidia-smi
```

If this command is not found or does not list your RTX card, fix the NVIDIA driver installation before continuing.

3. Use 64-bit Python 3.11 or 3.12 and create a clean environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal, then activate again.

4. Install the CUDA-enabled PyTorch build. The current PyTorch installation selector is the source of truth for the best wheel for your driver and platform: [PyTorch Get Started](https://pytorch.org/get-started/locally/). Select **Windows / Pip / Python / CUDA** and run the exact command shown. Do not install the Intel XPU build on the RTX machine.

For a current CUDA 13.0 wheel, the command is:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

If the selector recommends another CUDA wheel for your driver, use its command instead.

5. Install the project's other dependencies:

```powershell
python -m pip install -r requirements.txt
```

6. Verify that PyTorch can use CUDA and identify your RTX card:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('Built with CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED')"
```

Do not start training unless `CUDA available: True` and your NVIDIA RTX model is printed.

### Download and arrange the IAM data

Download the IAM line images and the matching ASCII annotation/split files from the IAM source or the approved [Zenodo record](https://zenodo.org/records/19003721). Extract the files into:

```text
data/iam/
  a01/a01-000u/a01-000u-00.png
  lines.txt
  trainset.txt
  validationset1.txt
  testset.txt
```

The loader also accepts images under `data/iam/lines/` and annotation/split files under the documented `ascii/` folders. Keep the dataset out of Git; the IAM data is large and has its own access terms.

### Start CUDA training

From the project folder with the virtual environment activated:

```powershell
python train_iam.py --device cuda
```

The script automatically prefers CUDA when using `--device auto`, but `--device cuda` is recommended here because it stops with an error rather than silently using the CPU if CUDA is unavailable.

Useful options:

```powershell
# Lower memory use if you get a CUDA out-of-memory error
python train_iam.py --device cuda --batch-size 2

# Resume from the best saved checkpoint
python train_iam.py --device cuda --resume

# Choose a smaller number of epochs for a test run
python train_iam.py --device cuda --epochs 2
```

The default batch size is 4, epochs are 30, and Windows data-loader workers default to 0 for stability. If CUDA runs out of memory, stop the run and retry with batch size 2 (or 1). The best validation-CER checkpoint is saved as `iam_crnn_ctc.pth` in the project root. The script evaluates the test split after training.

### Resume training

The checkpoint file is local and is not committed to Git. Keep a backup before replacing or resuming it. Resume with:

```powershell
python train_iam.py --device cuda --resume
```

### Run the Streamlit app

The existing `streamlit_app.py` is the EMNIST character/sequence demo. It is not yet a full-page IAM handwriting reader. The IAM model currently trains and evaluates handwritten **line images**; page segmentation and a Streamlit page-recognition interface are separate future work.

To launch the existing EMNIST app:

```powershell
streamlit run streamlit_app.py
```

The script uses grayscale conversion and contrast normalization, trains a CNN-BiLSTM with CTC loss, selects the best checkpoint using validation character error rate (CER), and evaluates the held-out test split. It saves `iam_crnn_ctc.pth`, separate from `emnist_byclass_cnn.pth`.

### New files

- `iam_model.py` — CNN + bidirectional LSTM + CTC model and greedy decoder
- `train_iam.py` — IAM line dataset loader, grayscale preprocessing, training and CER evaluation

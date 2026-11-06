"""Train an EMNIST ByClass CNN with an Intel XPU-friendly input pipeline."""
from pathlib import Path
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

SEED = 42
BATCH_SIZE = 512
EPOCHS = 15
NUM_WORKERS = max(0, int(os.environ.get("EMNIST_WORKERS", "2")) )
MODEL_PATH = Path("emnist_byclass_cnn.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
NUM_CLASSES = len(CLASS_NAMES)


def transpose_emnist(image):
    """Correct raw EMNIST orientation; top-level for Windows worker pickling."""
    return image.transpose(Image.Transpose.TRANSPOSE)


class EMNISTView(Dataset):
    """Apply different transforms to indexed views of one shared EMNIST dataset."""
    def __init__(self, source, indices, transform):
        self.source = source
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source_index = int(self.indices[position])
        image = Image.fromarray(self.source.data[source_index].numpy())
        if self.transform is not None:
            image = self.transform(image)
        label = int(self.source.targets[source_index])
        return image, label


def get_device():
    """Prefer Intel XPU, then NVIDIA CUDA, otherwise CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Intel GPU detected: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No supported GPU detected; using CPU.")
    print(f"Training device: {device}")
    return device


class EMNISTCNN(nn.Module):
    """CNN classifier with 62 EMNIST ByClass output classes."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(dataset, *, shuffle, device):
    # Two workers overlap CPU image augmentation/loading with XPU computation.
    # Set EMNIST_WORKERS=0 if Windows runs short of RAM.
    workers = NUM_WORKERS
    options = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": False,
    }
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(SEED)

    device = get_device()
    print(f"Batch size: {BATCH_SIZE} | DataLoader workers: {NUM_WORKERS}")

    # Keep only ONE copy of the large EMNIST training dataset in memory.
    # Separate views apply augmentation to training and deterministic transforms
    # to validation without loading a second copy of the training images.
    orientation_fix = transforms.Lambda(transpose_emnist)
    train_transform = transforms.Compose([
        orientation_fix,
        transforms.RandomAffine(
            degrees=10, translate=(0.08, 0.08),
            scale=(0.90, 1.10), shear=5, fill=0,
        ),
        transforms.ToTensor(),
    ])
    evaluation_transform = transforms.Compose([
        orientation_fix,
        transforms.ToTensor(),
    ])

    train_source = datasets.EMNIST(
        "data", split="byclass", train=True, download=True,
    )
    test_source = datasets.EMNIST(
        "data", split="byclass", train=False, download=True,
    )

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(train_source), generator=generator)
    split_at = int(len(permutation) * 0.9)
    train_indices, val_indices = permutation[:split_at], permutation[split_at:]
    train_dataset = EMNISTView(train_source, train_indices, train_transform)
    val_dataset = EMNISTView(train_source, val_indices, evaluation_transform)
    test_dataset = EMNISTView(test_source, range(len(test_source)), evaluation_transform)

    train_loader = make_loader(train_dataset, shuffle=True, device=device)
    val_loader = make_loader(val_dataset, shuffle=False, device=device)
    test_loader = make_loader(test_dataset, shuffle=False, device=device)

    model = EMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = 0.0
    patience, epochs_without_improvement = 4, 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2%}"
        )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(f"\nEMNIST ByClass test loss: {test_loss:.4f}")
    print(f"EMNIST ByClass test accuracy: {test_accuracy:.2%}")

    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device, non_blocking=True))
            all_predictions.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification report:")
    print(classification_report(
        all_labels, all_predictions, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, zero_division=0, digits=4,
    ))
    with (OUTPUT_DIR / "emnist_metrics.json").open("w") as file:
        json.dump({
            "dataset": "EMNIST ByClass",
            "classes": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_validation_accuracy": best_val_accuracy,
            "batch_size": BATCH_SIZE,
            "dataloader_workers": NUM_WORKERS,
            "augmentation": "RandomAffine: rotation 10 degrees, translation 8%, scale 0.90-1.10, shear 5 degrees",
        }, file, indent=2)

    for metric, title, filename in [
        ("accuracy", "EMNIST ByClass Accuracy", "emnist_accuracy_curve.png"),
        ("loss", "EMNIST ByClass Loss", "emnist_loss_curve.png"),
    ]:
        plt.figure()
        plt.plot(history[f"train_{metric}"], label="Training")
        plt.plot(history[f"val_{metric}"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()

    print(f"\nSaved 62-class model weights to {MODEL_PATH}")
    print(f"Saved EMNIST metrics and curves in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

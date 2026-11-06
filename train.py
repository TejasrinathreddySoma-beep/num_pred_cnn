"""Train and evaluate a PyTorch CNN on the MNIST handwritten digit dataset."""
from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
MODEL_PATH = Path("mnist_cnn.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


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


class MNISTCNN(nn.Module):
    """CNN with two convolutional layers and one dense hidden layer."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
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
    transform = transforms.ToTensor()

    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_dataset, val_dataset = random_split(
        full_train, [54_000, 6_000],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_accuracy = 0.0
    patience, epochs_without_improvement = 3, 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4%}"
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
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    model.eval()
    all_predictions, all_labels, sample_images = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(device))
            all_predictions.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            if len(sample_images) < 12:
                sample_images.extend(images[:12 - len(sample_images)])

    print("\nClassification report:")
    print(classification_report(all_labels, all_predictions, digits=4))
    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump({"test_loss": test_loss, "test_accuracy": test_accuracy}, f, indent=2)

    for metric, title, filename in [
        ("accuracy", "CNN Accuracy", "accuracy_curve.png"),
        ("loss", "CNN Loss", "loss_curve.png"),
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

    ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions, cmap="Blues")
    plt.title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for index, ax in enumerate(axes.flat):
        ax.imshow(sample_images[index].squeeze().numpy(), cmap="gray")
        ax.set_title(f"True: {all_labels[index]} | Pred: {all_predictions[index]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_gallery.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved model weights to {MODEL_PATH}")
    print(f"Saved plots and metrics in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

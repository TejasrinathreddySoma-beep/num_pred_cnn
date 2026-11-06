"""Predict a handwritten digit using the trained PyTorch MNIST CNN."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from train import MNISTCNN


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit.")
    parser.add_argument("image", help="Path to an image containing one digit")
    parser.add_argument("--model", default="mnist_cnn.pth", help="Model weights path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. Run python train.py first."
        )

    image = Image.open(args.image).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))
    tensor = transforms.ToTensor()(np.asarray(image)).unsqueeze(0).to(device)

    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, digit = probabilities.max(dim=0)

    print(f"Predicted digit: {digit.item()}")
    print(f"Confidence: {confidence.item():.2%}")


if __name__ == "__main__":
    main()

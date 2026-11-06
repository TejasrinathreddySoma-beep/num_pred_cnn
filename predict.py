"""Predict one handwritten EMNIST character from an image file."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from train import EMNISTCNN, CLASS_NAMES, get_device


def main():
    parser = argparse.ArgumentParser(description="Predict one EMNIST character.")
    parser.add_argument("image", help="Image path containing one digit or letter")
    parser.add_argument("--model", default="emnist_byclass_cnn.pth", help="Model weights path")
    args = parser.parse_args()

    device = get_device()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}. Run python train.py first.")

    image = Image.open(args.image).convert("L")
    border = np.concatenate((
        np.asarray(image)[0, :], np.asarray(image)[-1, :],
        np.asarray(image)[:, 0], np.asarray(image)[:, -1],
    ))
    if np.median(border) > 127:
        image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(np.asarray(image)).unsqueeze(0).to(device)

    model = EMNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        top_probs, top_ids = torch.topk(probabilities, k=3)
    for rank, (class_id, probability) in enumerate(zip(top_ids, top_probs), start=1):
        print(f"{rank}. {CLASS_NAMES[int(class_id)]}: {float(probability):.2%}")


if __name__ == "__main__":
    main()

"""Predict a digit from a custom image using the trained MNIST CNN."""
import argparse

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(
        description="Predict one handwritten digit from an image."
    )
    parser.add_argument("image", help="Path to an image containing one digit")
    parser.add_argument(
        "--model", default="mnist_cnn.keras", help="Saved Keras model path"
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.fit(image, (28, 28))
    array = np.asarray(image, dtype="float32") / 255.0
    array = array.reshape(1, 28, 28, 1)

    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(array, verbose=0)[0]
    digit = int(np.argmax(probabilities))

    print(f"Predicted digit: {digit}")
    print(f"Confidence: {probabilities[digit]:.2%}")


if __name__ == "__main__":
    main()

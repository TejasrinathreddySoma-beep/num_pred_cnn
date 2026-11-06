"""Train and evaluate a CNN on the MNIST handwritten digit dataset."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # Load 60,000 training images and 10,000 test images.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale pixels to [0, 1] and add the grayscale channel dimension.
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "mnist_cnn.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    predictions = model.predict(x_test, verbose=0).argmax(axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=4))

    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump(
            {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)},
            f,
            indent=2,
        )

    # Training and validation accuracy.
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=150)
    plt.close()

    # Training and validation loss.
    plt.figure()
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    plt.title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Display a random selection of predictions.
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x_test), size=12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        ax.set_title(f"True: {y_test[idx]} | Pred: {predictions[idx]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_gallery.png", dpi=150)
    plt.close(fig)

    print("\nSaved model: mnist_cnn.keras")
    print(f"Saved plots and metrics in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

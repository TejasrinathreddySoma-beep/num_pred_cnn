"""Streamlit drawing interface for the trained MNIST CNN."""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from train import MNISTCNN, get_device

MODEL_PATH = Path("mnist_cnn.pth")


@st.cache_resource
def load_model(model_path_string: str):
    device = get_device()
    model_path = Path(model_path_string)
    if not model_path.is_file():
        return None, device
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def prepare_canvas_image(rgba_image: np.ndarray) -> torch.Tensor:
    """Convert a white canvas with black ink to a centered MNIST-like tensor."""
    rgb = Image.fromarray(rgba_image.astype("uint8")).convert("RGB")
    grayscale = ImageOps.grayscale(rgb)
    digit = ImageOps.invert(grayscale)

    # Crop empty margins and fit the digit inside a 20x20 area.
    bbox = digit.getbbox()
    mnist_image = Image.new("L", (28, 28), 0)
    if bbox:
        digit = digit.crop(bbox)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
        x = (28 - digit.width) // 2
        y = (28 - digit.height) // 2
        mnist_image.paste(digit, (x, y))

    return transforms.ToTensor()(mnist_image).unsqueeze(0)


st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✍️", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit from 0 to 9 in the box. The trained CNN will predict it using the saved model weights.")

model, device = load_model(str(MODEL_PATH))
if model is None:
    st.error("Model weights were not found. Train the model first by running python train.py.")
    st.code("python train.py\nstreamlit run streamlit_app.py", language="powershell")
    st.stop()

st.caption(f"Model device: {device}")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    display_toolbar=True,
)

if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] < 250):
    input_tensor = prepare_canvas_image(canvas_result.image_data).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        confidence, prediction = probabilities.max(dim=0)

    st.subheader(f"Prediction: {prediction.item()}")
    st.metric("Model confidence", f"{confidence.item():.1%}")
    st.bar_chart(
        {str(i): float(probabilities[i].item()) for i in range(10)},
        y_label="Probability",
    )
else:
    st.info("Start drawing in the box above. Use the eraser or clear button to try again.")

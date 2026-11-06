"""Streamlit drawing UI for the trained MNIST CNN."""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from train import MNISTCNN, get_device

MODEL_PATH = Path("mnist_cnn.pth")
CANVAS_SIZE = 280


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
    """Convert black canvas ink to a centered 28x28 MNIST-style white-on-black digit."""
    rgb = Image.fromarray(rgba_image.astype(np.uint8)).convert("RGB")
    gray = ImageOps.grayscale(rgb)
    digit = ImageOps.invert(gray)

    # Remove antialiasing noise, then crop tightly around the actual ink.
    digit = digit.point(lambda pixel: 255 if pixel > 35 else 0)
    bbox = digit.getbbox()
    output = Image.new("L", (28, 28), 0)
    if bbox is None:
        return transforms.ToTensor()(output).unsqueeze(0)

    digit = digit.crop(bbox)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2
    output.paste(digit, (x, y))

    # Shift the digit's intensity-weighted center of mass to the image center.
    pixels = np.asarray(output, dtype=np.float32)
    mass = float(pixels.sum())
    if mass > 0:
        ys, xs = np.indices(pixels.shape)
        center_x = float((xs * pixels).sum() / mass)
        center_y = float((ys * pixels).sum() / mass)
        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))
        output = output.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    return transforms.ToTensor()(output).unsqueeze(0)


st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✍️", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw one digit from 0 to 9. Press **Predict** when you are ready.")

model, device = load_model(str(MODEL_PATH))
if model is None:
    st.error("Model weights are missing. Train the improved model first:")
    st.code("python train.py")
    st.stop()

st.caption(f"Model device: {device}")
st.info("Tip: draw one complete digit in the center of the canvas. Avoid touching the edges.")

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key=f"digit_canvas_{st.session_state.canvas_version}",
    return_image_data=True,
)

predict_col, clear_col = st.columns(2)
with predict_col:
    predict_clicked = st.button("🔍 Predict digit", type="primary", use_container_width=True)
with clear_col:
    clear_clicked = st.button("🧹 Clear canvas", use_container_width=True)

if clear_clicked:
    st.session_state.canvas_version += 1
    st.session_state.last_prediction = None
    st.rerun()

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Draw a digit first, then press Predict.")
    else:
        rgb_pixels = canvas_result.image_data[:, :, :3]
        if not np.any(rgb_pixels < 245):
            st.warning("The canvas is empty. Draw a digit first.")
        else:
            input_tensor = prepare_canvas_image(canvas_result.image_data).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(input_tensor), dim=1)[0]
                top_probabilities, top_indices = torch.topk(probabilities, k=3)
            st.session_state.last_prediction = [
                (int(index.item()), float(prob.item()))
                for index, prob in zip(top_indices, top_probabilities)
            ]

if st.session_state.last_prediction:
    predictions = st.session_state.last_prediction
    st.subheader(f"Predicted digit: {predictions[0][0]}")
    st.metric("Confidence", f"{predictions[0][1]:.1%}")
    st.markdown("**Top 3 predictions**")
    for rank, (digit, probability) in enumerate(predictions, start=1):
        st.write(f"{rank}. Digit **{digit}** — {probability:.1%}")
        st.progress(probability)
else:
    st.caption("Your prediction and top 3 alternatives will appear here.")

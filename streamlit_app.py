"""Streamlit multi-digit MNIST recognizer: draw or upload separated digits."""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from train import MNISTCNN, get_device

MODEL_PATH = Path("mnist_cnn.pth")
CANVAS_SIZE = 320


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


def to_foreground(image: Image.Image) -> np.ndarray:
    """Convert an image to a binary mask where digit ink is True."""
    gray = np.asarray(ImageOps.grayscale(image).convert("L"), dtype=np.uint8)
    h, w = gray.shape
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background_is_light = float(np.median(border)) >= 127
    intensity = 255 - gray if background_is_light else gray

    # Scale contrast and suppress faint background noise.
    peak = int(intensity.max())
    if peak == 0:
        return np.zeros_like(intensity, dtype=bool)
    threshold = max(25, int(peak * 0.18))
    mask = intensity >= threshold

    # Remove isolated one-pixel noise without joining neighboring digits.
    structure = np.ones((2, 2), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure)
    return mask


def find_digit_regions(mask: np.ndarray):
    """Find separate, non-touching digit components and return left-to-right crops."""
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    components = []
    total_ink = int(mask.sum())
    min_area = max(8, int(total_ink * 0.001))

    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        component_mask = labels[slices] == component_id
        area = int(component_mask.sum())
        height = y_slice.stop - y_slice.start
        width = x_slice.stop - x_slice.start
        if area < min_area or height < 4 or width < 2:
            continue
        components.append((x_slice.start, y_slice.start, x_slice.stop, y_slice.stop))

    components.sort(key=lambda box: box[0])
    return components


def normalize_digit(mask: np.ndarray, box) -> torch.Tensor:
    """Crop one component, preserve aspect ratio, and center it like MNIST."""
    x0, y0, x1, y1 = box
    pad = 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)
    crop = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))

    # Shift the ink's center of mass to the center of the 28x28 input.
    pixels = np.asarray(canvas, dtype=np.float32)
    mass = float(pixels.sum())
    if mass > 0:
        ys, xs = np.indices(pixels.shape)
        cx = float((xs * pixels).sum() / mass)
        cy = float((ys * pixels).sum() / mass)
        dx, dy = int(round(13.5 - cx)), int(round(13.5 - cy))
        canvas = canvas.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )
    return transforms.ToTensor()(canvas)


def image_to_pil(source) -> Image.Image:
    """Accept Streamlit canvas RGBA arrays or uploaded image files."""
    if isinstance(source, np.ndarray):
        return Image.fromarray(source.astype(np.uint8)).convert("RGB")
    return Image.open(source).convert("RGB")


def predict_digits(model, device, source):
    pil_image = image_to_pil(source)
    foreground = to_foreground(pil_image)
    boxes = find_digit_regions(foreground)
    if not boxes:
        return [], pil_image

    tensors = torch.stack([normalize_digit(foreground, box).squeeze(0) for box in boxes])
    with torch.no_grad():
        logits = model(tensors.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
    results = []
    for index, row in enumerate(probabilities):
        top_probs, top_ids = torch.topk(row, k=3)
        results.append({
            "position": index + 1,
            "digit": int(top_ids[0]),
            "confidence": float(top_probs[0]),
            "top3": [(int(d), float(p)) for d, p in zip(top_ids, top_probs)],
            "box": boxes[index],
        })
    return results, pil_image


st.set_page_config(page_title="Multi-Digit Recognizer", page_icon="✍️", layout="wide")
st.title("✍️ Handwritten Multi-Digit Recognizer")
st.write(
    "Draw or upload an image containing multiple **separated, non-touching digits**. "
    "The app segments the digits from left to right and predicts each one separately."
)

model, device = load_model(str(MODEL_PATH))
if model is None:
    st.error("Trained model weights are missing. Run python train.py first.")
    st.code("python train.py\nstreamlit run streamlit_app.py", language="powershell")
    st.stop()

st.caption(f"Model device: {device}")
st.info(
    "For reliable segmentation, leave a visible gap between digits. "
    "Digits that touch or overlap may be detected as one component."
)

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "multi_results" not in st.session_state:
    st.session_state.multi_results = None

input_mode = st.radio("Input method", ["Draw digits", "Upload image"], horizontal=True)
source = None

if input_mode == "Draw digits":
    st.caption("Write two or more digits with space between them.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=16,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=CANVAS_SIZE,
        height=180,
        drawing_mode="freedraw",
        key=f"multi_digit_canvas_{st.session_state.canvas_version}",
        return_image_data=True,
    )
    if canvas_result.image_data is not None:
        source = canvas_result.image_data
else:
    uploaded = st.file_uploader(
        "Choose a PNG, JPG or JPEG containing separated handwritten digits",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        source = uploaded
        st.image(uploaded, caption="Uploaded input", use_container_width=True)

predict_col, clear_col = st.columns(2)
with predict_col:
    predict_clicked = st.button("🔍 Predict all digits", type="primary", use_container_width=True)
with clear_col:
    clear_clicked = st.button("🧹 Clear / reset", use_container_width=True)

if clear_clicked:
    st.session_state.canvas_version += 1
    st.session_state.multi_results = None
    st.rerun()

if predict_clicked:
    if source is None:
        st.warning("Draw digits or upload an image before predicting.")
    else:
        results, source_image = predict_digits(model, device, source)
        if not results:
            st.warning("No clear digit components were found. Use darker, larger digits.")
            st.session_state.multi_results = None
        else:
            st.session_state.multi_results = results

if st.session_state.multi_results:
    results = st.session_state.multi_results
    predicted_string = "".join(str(item["digit"]) for item in results)
    st.subheader(f"Recognized sequence: {predicted_string}")
    st.write(f"Detected {len(results)} separate digit(s), ordered from left to right.")

    for item in results:
        with st.container(border=True):
            st.markdown(
                f"**Digit {item['position']} — prediction: {item['digit']}** "
                f"({item['confidence']:.1%} confidence)"
            )
            st.markdown("Top 3 alternatives")
            for rank, (digit, probability) in enumerate(item["top3"], start=1):
                st.write(f"{rank}. **{digit}** — {probability:.1%}")
                st.progress(probability)
else:
    st.caption("The recognized digit sequence and per-digit top 3 predictions will appear here.")

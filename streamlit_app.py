"""Streamlit handwritten EMNIST recognizer for digit and letter sequences."""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from scipy import ndimage
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

from train import EMNISTCNN, CLASS_NAMES, get_device

MODEL_PATH = Path("emnist_byclass_cnn.pth")
CANVAS_WIDTH = 850
CANVAS_HEIGHT = 320


@st.cache_resource
def load_model(model_path_string: str):
    device = get_device()
    model_path = Path(model_path_string)
    if not model_path.is_file():
        return None, device
    model = EMNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def to_foreground(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image).convert("L"), dtype=np.uint8)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background_is_light = float(np.median(border)) >= 127
    intensity = 255 - gray if background_is_light else gray
    peak = int(intensity.max())
    if peak == 0:
        return np.zeros_like(intensity, dtype=bool)
    mask = intensity >= max(25, int(peak * 0.18))
    return ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))


def find_character_regions(mask: np.ndarray):
    """Find separated connected components and order them left-to-right."""
    labels, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    components = []
    min_area = max(8, int(mask.sum() * 0.001))
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        ys, xs = slices
        area = int((labels[slices] == component_id).sum())
        height, width = ys.stop - ys.start, xs.stop - xs.start
        if area >= min_area and height >= 4 and width >= 2:
            components.append((xs.start, ys.start, xs.stop, ys.stop))
    components.sort(key=lambda box: box[0])
    return components


def normalize_character(mask: np.ndarray, box) -> torch.Tensor:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
    x1, y1 = min(mask.shape[1], x1 + 2), min(mask.shape[0], y1 + 2)
    crop = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28 - crop.width) // 2, (28 - crop.height) // 2))

    pixels = np.asarray(canvas, dtype=np.float32)
    mass = float(pixels.sum())
    if mass > 0:
        ys, xs = np.indices(pixels.shape)
        cx = float((xs * pixels).sum() / mass)
        cy = float((ys * pixels).sum() / mass)
        dx, dy = int(round(13.5 - cx)), int(round(13.5 - cy))
        canvas = canvas.transform(
            (28, 28), Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR, fillcolor=0,
        )
    return transforms.ToTensor()(canvas)


def image_to_pil(source) -> Image.Image:
    if isinstance(source, np.ndarray):
        return Image.fromarray(source.astype(np.uint8)).convert("RGB")
    return Image.open(source).convert("RGB")


def predict_characters(model, device, source):
    pil_image = image_to_pil(source)
    foreground = to_foreground(pil_image)
    boxes = find_character_regions(foreground)
    if not boxes:
        return []
    batch = torch.stack([normalize_character(foreground, box) for box in boxes])
    with torch.no_grad():
        probabilities = torch.softmax(model(batch.to(device)), dim=1).cpu()
    results = []
    for index, row in enumerate(probabilities):
        top_probs, top_ids = torch.topk(row, k=3)
        results.append({
            "position": index + 1,
            "character": CLASS_NAMES[int(top_ids[0])],
            "confidence": float(top_probs[0]),
            "top3": [(CLASS_NAMES[int(class_id)], float(prob))
                     for class_id, prob in zip(top_ids, top_probs)],
        })
    return results


st.set_page_config(page_title="Handwriting Reader", page_icon="✍️", layout="wide")
st.markdown("<style>.block-container{max-width:1180px;padding-top:2rem;}div[data-testid='stMetric']{border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px;}</style>", unsafe_allow_html=True)

recognition_mode = st.radio(
    "Choose recognition model",
    ["IAM handwritten text line", "EMNIST digits and letters"],
    horizontal=True,
)

if recognition_mode == "IAM handwritten text line":
    from predict_iam import get_inference_device, load_model as load_iam_model, predict_line

    st.title("✍️ IAM Handwriting Reader")
    st.write("Read a handwritten line or a page with multiple separated lines. The trained CNN + BiLSTM + CTC model predicts words, not isolated character boxes.")
    iam_checkpoint = Path("iam_crnn_ctc.pth")
    device = get_inference_device()

    @st.cache_resource
    def cached_iam_model(checkpoint_path, checkpoint_mtime, device_name):
        return load_iam_model(checkpoint_path, torch.device(device_name))

    if not iam_checkpoint.is_file():
        st.error("The trained IAM weights are missing. Copy iam_crnn_ctc.pth into the project root.")
        st.code("iam_crnn_ctc.pth", language="text")
        st.stop()

    model = cached_iam_model(str(iam_checkpoint.resolve()), iam_checkpoint.stat().st_mtime, str(device))
    st.caption(f"Loaded weights: iam_crnn_ctc.pth · Device: {device}")
    st.info("For best results use a straight, clear scan. Whole-page mode attempts to split separated handwriting lines automatically.")
    input_kind = st.radio("Input type", ["Whole page (auto-split lines)", "Single text line"], horizontal=True)
    uploaded_line = st.file_uploader("Upload handwriting image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_line is not None:
        line_image = Image.open(uploaded_line).convert("L")
        st.image(line_image, caption=f"Uploaded image · {line_image.width} × {line_image.height}px", use_container_width=True)
        if st.button("✨ Recognize handwriting", type="primary", use_container_width=True):
            with st.spinner("Recognizing handwriting..."):
                if input_kind == "Single text line":
                    line_crops = [line_image]
                else:
                    import numpy as np
                    from PIL import ImageOps
                    gray = ImageOps.autocontrast(line_image)
                    arr = np.asarray(gray, dtype=np.uint8)
                    ink = arr < 190
                    projection = np.convolve(ink.sum(axis=1).astype(float), np.ones(7)/7, mode="same")
                    active = projection >= max(2, arr.shape[1] * 0.002)
                    indices = np.flatnonzero(active)
                    line_crops = []
                    if len(indices):
                        groups, start_y, previous = [], int(indices[0]), int(indices[0])
                        for y in indices[1:]:
                            y = int(y)
                            if y - previous > 8:
                                if previous - start_y >= 7: groups.append((start_y, previous + 1))
                                start_y = y
                            previous = y
                        if previous - start_y >= 7: groups.append((start_y, previous + 1))
                        for y0, y1 in groups:
                            if y1-y0 < 10: continue
                            cols = np.flatnonzero(ink[y0:y1].any(axis=0))
                            if len(cols):
                                pad = max(5, int(arr.shape[1]*.008))
                                line_crops.append(gray.crop((max(0,int(cols[0])-pad), max(0,y0-5), min(arr.shape[1],int(cols[-1])+1+pad), min(arr.shape[0],y1+5))))
                if not line_crops:
                    st.warning("No lines detected. Try Single text line mode or upload a clearer image.")
                else:
                    predictions = [predict_line(model, crop, device) for crop in line_crops]
                    recognized = "\\n".join(text.strip() for text in predictions)
                    st.success(f"Processed {len(predictions)} line(s)")
                    st.subheader("Recognized text")
                    st.text_area("Review and copy the result", value=recognized, height=min(360,max(130,48*len(predictions))))
                    st.download_button("Download recognized text", data=recognized, file_name="recognized_handwriting.txt", mime="text/plain")
                    with st.expander("View detected line crops"):
                        for index, (crop, text) in enumerate(zip(line_crops,predictions),1):
                            st.image(crop, caption=f"Line {index}: {text}", use_container_width=True)
    st.stop()

st.title("✍️ Handwritten Digit & Alphabet Recognizer")

st.write(
    "Draw or upload a sequence containing digits (0–9), uppercase letters (A–Z), "
    "or lowercase letters (a–z). Separate characters with visible spaces."
)

model, device = load_model(str(MODEL_PATH))
if model is None:
    st.error("The EMNIST model weights are missing. Train the new model first.")
    st.code("python train.py\nstreamlit run streamlit_app.py", language="powershell")
    st.stop()

st.caption(f"Model device: {device} · 62 classes (0–9, A–Z, a–z)")
st.info(
    "The segmenter detects separate connected components. Touching characters, "
    "dots on i/j, and disconnected strokes may be split or joined incorrectly."
)

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "character_results" not in st.session_state:
    st.session_state.character_results = None

input_mode = st.radio("Input method", ["Draw characters", "Upload image"], horizontal=True)
source = None
if input_mode == "Draw characters":
    st.caption("Write a sequence such as A7b2 with a clear gap between each character.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        drawing_mode="freedraw",
        key=f"emnist_canvas_{st.session_state.canvas_version}",
        return_image_data=True,
    )
    if canvas_result.image_data is not None:
        source = canvas_result.image_data
else:
    uploaded = st.file_uploader(
        "Upload a PNG, JPG or JPEG containing separated digits/letters",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        source = uploaded
        st.image(uploaded, caption="Uploaded input", use_container_width=True)

predict_col, clear_col = st.columns(2)
with predict_col:
    predict_clicked = st.button("🔍 Predict sequence", type="primary", use_container_width=True)
with clear_col:
    clear_clicked = st.button("🧹 Clear / reset", use_container_width=True)

if clear_clicked:
    st.session_state.canvas_version += 1
    st.session_state.character_results = None
    st.rerun()

if predict_clicked:
    if source is None:
        st.warning("Draw characters or upload an image before predicting.")
    else:
        results = predict_characters(model, device, source)
        if not results:
            st.warning("No clear character components were found. Draw larger, darker characters.")
            st.session_state.character_results = None
        else:
            st.session_state.character_results = results

if st.session_state.character_results:
    results = st.session_state.character_results
    predicted_text = "".join(item["character"] for item in results)
    st.subheader(f"Recognized sequence: {predicted_text}")
    st.write(f"Detected {len(results)} character(s), ordered from left to right.")
    for item in results:
        with st.container(border=True):
            st.markdown(
                f"**Character {item['position']} — {item['character']}** "
                f"({item['confidence']:.1%} confidence)"
            )
            st.markdown("Top 3 alternatives")
            for rank, (character, probability) in enumerate(item["top3"], start=1):
                st.write(f"{rank}. **{character}** — {probability:.1%}")
                st.progress(probability)
else:
    st.caption("The recognized sequence and per-character top 3 predictions will appear here.")

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
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import hashlib

    # ChatGPT-inspired layout: centered conversation column, restrained chrome,
    # one upload action, then a clean assistant-style result.
    st.markdown("""
    <style>
      .block-container {max-width: 850px; padding-top: 2.0rem; padding-bottom: 5rem;}
      header[data-testid="stHeader"] {background: transparent;}
      .iam-heading {font-size: 2rem; font-weight: 650; letter-spacing: -.035em; margin: 0 0 .35rem;}
      .iam-subtitle {color: var(--text-color); opacity: .68; font-size: .96rem; margin-bottom: 1.5rem;}
      .iam-status {display:inline-flex; gap:.45rem; align-items:center; padding:.35rem .7rem;
        border:1px solid rgba(128,128,128,.25); border-radius:999px; font-size:.78rem;
        opacity:.82; margin:.15rem .35rem .9rem 0;}
      .iam-panel {border:1px solid rgba(128,128,128,.24); border-radius:18px;
        padding:1rem 1.1rem; background:rgba(128,128,128,.045); margin:.5rem 0 1.1rem;}
      .iam-assistant {border:1px solid rgba(128,128,128,.20); border-radius:18px;
        padding:1.1rem 1.25rem; margin-top:1rem; background:rgba(128,128,128,.055);}
      .iam-result-label {font-size:.78rem; text-transform:uppercase; letter-spacing:.08em;
        opacity:.58; margin-bottom:.55rem;}
      div.stButton > button[kind="primary"] {border-radius:999px; min-height:2.8rem;}
      div[data-testid="stTextArea"] textarea {font-size:1rem; line-height:1.65; border-radius:12px;}
      div[data-testid="stFileUploader"] {border-radius:14px;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="iam-heading">✍️ Handwriting Reader</div>', unsafe_allow_html=True)
    st.markdown('<div class="iam-subtitle">Upload a handwritten page or line. I’ll turn it into editable text.</div>', unsafe_allow_html=True)

    iam_checkpoint = Path("iam_crnn_ctc.pth")
    device = get_inference_device()

    @st.cache_resource
    def cached_iam_model(checkpoint_path, checkpoint_mtime, device_name):
        return load_iam_model(checkpoint_path, torch.device(device_name))

    if not iam_checkpoint.is_file():
        st.error("The IAM model file is missing. Put iam_crnn_ctc.pth in the project folder.")
        st.stop()
    try:
        model = cached_iam_model(str(iam_checkpoint.resolve()), iam_checkpoint.stat().st_mtime, str(device))
    except Exception as exc:
        st.error(f"Could not load IAM model: {exc}")
        st.stop()

    device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device).upper()
    st.markdown(
        f'<span class="iam-status">● IAM model loaded</span>'
        f'<span class="iam-status">▣ iam_crnn_ctc.pth</span>'
        f'<span class="iam-status">◉ {device_label}</span>',
        unsafe_allow_html=True,
    )

    with st.container():
        input_kind = st.radio(
            "What are you uploading?",
            ["Whole page", "Single text line"],
            horizontal=True,
            key="iam_input_kind",
        )
        uploaded_line = st.file_uploader(
            "Drop an image here or browse files",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="iam_upload",
            label_visibility="visible",
        )

    if uploaded_line is None:
        st.caption("For best results: use a straight, well-lit image with dark handwriting on a light background.")
    else:
        raw_bytes = uploaded_line.getvalue()
        upload_id = hashlib.sha256(raw_bytes + input_kind.encode()).hexdigest()
        try:
            line_image = Image.open(uploaded_line).convert("L")
        except Exception as exc:
            st.error(f"Could not open the image: {exc}")
            st.stop()

        with st.expander(f"Preview uploaded image · {line_image.width} × {line_image.height} px", expanded=True):
            st.image(line_image, use_container_width=True)

        recognize_clicked = st.button("Recognize handwriting", type="primary", use_container_width=True, key="iam_recognize")
        clear_clicked = st.button("Start over", use_container_width=False, key="iam_clear")
        if clear_clicked:
            st.session_state.pop("iam_result", None)
            st.rerun()

        if recognize_clicked:
            with st.spinner("Reading handwriting…"):
                if input_kind == "Single text line":
                    line_crops = [line_image]
                else:
                    gray = ImageOps.autocontrast(line_image)
                    arr = np.asarray(gray, dtype=np.uint8)
                    # Support dark ink on light paper and inverted scans.
                    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
                    if float(np.median(border)) < 127:
                        arr = 255 - arr
                    # A stronger adaptive row threshold suppresses isolated noise,
                    # which previously joined neighbouring text lines into one band.
                    ink = arr < 205
                    row_ink = ink.sum(axis=1).astype(float)
                    smooth = gaussian_filter1d(row_ink, sigma=1.8)
                    threshold = max(2.0, arr.shape[1] * 0.0035)
                    active = smooth >= threshold
                    # Fill only tiny gaps inside a stroke, never normal inter-line space.
                    active = ndimage.binary_closing(active, structure=np.ones(3, dtype=bool))
                    indices = np.flatnonzero(active)
                    groups = []
                    if len(indices):
                        group_start = previous = int(indices[0])
                        max_gap = max(5, int(arr.shape[0] * 0.008))
                        for y in indices[1:]:
                            y = int(y)
                            if y - previous > max_gap:
                                if previous - group_start >= 5:
                                    groups.append((group_start, previous + 1))
                                group_start = y
                            previous = y
                        if previous - group_start >= 5:
                            groups.append((group_start, previous + 1))

                    # If ink strokes connect vertically, use row-projection peaks
                    # to estimate separate text baselines and cut at the valleys.
                    if len(groups) <= 2:
                        min_distance = max(20, int(arr.shape[0] * 0.065))
                        peaks, _ = find_peaks(
                            smooth, distance=min_distance,
                            prominence=max(2.0, float(smooth.max()) * 0.035),
                        )
                        if len(peaks) >= 3:
                            cuts = []
                            for left, right in zip(peaks[:-1], peaks[1:]):
                                valley = int(left + np.argmin(smooth[left:right + 1]))
                                cuts.append(valley)
                            bounds = [0] + cuts + [arr.shape[0]]
                            peak_groups = []
                            for top, bottom in zip(bounds[:-1], bounds[1:]):
                                if bottom - top < 12:
                                    continue
                                local = smooth[top:bottom]
                                if local.size and local.max() >= threshold:
                                    peak_groups.append((top, bottom))
                            if len(peak_groups) > len(groups):
                                groups = peak_groups

                    line_crops = []
                    for y0, y1 in groups:
                        region = ink[y0:y1]
                        cols = np.flatnonzero(region.any(axis=0))
                        if not len(cols):
                            continue
                        pad_x = max(5, int(arr.shape[1] * 0.006))
                        pad_y = max(4, int(arr.shape[0] * 0.006))
                        top, bottom = max(0, y0 - pad_y), min(arr.shape[0], y1 + pad_y)
                        left, right = max(0, int(cols[0]) - pad_x), min(arr.shape[1], int(cols[-1]) + 1 + pad_x)
                        if bottom - top >= 10 and right - left >= 20:
                            line_crops.append(Image.fromarray(arr[top:bottom, left:right]))

                if not line_crops:
                    st.session_state["iam_result"] = {
                        "upload_id": upload_id, "crops": [], "predictions": [],
                        "error": "I couldn't find text lines. Try Single text line or a clearer, closer image.",
                    }
                else:
                    predictions = [predict_line(model, crop, device).strip() for crop in line_crops]
                    st.session_state["iam_result"] = {
                        "upload_id": upload_id, "crops": line_crops,
                        "predictions": predictions, "error": None,
                    }

        result = st.session_state.get("iam_result")
        if result and result.get("upload_id") == upload_id:
            if result.get("error"):
                st.warning(result["error"])
            else:
                predictions = result["predictions"]
                recognized = "\\n".join(predictions)
                st.markdown('<div class="iam-assistant"><div class="iam-result-label">Recognized text</div>', unsafe_allow_html=True)
                st.caption(f"{len(predictions)} text line(s) detected")
                if not recognized.strip():
                    st.warning("The model returned no text. Try a clearer crop containing one complete line.")
                text_key = f"iam_text_{upload_id[:16]}"
                edited_text = st.text_area(
                    "Edit the result", value=recognized, height=max(150, min(420, 52 * max(1, len(predictions)))),
                    key=text_key, label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.download_button(
                    "Download text", data=edited_text, file_name="recognized_handwriting.txt",
                    mime="text/plain", use_container_width=True,
                )
                with st.expander(f"Show the {len(result['crops'])} detected line crops"):
                    for number, (crop, line_text) in enumerate(zip(result["crops"], predictions), start=1):
                        st.markdown(f"**Line {number}**")
                        st.image(crop, use_container_width=True)
                        if line_text:
                            st.caption(line_text)

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

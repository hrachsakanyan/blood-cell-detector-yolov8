"""Blood Cell Detector - Streamlit app.

Detects and counts RBC, WBC and Platelets in blood smear images
using a YOLOv8n model trained on BCCD (models/best.pt).

Run:
    streamlit run app.py
"""

import io
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

st.set_page_config(page_title="Blood Cell Detector", page_icon="🩸", layout="wide")
st.title("🩸 Blood Cell Detector")
st.caption("YOLOv8n trained on BCCD - RBC, WBC and Platelets. Educational use only, not a medical device.")


@st.cache_resource
def load_model(path: str, mtime: float) -> YOLO:
    # mtime is part of the cache key so a retrained model is reloaded
    return YOLO(path)


if not MODEL_PATH.is_file():
    st.error(
        "Trained model not found at `models/best.pt`.\n\n"
        "Train it first with `python train.py --data path/to/data.yaml`, "
        "or copy a `best.pt` trained in Google Colab into the `models/` folder."
    )
    st.stop()

model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
uploaded = st.file_uploader("Upload a blood smear image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

with st.spinner("Detecting cells..."):
    result = model.predict(image, conf=conf, verbose=False)[0]

counts = Counter(result.names[int(c)] for c in result.boxes.cls.tolist())
annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB

left, right = st.columns(2)
left.subheader("Original")
left.image(image, width="stretch")
right.subheader("Detections")
right.image(annotated, width="stretch")

buffer = io.BytesIO()
annotated.save(buffer, format="PNG")
right.download_button(
    "Download annotated image",
    data=buffer.getvalue(),
    file_name=f"{Path(uploaded.name).stem}_detections.png",
    mime="image/png",
)

st.subheader("Cell counts")
names = list(result.names.values())
for col, name in zip(st.columns(len(names) + 1), names + ["Total"]):
    col.metric(name, sum(counts.values()) if name == "Total" else counts.get(name, 0))

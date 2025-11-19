# Python In-built packages
from pathlib import Path
from datetime import datetime
import time
import PIL

# External packages
import streamlit as st
import pandas as pd

# Local modules
import settings
import helper

st.set_page_config(
    page_title="WasteVision Studio",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

GLOBAL_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #140032 0%, #1a0f3c 45%, #05020d 100%);
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background: rgba(8, 9, 20, 0.95);
        border-right: 1px solid rgba(236, 72, 153, 0.25);
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    .glass-card {
        border-radius: 18px;
        padding: 1rem 1.2rem;
        background: rgba(30, 8, 50, 0.6);
        border: 1px solid rgba(244, 114, 182, 0.35);
        box-shadow: 0 30px 60px rgba(2, 6, 23, 0.4);
        backdrop-filter: blur(18px);
    }
    .glass-card h4 {
        margin-bottom: .35rem;
        font-size: .9rem;
        color: #fbcfe8;
        letter-spacing: .04em;
    }
    .glass-card p {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 600;
        color: #fde4ff;
    }
    .chip-stack span {
        display: inline-block;
        padding: .3rem .85rem;
        border-radius: 999px;
        margin: 0 .3rem .3rem 0;
        font-size: .85rem;
        color: #ffe4f1;
        border: 1px solid rgba(251, 113, 133, 0.4);
        background: linear-gradient(120deg, rgba(236,72,153,.4), rgba(244,114,182,.25));
    }
    .workspace-container {
        border-radius: 22px;
        padding: 1.5rem;
        background: rgba(12, 6, 28, 0.7);
        border: 1px solid rgba(236, 72, 153, 0.2);
        box-shadow: inset 0 0 0 1px rgba(236,72,153,0.12);
    }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_detection" not in st.session_state:
        st.session_state.latest_detection = None


def add_detection_to_history(labels):
    if not labels:
        return
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.history.append({"time": timestamp, **counts})


def render_history():
    if not st.session_state.history:
        st.info("No detection history yet. Upload an image and run inference.")
        return
    df = pd.DataFrame(st.session_state.history).fillna(0)
    st.dataframe(df, use_container_width=True)


def render_status_badge(text, variant):
    st.markdown(
        f"""
        <div style="
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .25rem .85rem;
            font-size: .8rem;
            color: white;
            background: {variant};
        ">
            <span style="font-size:1rem;">●</span>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_stack(labels):
    if not labels:
        return '<span style="opacity:.6;">Run an image to populate this panel.</span>'
    chips = "".join(f"<span>{label}</span>" for label in labels)
    return f'<div class="chip-stack">{chips}</div>'


init_state()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
    st.session_state.model_loaded = True
except Exception as ex:
    st.session_state.model_loaded = False
    st.error("Model failed to load.")
    st.error(ex)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Control Panel")
confidence = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.45, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### Live status")
if st.session_state.get("model_loaded"):
    render_status_badge("Model online", "#16a34a")
else:
    render_status_badge("Model offline", "#dc2626")
st.sidebar.markdown("---")
st.sidebar.markdown("### Tips")
st.sidebar.write(
    "- Use sharp, well-lit images for best accuracy.\n"
    "- Lower the threshold if few detections are appearing.\n"
    "- Switch to the Insights tab to track class frequencies."
)

# ---------------------------------------------------------------------------
# Hero / summary
# ---------------------------------------------------------------------------
hero_col, stat_col = st.columns([2.2, 1.2])
with hero_col:
    st.markdown(
        """
        <div style="padding:2rem;border-radius:20px;background:linear-gradient(120deg,#f43f5e,#7c3aed);color:white;">
            <div style="font-size:2.2rem;font-weight:600;margin-bottom:.5rem;">WasteVision Studio</div>
            <p style="font-size:1rem;opacity:.92;">
                Dynamic workspace for YOLOv8 waste-classification. Upload or capture frames,
                tweak inference settings, and watch the system highlight recyclable streams in real time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_col:
    metric_cols = st.columns(2)
    metric_cols[0].markdown(
        f"""
        <div class="glass-card">
            <h4>Confidence</h4>
            <p>{int(confidence*100)}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown(
        """
        <div class="glass-card">
            <h4>Mode</h4>
            <p>Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Recent classes")
    st.markdown(
        render_chip_stack(st.session_state.latest_detection),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_studio, tab_camera, tab_insights = st.tabs(
    ["🎛 Workspace", "📷 Live Camera", "📊 Insights"]
)

with tab_studio:
    st.markdown('<div class="workspace-container">', unsafe_allow_html=True)
    left, right = st.columns([1.1, 1.6], gap="large")

    with left:
        st.markdown("#### Upload center")
        source_img = st.file_uploader(
            "Drop image here", type=("jpg", "jpeg", "png", "bmp", "webp")
        )
        st.caption("Need inspiration? switch to default preview on the right.")
        run_button = st.button("Run YOLOv8", use_container_width=True)

    with right:
        st.markdown("#### Visual output")
        placeholder = st.empty()
        if source_img is None:
            placeholder.image(
                str(settings.DEFAULT_IMAGE),
                caption="Default sample",
                use_column_width=True,
            )
        else:
            uploaded_img = PIL.Image.open(source_img)
            placeholder.image(uploaded_img, caption="Uploaded image", use_column_width=True)

    labels = []
    if run_button:
        if source_img is None:
            st.warning("Upload an image before running inference.")
        elif not st.session_state.get("model_loaded"):
            st.error("Model unavailable. Check weights path in settings.")
        else:
            with st.spinner("Detecting recyclable material..."):
                start_time = time.time()
                res = model.predict(uploaded_img, conf=confidence)
                boxes = res[0].boxes
                result_img = res[0].plot()[:, :, ::-1]
                duration = time.time() - start_time
                placeholder.image(result_img, caption="Detection result", use_column_width=True)
                labels = []
                for box in boxes:
                    class_id = int(box.cls[0])
                    labels.append(model.names[class_id])
                st.session_state.latest_detection = labels
                add_detection_to_history(labels)
            if labels:
                st.success(f"Detected: {', '.join(labels)}")
                total = len(labels)
                unique = len(set(labels))
                summary_cols = st.columns(3)
                summary_cols[0].markdown(
                    f'<div class="glass-card"><h4>Total detections</h4><p>{total}</p></div>',
                    unsafe_allow_html=True,
                )
                summary_cols[1].markdown(
                    f'<div class="glass-card"><h4>Unique classes</h4><p>{unique}</p></div>',
                    unsafe_allow_html=True,
                )
                summary_cols[2].markdown(
                    f'<div class="glass-card"><h4>Runtime</h4><p>{duration:.2f}s</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No objects detected. Try reducing the confidence threshold.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_insights:
    st.markdown("#### Session insights")
    render_history()
    st.markdown("##### Reset session")
    if st.button("Clear history", type="secondary"):
        st.session_state.history = []
        st.session_state.latest_detection = None
        st.success("History cleared.")

with tab_camera:
    st.markdown("#### Live camera detection")
    st.write(
        "Use your webcam for real-time detections. Once you allow camera permissions, "
        "return to the sidebar and press **Detect Trash** to start streaming. You can "
        "toggle tracking inside the sidebar controls that appear when the webcam starts running."
    )
    helper.play_webcam(confidence, model)


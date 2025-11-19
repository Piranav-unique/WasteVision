# Python In-built packages
from pathlib import Path
from datetime import datetime
import time
import PIL

# External packages
import streamlit as st

# Local modules
import settings
import helper

st.set_page_config(
    page_title="WasteVision Nexus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

GLOBAL_STYLES = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #020617 0%, #0f172a 45%, #020617 100%);
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background: rgba(2, 6, 23, 0.94);
        border-right: 1px solid rgba(248, 250, 252, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    .glass-card {
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 25px 45px rgba(2, 6, 23, 0.5);
        backdrop-filter: blur(18px);
    }
    .metric-box h4 {
        margin: 0;
        font-size: .85rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #a5b4fc;
    }
    .metric-box p {
        margin: .2rem 0 0;
        font-size: 1.9rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        padding: .25rem .7rem;
        border-radius: 999px;
        font-size: .85rem;
        margin: 0 .35rem .35rem 0;
        background: linear-gradient(120deg, rgba(14,165,233,.25), rgba(99,102,241,.2));
        border: 1px solid rgba(148,163,184,.35);
    }
    .workspace {
        border-radius: 24px;
        padding: 1.5rem;
        margin-top: 1rem;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
"""
st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)

if "recent_labels" not in st.session_state:
    st.session_state.recent_labels = []


def render_badge(text, color):
    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            gap:.4rem;
            align-items:center;
            padding:.35rem .9rem;
            border-radius:999px;
            background:{color};
            color:#020617;
            font-weight:600;
        ">
            <span style="font-size:1.1rem;">●</span>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar -------------------------------------------------------------------
st.sidebar.markdown("## Control Deck")
confidence = st.sidebar.slider("Confidence", 0.2, 0.95, 0.45, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")

model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
    render_badge("Model online", "#86efac")
except Exception as ex:
    render_badge("Model offline", "#fca5a5")
    st.sidebar.error(f"Unable to load model from `{model_path}`")
    st.sidebar.error(ex)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Tips")
st.sidebar.write(
    "- Prefer bright, clutter-free pictures.\n"
    "- Drop the slider if objects are missed.\n"
    "- Switch tabs for live webcam monitoring."
)

# Hero ----------------------------------------------------------------------
hero_col, metric_col = st.columns([2.3, 1.7])
with hero_col:
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-size:2.3rem;font-weight:700;margin-bottom:.4rem;">
                WasteVision Nexus
            </div>
            <p style="opacity:.9;font-size:1rem;">
                Monitor recyclable streams with YOLOv8. Upload images, tune inference settings,
                or jump into the live lab to keep sorting lines informed in real time.
            </p>
            <div style="display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap;">
                <div class="chip">Updated {datetime.now().strftime('%d %b')}</div>
                <div class="chip">Ultralytics YOLOv8</div>
                <div class="chip">Custom weights</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col:
    metric_box = st.columns(2)
    metric_box[0].markdown(
        f"""
        <div class="glass-card metric-box">
            <h4>Confidence</h4>
            <p>{int(confidence*100)}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_box[1].markdown(
        """
        <div class="glass-card metric-box">
            <h4>Mode</h4>
            <p>Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Recent labels")
    if st.session_state.recent_labels:
        st.markdown(
            "".join(f"<span class='chip'>{label}</span>" for label in st.session_state.recent_labels),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No detections yet. Upload an image to get started.")

# Tabs ----------------------------------------------------------------------
tab_visual, tab_live = st.tabs(["🧪 Visual Lab", "🎥 Live Cam"])

# Visual Lab ----------------------------------------------------------------
with tab_visual:
    st.markdown('<div class="workspace">', unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.4], gap="large")

    with left:
        st.subheader("Upload centre")
        source_img = st.file_uploader(
            "Drop or browse a file",
            type=("jpg", "jpeg", "png", "bmp", "webp"),
            label_visibility="collapsed",
        )
        st.caption("Tip: Aim for ≥640px resolution for richer detections.")
        detect_btn = st.button("Run detection", use_container_width=True)

    with right:
        st.subheader("Preview")
        canvas = st.empty()
        if source_img is None:
            canvas.image(
                str(settings.DEFAULT_IMAGE),
                caption="Default sample",
                use_column_width=True,
            )
        else:
            uploaded_image = PIL.Image.open(source_img)
            canvas.image(uploaded_image, caption="Uploaded image", use_column_width=True)

    if detect_btn:
        if source_img is None:
            st.warning("Please upload an image.")
        else:
            with st.spinner("Running YOLOv8..."):
                start = time.time()
                res = model.predict(uploaded_image, conf=confidence)
                elapsed = time.time() - start
                boxes = res[0].boxes
                rendered = res[0].plot()[:, :, ::-1]
                canvas.image(rendered, caption="Detection result", use_column_width=True)

                labels = []
                for box in boxes:
                    class_id = int(box.cls[0])
                    labels.append(model.names[class_id])
                st.session_state.recent_labels = labels

            if labels:
                st.success(f"Detected: {', '.join(labels)}")
                stat_cols = st.columns(3)
                stat_cols[0].metric("Objects", len(labels))
                stat_cols[1].metric("Unique labels", len(set(labels)))
                stat_cols[2].metric("Runtime", f"{elapsed:.2f}s")
                with st.expander("Detection data"):
                    for idx, box in enumerate(boxes):
                        st.write(f"#{idx+1}", box.data)
            else:
                st.info("No objects detected. Consider lowering the confidence slider.")
    st.markdown("</div>", unsafe_allow_html=True)

# Live Cam ------------------------------------------------------------------
with tab_live:
    st.subheader("Live monitoring")
    st.write(
        "Allow camera permissions and click **Detect Trash** from the sidebar helper "
        "to start streaming. Tracking toggles appear when the webcam launches."
    )
    helper.play_webcam(confidence, model)

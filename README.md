# WasteVision Nexus

WasteVision Nexus is a Streamlit front-end that wraps a custom YOLOv8 model to monitor recyclable waste streams. The dashboard exposes a polished “visual lab” for single-image experimentation and a live webcam tab for real-time monitoring, making it easier to spot contamination or recoverable material without wiring up a full MLOps stack.

## Highlights
- **YOLOv8 detections** with support for custom weights stored under `src/weights/`.
- **Streamlit UX polish**: glassmorphic theme, status badges, confidence metrics, tabbed workspace.
- **Multiple sources**: upload still images, switch to webcam mode, or extend to stored/YouTube videos via `helper.py`.
- **Lightweight helper API** for loading `.pt` or `.pkl` checkpoints and enabling trackers such as ByteTrack/BOTSort.

## Project Structure
```
yolov8/
├── src/
│   ├── app.py              # Main Streamlit app (visual lab + live cam)
│   ├── app_dynamic.py      # Alternate playground (experimental)
│   ├── helper.py           # Model + tracker helpers & streaming utilities
│   ├── settings.py         # Paths, source constants, webcam/video config
│   ├── requirements.txt    # Full Python dependency lock
│   ├── images/             # Placeholder visuals & defaults
│   └── weights/            # Custom YOLO weights (tracked for now)
└── results/                # Reserved for export artifacts
```

## Getting Started
1. **Clone & enter the repo**
   ```powershell
   git clone https://github.com/Piranav-unique/WasteVision.git
   cd WasteVision\yolov8
   ```
2. **Create an environment (recommended)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Install requirements**
   ```powershell
   pip install -r src\requirements.txt
   ```
4. **Verify model paths**  
   Update `DETECTION_MODEL` inside `src/settings.py` if your weights live outside `src/weights/`. `helper.load_model` will automatically switch to `yoloooo.pt` when a `.pkl` pointer is supplied.

## Running the App
```powershell
cd src
streamlit run app.py
```
- The sidebar slider controls confidence; drop it when detections are missed.
- Use **Run detection** after uploading a still image, or hop to the **Live Cam** tab and press **Detect Trash** to start webcam inference.
- Recent class labels are cached per session (`st.session_state.recent_labels`).

## Working With Weights
- Current repo ships several sample checkpoints (`best.pt`, `yoloooo.pt`, `yolov8n.pt`, and a pickled reference). For larger/final models consider moving them to cloud/object storage and referencing download scripts to keep the repo lean.
- To swap models at runtime, point `settings.DETECTION_MODEL` to the desired `.pt/.pkl` file and restart Streamlit.

## Extending
- **Video/YouTube feeds**: `helper.py` already includes utilities (`play_stored_video`, `play_youtube_video`) you can expose via new Streamlit tabs.
- **Tracking overlays**: Enable trackers from the sidebar radio when using video sources for smoother ID persistence.
- **Deployment**: Package with `streamlit run app.py --server.port 8501` inside a container or deploy via Streamlit Community Cloud; make sure model files are available at startup.

## Roadmap Ideas
- Add a `.gitignore` (especially for `__pycache__` and heavy weights).
- Automate weight downloads (e.g., Hugging Face or Azure Blob) and document training data.
- Capture inference metrics in `results/` for auditing.
- Add tests or linting checks for helper utilities.

Feel free to adapt this README as the project evolves—add architecture diagrams, dataset citations, or demo GIFs once you have them.


import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np
import av

# Page Design
st.set_page_config(page_title="On-Road Threat Detection AI", layout="wide")
st.title("🛡️ On-Road Threat Detection & Action Radar Engine")
st.markdown("Equipped with real-time collision boundaries, velocity tracing, and drop-frame lag optimization.")

# 1. Load Lightweight YOLO Engine (YOLOv8 nano optimized for low-latency CPU/GPU execution)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    try:
        import torch
        if torch.cuda.is_available():
            model.to("cuda")
    except ImportError:
        pass
    return model

model = load_model()

# 2. Control Metrics
st.sidebar.title("🚨 Threat Boundaries")
conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.20, 1.0, 0.35, 0.05)
proximity_threshold = st.sidebar.slider("Danger Zone Proximity (Pixels from Bottom)", 50, 250, 150, 10)
frame_skip_rate = st.sidebar.slider("Frame Dropper Rate (Inference Frequency)", 1, 4, 2, 1)

# Persistent storage across instances for speed calculation
if "radar_history" not in st.session_state:
    st.session_state.radar_history = {}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ThreatDetectionProcessor:
    def __init__(self):
        self.frame_index = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_index += 1
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # --- LAG DEFENSE ---
        # Instantly bypass the heavy ML logic for intermediate frames to prevent video freezing
        if self.frame_index % frame_skip_rate != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Static HUD Layer: Draw Threat Proximity Boundary Overlay (Danger Zone)
        danger_line_y = h - proximity_threshold
        # Draw translucent red boundary overlay
        overlay_mask = img.copy()
        cv2.rectangle(overlay_mask, (0, danger_line_y), (w, h), (0, 0, 80), -1)
        cv2.addWeighted(overlay_mask, 0.25, img, 0.75, 0, img)
        cv2.line(img, (0, danger_line_y), (w, danger_line_y), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "CRITICAL COLLISION BOUNDARY", (15, danger_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Run AI Tracking (Targeting Persons=0, Cars=2, Bikes=3, Buses=5, Trucks=7)
        results = model.track(source=img, persist=True, conf=conf_threshold, classes=[0,2,3,5,7], verbose=False)

        global_threat_level = "CLEAR"

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                label = names[class_id].upper()

                # Heuristic 1: Track Action (Movement Delta)
                action_status = "STABLE"
                if track_id in st.session_state.radar_history:
                    pcx, pcy = st.session_state.radar_history[track_id]
                    movement_delta = np.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                    if movement_delta > 30: # Custom pixel motion limit
                        action_status = "ERRATIC/FAST"
                
                st.session_state.radar_history[track_id] = [cx, cy]

                # Heuristic 2: Threat Evaluation (Proximity Check)
                # If the base of the object bounding box passes below the warning line, it's a threat
                is_breaching = y2 > danger_line_y
                
                # Style determination based on critical threats
                if is_breaching:
                    border_color = (0, 0, 255)  # Serious Red
                    threat_label = "🚨 CRITICAL THREAT: COLLISION RISK"
                    global_threat_level = "CRITICAL"
                elif action_status == "ERRATIC/FAST":
                    border_color = (0, 165, 255)  # Orange
                    threat_label = "⚠️ WARNING: RECKLESS MOVE"
                    if global_threat_level != "CRITICAL":
                        global_threat_level = "WARNING"
                else:
                    border_color = (0, 255, 0)  # Neon Green
                    threat_label = f"ID {track_id} | SAFE"

                # Render High-Visibility Geometric Shapes
                cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 4, border_color, -1)

                # Floating UI Tag text compilation
                ui_text = f"[{label}] {threat_label}"
                cv2.rectangle(img, (x1, y1 - 22), (x1 + len(ui_text)*8, y1), border_color, -1)
                cv2.putText(img, ui_text, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Global HUD Status banner at top left corner of the feed
        hud_color = (0, 0, 255) if global_threat_level == "CRITICAL" else ((0, 140, 255) if global_threat_level == "WARNING" else (0, 200, 0))
        cv2.rectangle(img, (10, 10), (280, 45), hud_color, -1)
        cv2.putText(img, f"SYSTEM RADAR: {global_threat_level}", (20, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Clear memory registry periodically
        if len(st.session_state.radar_history) > 200:
            st.session_state.radar_history.clear()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Stream Engine Mount
webrtc_streamer(
    key="anti-freeze-threat-radar",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ThreatDetectionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True,
)
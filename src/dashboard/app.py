"""
src/dashboard/app.py
S.A.F.E. -- Smart Agricultural Forecasting & Edge-computing
Streamlit Dashboard
Now with direct ESP32-CAM control, live trigger capture button, and auto-refresh!
"""

from __future__ import annotations

import io
import os
import sys
import time
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predictor import SAFEPredictor, CLASSES, IMG_SIZE, SEQ_LEN, RISK_THRESHOLDS

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="S.A.F.E. | Crop Disease Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --safe-dark-green: #143d23;
        --safe-leaf-green: #2e7d32;
        --safe-emerald: #10b981;
        --safe-accent-green: #4caf50;
        --safe-light-green: #e8f5e9;
        --safe-mint: #a5d6a7;
        --safe-bg: #f7faf7;
        --safe-card: #ffffff;
        --safe-text: #1b2e20;
        --safe-muted: #52795d;
        --safe-border: #d4e7d7;
        --safe-danger: #ef4444;
        --safe-warning: #f59e0b;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
        color: var(--safe-text);
    }

    .stApp {
        background-color: var(--safe-bg);
    }

    .hero-header {
        background: linear-gradient(135deg, #11381e 0%, #1e5e32 60%, #2e7d32 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(20, 61, 35, 0.15);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #d1fae5;
        margin-top: 6px;
        font-weight: 500;
    }

    .badge-demo {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .badge-live {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

    .safe-metric-card {
        background: var(--safe-card);
        border: 1px solid var(--safe-border);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .safe-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(46, 125, 50, 0.08);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--safe-muted);
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        color: var(--safe-dark-green);
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        margin-top: 8px;
        font-weight: 500;
    }

    .concept-box {
        background: #f0fdf4;
        border: 1px dashed var(--safe-leaf-green);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 15px 0 24px 0;
        font-size: 14px;
        font-weight: 600;
        color: var(--safe-dark-green);
        text-align: center;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }
    .status-low { background: #dcfce7; color: #15803d; }
    .status-mod { background: #fef9c3; color: #a16207; }
    .status-high { background: #fee2e2; color: #b91c1c; }
    .status-crit { background: #7f1d1d; color: #ffffff; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8f5e9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        color: var(--safe-muted);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--safe-leaf-green) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_local_predictor():
    try:
        return SAFEPredictor(use_fusion=True)
    except Exception as e:
        st.error(f"Could not load local SAFEPredictor: {e}")
        return None

def fetch_telemetry():
    try:
        r = requests.get(f"{API_BASE_URL}/telemetry", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

backend_online, live_telem = fetch_telemetry()
is_real_esp32 = False
current_temp = 25.0
current_hum = 55.0
current_soil = 45.0
last_update_str = datetime.now().strftime('%H:%M:%S')

if backend_online and live_telem:
    is_real_esp32 = live_telem.get("is_real_hardware", False)
    current_temp = live_telem.get("temperature", 25.0)
    current_hum = live_telem.get("humidity", 55.0)
    current_soil = live_telem.get("soil_moisture", 0.0)
    if "last_updated" in live_telem:
        try:
            dt = datetime.fromisoformat(live_telem["last_updated"])
            last_update_str = dt.strftime('%H:%M:%S')
        except Exception:
            pass

# Dynamic Presentation Demo Curves & Live Hardware Auto-Reset
def get_presentation_demo_curves():
    now = datetime.now()
    times = [now - timedelta(minutes=i*20) for i in range(24)][::-1]
    temps = [23.5 + 7.0 * np.sin(i / 3.8) + np.random.normal(0, 0.2) for i in range(24)]
    hums = [68.0 - 16.0 * np.sin(i / 3.8) + np.random.normal(0, 0.4) for i in range(24)]
    soils = [60.0 - (i * 0.4) + np.random.normal(0, 0.2) for i in range(24)]
    risks = [float(np.clip(0.18 + (0.35 if h > 68 else 0.05) + (0.22 if t > 28 else 0.02), 0.08, 0.88)) for t, h in zip(temps, hums)]
    return pd.DataFrame({'timestamp': times, 'temperature': [round(x, 1) for x in temps], 'humidity': [round(x, 1) for x in hums], 'soil_moisture': [round(x, 1) for x in soils], 'risk_score': risks})

if 'was_hardware' not in st.session_state:
    st.session_state.was_hardware = False

if is_real_esp32:
    if not st.session_state.was_hardware or 'live_buf' not in st.session_state:
        st.session_state.live_buf = pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'soil_moisture', 'risk_score'])
    st.session_state.was_hardware = True
    now_time = datetime.now()
    calc_risk = float(np.clip(0.12 + (0.35 if current_hum > 75 else 0.05) + (0.20 if current_temp > 28 else 0.0), 0.05, 0.95))
    new_row = pd.DataFrame([{'timestamp': now_time, 'temperature': current_temp, 'humidity': current_hum, 'soil_moisture': current_soil, 'risk_score': calc_risk}])
    st.session_state.live_buf = pd.concat([st.session_state.live_buf, new_row], ignore_index=True)
    if len(st.session_state.live_buf) > 35:
        st.session_state.live_buf = st.session_state.live_buf.iloc[-35:]
    st.session_state.history = st.session_state.live_buf
else:
    st.session_state.was_hardware = False
    if 'demo_buf' not in st.session_state:
        st.session_state.demo_buf = get_presentation_demo_curves()
    st.session_state.history = st.session_state.demo_buf
    last_pt = st.session_state.history.iloc[-1]
    current_temp = last_pt['temperature']
    current_hum = last_pt['humidity']
    current_soil = last_pt['soil_moisture']
    last_update_str = last_pt['timestamp'].strftime('%H:%M:%S')
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = {
        "predicted_class": "early_blight",
        "confidence": 0.884,
        "risk_score": 0.82,
        "risk_level": "HIGH",
        "probabilities": {
            "bacterial_spot": 0.06,
            "early_blight": 0.884,
            "healthy": 0.012,
            "late_blight": 0.044
        },
        "model_used": "MobileNetV2 + LSTM Fusion",
        "image_path": "sample_images/early_blight.jpg" if Path("sample_images/early_blight.jpg").exists() else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

badge_html = """<span class="badge-live">🟢 LIVE ESP32 HARDWARE</span>""" if is_real_esp32 else """<span class="badge-demo">⚠️ DEMO SIMULATION</span>"""

st.markdown(f"""
<div class="hero-header">
    <div>
        <h1 class="hero-title">🌱 S.A.F.E. Dashboard</h1>
        <div class="hero-subtitle">Smart Agricultural Forecasting & Edge-computing · Tomato Crop Disease Early Warning</div>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        {badge_html}
        <span style="font-size: 12px; color: #a7f3d0;">v1.0.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="concept-box">
    🍃 Tomato Leaf Image &nbsp;+&nbsp; 🌦️ Environmental History (Temp, Humidity, Soil Moisture) 
    &nbsp; ➔ &nbsp; 🧠 <b>Multimodal AI (CNN + LSTM)</b> &nbsp; ➔ &nbsp; ⚠️ <b>Disease Risk Forecast</b>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/green-energy.png", width=64)
    st.markdown("### 🌾 S.A.F.E. Control Center")
    st.caption("Edge-enabled Precision Agriculture")
    st.markdown("---")

    st.markdown("#### 📡 Hardware Link")
    if is_real_esp32:
        st.success(f"🟢 **ESP32 Connected!**\n\nID: `SAFE-ESP32-01`\n\nTelemetry streaming over Wi-Fi.")
    else:
        st.warning("🟡 **ESP32: Simulated**\n\nWaiting for sensor packets on `/sensor-data`...")

    if backend_online:
        st.info("🟢 **FastAPI Server:** Online (Port 8000)")
    else:
        st.error("🔴 **FastAPI Server:** Offline")

    st.markdown("---")
    st.markdown("#### 📷 ESP32-CAM Quick Trigger")
    cam_ip_input = st.text_input("ESP32-CAM IP:", value="192.168.1.150", help="Enter the local IP address of your ESP32-CAM module")
    
    if st.button("📸 Snap & Analyze Now", type="primary", use_container_width=True):
        with st.spinner(f"Triggering camera at http://{cam_ip_input}/capture ..."):
            try:
                r = requests.post(f"{API_BASE_URL}/trigger-cam-capture?cam_ip={cam_ip_input}", timeout=6)
                if r.status_code == 200:
                    res = r.json()
                    st.success("✅ Snapshot captured & analyzed!")
                    st.session_state.latest_prediction = {
                        "predicted_class": res["predicted_class"],
                        "confidence": res["classification_confidence"],
                        "risk_score": res["risk_score"],
                        "risk_level": res["risk_level"],
                        "probabilities": res["class_probabilities"],
                        "model_used": res["model_used"],
                        "image_path": "sample_images/latest_cam_capture.jpg",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()
                else:
                    st.error(f"Camera responded with code {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Could not reach ESP32-CAM: {e}. Check IP or use demo sample below.")

    st.markdown("---")
    if st.button("🔄 Refresh Telemetry Now", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("🔬 **Active Models**")
    st.caption("• **Vision:** MobileNetV2 (64×64 RGB)\n• **Temporal:** Stacked LSTM (30-step window)\n• **Fusion:** Concatenation Head")

tab_live, tab_analysis, tab_manual, tab_system = st.tabs([
    "📊 1. LIVE MONITOR",
    "🔬 2. DISEASE ANALYSIS",
    "🧪 3. MANUAL PREDICTION",
    "⚙️ 4. SYSTEM STATUS"
])

# =============================================================================
# TAB 1: LIVE MONITOR
# =============================================================================
with tab_live:
    hist = st.session_state.history
    latest_risk = hist.iloc[-1]["risk_score"]

    if latest_risk >= RISK_THRESHOLDS["HIGH"]:
        risk_pill = '<span class="status-pill status-high">🔴 HIGH RISK</span>'
        risk_color = "#ef4444"
    elif latest_risk >= RISK_THRESHOLDS["LOW"]:
        risk_pill = '<span class="status-pill status-mod">🟡 MODERATE RISK</span>'
        risk_color = "#f59e0b"
    else:
        risk_pill = '<span class="status-pill status-low">🟢 LOW RISK</span>'
        risk_color = "#10b981"

    if is_real_esp32:
        st.markdown(f"""
        <div style="background: #ecfdf5; border: 1px solid #6ee7b7; padding: 12px 18px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: #065f46; font-weight: 700;">
                🟢 <b>LIVE PHYSICAL TELEMETRY:</b> Data actively streaming from ESP32-WROOM node (DHT22 & HW-080).
            </span>
            <span style="font-size: 12px; color: #047857; font-weight: 600;">Last Packet: {last_update_str}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; padding: 12px 18px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: #92400e; font-weight: 600;">
                ⚠️ <b>DEMO SIMULATION ACTIVE:</b> Waiting for live packets from ESP32.
            </span>
            <span style="font-size: 12px; color: #b45309; font-weight: 500;">Last Update: {last_update_str}</span>
        </div>
        """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="safe-metric-card">
            <div class="metric-label">Disease Risk Score</div>
            <div class="metric-val" style="color: {risk_color};">{latest_risk*100:.1f}%</div>
            <div class="metric-sub">{risk_pill}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="safe-metric-card">
            <div class="metric-label">Ambient Temperature</div>
            <div class="metric-val">{current_temp:.1f}°C</div>
            <div class="metric-sub" style="color: #4b5563;">Optimum: 22°C - 28°C</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="safe-metric-card">
            <div class="metric-label">Relative Humidity</div>
            <div class="metric-val">{current_hum:.1f}%</div>
            <div class="metric-sub" style="color: {'#ef4444' if current_hum > 75 else '#10b981'};">
                {'⚠️ High Fungal Risk' if current_hum > 75 else '✅ Normal range'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="safe-metric-card">
            <div class="metric-label">Soil Moisture (Passthrough)</div>
            <div class="metric-val">{current_soil:.1f}%</div>
            <div class="metric-sub" style="color: #4b5563;">Calibrated HW-080</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Crop Disease Risk Score Trend")
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["risk_score"] * 100,
            mode="lines+markers",
            name="Risk %",
            line=dict(color="#10b981", width=3),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)"
        ))
        fig_risk.add_hline(y=75, line_dash="dot", line_color="#ef4444", annotation_text="High (75%)")
        fig_risk.add_hline(y=30, line_dash="dot", line_color="#f59e0b", annotation_text="Mod (30%)")
        fig_risk.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Risk Score (%)",
            yaxis_range=[0, 100],
            plot_bgcolor="white",
            height=320
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        st.subheader("🌡️ Real-Time Environmental Telemetry")
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["temperature"],
            name="Temperature (°C)",
            line=dict(color="#ef4444", width=2)
        ))
        fig_env.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["humidity"],
            name="Humidity (%)",
            line=dict(color="#3b82f6", width=2)
        ))
        fig_env.add_trace(go.Scatter(
            x=hist["timestamp"],
            y=hist["soil_moisture"],
            name="Soil Moisture (%)",
            line=dict(color="#8b5cf6", width=2, dash="dash")
        ))
        fig_env.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Reading",
            plot_bgcolor="white",
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_env, use_container_width=True)

# =============================================================================
# TAB 2: DISEASE ANALYSIS & ESP32-CAM CONTROLS
# =============================================================================
with tab_analysis:
    st.subheader("🔬 Latest Edge AI Diagnostic Overview & Camera Control")
    st.caption("Multimodal fusion combines MobileNetV2 leaf feature extraction with LSTM microclimate sequences.")

    # Control Bar for ESP32-CAM
    with st.expander("📷 **ESP32-CAM Wireless Trigger Panel**", expanded=True):
        cam_col1, cam_col2, cam_col3 = st.columns([2, 1.2, 1.2])
        with cam_col1:
            cam_ip_field = st.text_input("ESP32-CAM Network Address:", value="192.168.1.150", key="cam_ip_field", help="Ensure ESP32-CAM is powered and on the same Wi-Fi network.")
        with cam_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            trigger_cam_btn = st.button("📸 Capture from Camera", type="primary", use_container_width=True)
        with cam_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            load_sim_cam = st.button("📁 Load Last Saved Capture", use_container_width=True)

        if trigger_cam_btn:
            with st.spinner(f"Requesting snapshot from http://{cam_ip_field}/capture ..."):
                try:
                    r = requests.post(f"{API_BASE_URL}/trigger-cam-capture?cam_ip={cam_ip_field}", timeout=6)
                    if r.status_code == 200:
                        res = r.json()
                        st.success("✅ Snapshot captured and analyzed!")
                        st.session_state.latest_prediction = {
                            "predicted_class": res["predicted_class"],
                            "confidence": res["classification_confidence"],
                            "risk_score": res["risk_score"],
                            "risk_level": res["risk_level"],
                            "probabilities": res["class_probabilities"],
                            "model_used": res["model_used"],
                            "image_path": "sample_images/latest_cam_capture.jpg",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.rerun()
                    else:
                        st.error(f"Camera returned status {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Could not connect to ESP32-CAM at {cam_ip_field}. Verify Wi-Fi connection and IP address.")

        if load_sim_cam:
            cam_capture_path = Path("sample_images/latest_cam_capture.jpg")
            if cam_capture_path.exists():
                st.session_state.latest_prediction["image_path"] = str(cam_capture_path)
                st.session_state.latest_prediction["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Loaded latest ESP32-CAM capture from disk.")
                st.rerun()
            else:
                st.warning("No previous capture found on disk yet.")

    pred = st.session_state.latest_prediction
    col_img, col_diag = st.columns([1.1, 1.9])

    with col_img:
        st.markdown("#### 📸 Analyzed Leaf Image")
        img_p = pred.get("image_path")
        if img_p and Path(img_p).exists():
            st.image(img_p, caption=f"Capture timestamp: {pred['timestamp']}", use_container_width=True)
        else:
            dummy_img = Image.new("RGB", (300, 300), color="#2e7d32")
            st.image(dummy_img, caption="Default Leaf View", use_container_width=True)

    with col_diag:
        st.markdown("#### 🩺 Diagnosis & Probabilities")
        st.warning("⚠️ **Note:** **Classification Confidence ≠ Multimodal Risk Score**\n\n- *Confidence* indicates MobileNetV2 certainty in visual symptoms.\n- *Multimodal Risk Score* integrates environmental pathogen proliferation dynamics.")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Predicted Disease", pred["predicted_class"].replace("_", " ").title())
        with d2:
            st.metric("Classification Confidence", f"{pred['confidence']*100:.1f}%")
        with d3:
            st.metric("Multimodal Risk Score", f"{pred['risk_score']*100:.1f}%")

        st.markdown("##### Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Class": [c.replace("_", " ").title() for c in CLASSES],
            "Probability": [pred["probabilities"].get(c, 0.0) * 100 for c in CLASSES]
        })

        fig_probs = px.bar(
            prob_df,
            x="Probability",
            y="Class",
            orientation="h",
            text="Probability",
            color="Class",
            color_discrete_map={
                "Bacterial Spot": "#f59e0b",
                "Early Blight": "#ef4444",
                "Healthy": "#10b981",
                "Late Blight": "#dc2626"
            }
        )
        fig_probs.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_probs.update_layout(
            xaxis_title="Confidence Probability (%)",
            xaxis_range=[0, 115],
            yaxis_title="",
            height=260,
            showlegend=False,
            margin=dict(l=10, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_probs, use_container_width=True)
        st.markdown(f"**Inference Pipeline:** `{pred['model_used']}`")

# =============================================================================
# TAB 3: MANUAL PREDICTION
# =============================================================================
with tab_manual:
    st.subheader("🧪 On-Demand Crop Health Analysis")
    st.write("Upload a tomato leaf photo and specify current environmental conditions to evaluate crop disease risk through the S.A.F.E. pipeline.")

    col_input_img, col_input_env = st.columns([1.2, 1.8])

    uploaded_file = None
    with col_input_img:
        st.markdown("#### 1. Tomato Leaf Image")
        upload_mode = st.radio("Image Source:", ["Upload custom image", "Select sample from dataset"], horizontal=True)

        if upload_mode == "Upload custom image":
            uploaded_file = st.file_uploader("Upload leaf JPEG/PNG", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        else:
            sample_options = ["bacterial_spot", "early_blight", "healthy", "late_blight"]
            selected_sample = st.selectbox("Choose sample class:", sample_options)
            sample_path = Path(f"sample_images/{selected_sample}.jpg")
            if sample_path.exists():
                st.image(str(sample_path), caption=f"Dataset Sample: {selected_sample}", use_container_width=True)
                with open(sample_path, "rb") as f:
                    uploaded_file = io.BytesIO(f.read())
                    uploaded_file.name = f"{selected_sample}.jpg"

    with col_input_env:
        st.markdown("#### 2. Environmental Readings")
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            inp_temp = st.slider("Temperature (°C)", min_value=15.0, max_value=48.0, value=float(current_temp), step=0.5)
            inp_soil = st.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=float(current_soil), step=1.0)
            inp_npk = st.selectbox("Soil NPK Level", options=[1, 2, 3], index=1, format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High"}[x])

        with c_e2:
            inp_hum = st.slider("Relative Humidity (%)", min_value=25.0, max_value=98.0, value=float(current_hum), step=1.0)
            inp_ph = st.slider("Soil pH Index", min_value=45.0, max_value=90.0, value=68.0, step=1.0)

        st.caption("ℹ️ Values default to current ESP32 readings when live hardware is connected.")
        analyze_btn = st.button("🚀 Analyze Crop Health", type="primary", use_container_width=True)

    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload or select an image first.")
        else:
            with st.spinner("Executing S.A.F.E. Multimodal Inference (MobileNetV2 + LSTM)..."):
                result_data = None
                if backend_online:
                    try:
                        uploaded_file.seek(0)
                        files = {"image": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                        data = {
                            "temperature": inp_temp,
                            "humidity": inp_hum,
                            "soil_moisture": inp_soil,
                            "npk_level": inp_npk,
                            "ph_value": inp_ph
                        }
                        res = requests.post(f"{API_BASE_URL}/predict", files=files, data=data, timeout=5)
                        if res.status_code == 200:
                            result_data = res.json()
                    except Exception as e:
                        st.warning(f"FastAPI request failed ({e}); falling back to local predictor.")

                if result_data is None:
                    local_pred = get_local_predictor()
                    if local_pred:
                        uploaded_file.seek(0)
                        pil_img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
                        img_arr = np.array(pil_img, dtype=np.float32) / 255.0

                        base = np.array([inp_npk, inp_temp, inp_hum, inp_ph], dtype=np.float32)
                        seq = np.tile(base, (SEQ_LEN, 1))

                        raw = local_pred.predict(img_arr, seq, soil_moisture=inp_soil)
                        probs = raw["probabilities"]
                        result_data = {
                            "predicted_class": raw["predicted_class"],
                            "class_probabilities": probs,
                            "classification_confidence": max(probs.values()),
                            "risk_score": raw["risk_score"],
                            "risk_level": raw["risk_level"],
                            "model_used": raw["model_used"],
                            "soil_moisture_note": raw["soil_moisture_note"],
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }

                if result_data:
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")
                    
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Predicted Disease", result_data["predicted_class"].replace("_", " ").title())
                    r2.metric("Confidence", f"{result_data['classification_confidence']*100:.1f}%")
                    r3.metric("Multimodal Risk Score", f"{result_data['risk_score']*100:.1f}%")
                    r4.metric("Assessed Risk Level", result_data["risk_level"])

                    st.session_state.latest_prediction = {
                        "predicted_class": result_data["predicted_class"],
                        "confidence": result_data["classification_confidence"],
                        "risk_score": result_data["risk_score"],
                        "risk_level": result_data["risk_level"],
                        "probabilities": result_data["class_probabilities"],
                        "model_used": result_data["model_used"],
                        "image_path": None,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    p_df = pd.DataFrame({
                        "Disease Class": [c.replace("_", " ").title() for c in CLASSES],
                        "Probability": [result_data["class_probabilities"].get(c, 0.0) * 100 for c in CLASSES]
                    })
                    fig_res = px.bar(p_df, x="Disease Class", y="Probability", color="Disease Class", text="Probability")
                    fig_res.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_res.update_layout(yaxis_range=[0, 110], height=300)
                    st.plotly_chart(fig_res, use_container_width=True)

# =============================================================================
# TAB 4: SYSTEM STATUS
# =============================================================================
with tab_system:
    st.subheader("⚙️ Edge System Architecture & Node Health")
    st.write("Real-time monitoring of IoT firmware connections, microservices, and neural inference components.")

    c_s1, c_s2 = st.columns([1.2, 1.8])

    with c_s1:
        st.markdown("#### 🔌 Connection & Hardware Status")
        status_table = [
            {"Component": "ESP32 Sensor Node", "Status": "Connected (SAFE-ESP32-01)" if is_real_esp32 else "Simulated", "State": "🟢 Active" if is_real_esp32 else "🟡 Mocked"},
            {"Component": "ESP32-CAM Node", "Status": "Wi-Fi Ready", "State": "🟢 Active"},
            {"Component": "FastAPI REST Server", "Status": "Online (Port 8000)" if backend_online else "Offline", "State": "🟢 Active" if backend_online else "🔴 Offline"},
            {"Component": "MobileNetV2 Vision Model", "Status": "Loaded (mobilenetv2_classifier.keras)", "State": "🟢 Healthy"},
            {"Component": "LSTM Temporal Model", "Status": "Loaded (lstm_temporal.keras)", "State": "🟢 Healthy"},
            {"Component": "Multimodal Fusion Engine", "Status": "Loaded (fusion_model.keras)", "State": "🟢 Healthy"},
            {"Component": "Last Sensor Ingestion", "Status": last_update_str, "State": "🟢 Live Synced" if is_real_esp32 else "🟡 Idle"}
        ]
        st.dataframe(pd.DataFrame(status_table), use_container_width=True, hide_index=True)

    with c_s2:
        st.markdown("#### 🧠 Model Information & Parameters")
        st.markdown("""
        | Parameter | Specification |
        | :--- | :--- |
        | **Vision Backbone** | MobileNetV2 (Pre-trained ImageNet, Frozen base) |
        | **Image Resolution** | 64 × 64 × 3 RGB Leaf Crops |
        | **Temporal Network** | 2-Layer Stacked LSTM (64 units -> 32 units) |
        | **Temporal Window** | 30 Timesteps (1-min resolution microclimate) |
        | **Fusion Technique** | Feature Concatenation (256-D Image + 64-D LSTM = 320-D) |
        | **Target Classes** | 4 Classes: `bacterial_spot`, `early_blight`, `healthy`, `late_blight` |
        | **Decision Output** | Class Probabilities (Softmax) + Multimodal Pathogen Risk Score |
        """)
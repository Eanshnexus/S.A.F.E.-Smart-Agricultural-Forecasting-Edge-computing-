"""
src/api/app.py
S.A.F.E. -- FastAPI Backend Server
Wraps SAFEPredictor to provide REST endpoints for tomato crop disease prediction
and environmental telemetry ingestion.
"""

from __future__ import annotations

import io
import sys
import requests
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predictor import SAFEPredictor, CLASSES, IMG_SIZE, SEQ_LEN, N_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SAFE_API")

app = FastAPI(
    title="S.A.F.E. API",
    description="Smart Agricultural Forecasting & Edge-computing REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton predictor instance
predictor: Optional[SAFEPredictor] = None

# Store latest live telemetry state (from ESP32 or simulated demo)
latest_telemetry = {
    "temperature": 27.5,
    "humidity": 64.0,
    "soil_moisture": 58.0,
    "npk_level": 2,
    "ph_value": 68.0,
    "source": "DEMO (Simulation)",
    "is_real_hardware": False,
    "esp32_connected": False,
    "last_updated": datetime.now(timezone.utc).isoformat(),
    "env_buffer": []  # Buffer of recent readings for sliding window
}

@app.on_event("startup")
def load_models():
    global predictor
    try:
        logger.info("Initializing SAFEPredictor...")
        predictor = SAFEPredictor(use_fusion=True)
        logger.info("SAFEPredictor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SAFEPredictor: {e}")
        predictor = None


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    predictor_loaded: bool
    model_mode: str
    classes: List[str]
    timestamp: str
    esp32_connected: bool
    is_demo_mode: bool

class TelemetryUpdate(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in %")
    soil_moisture: Optional[float] = Field(None, description="Soil moisture in %")
    npk_level: Optional[int] = Field(2, ge=1, le=3, description="NPK level: 1 (Low), 2 (Med), 3 (High)")
    ph_value: Optional[float] = Field(68.0, description="pH value or index")
    is_real_hardware: bool = False

class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    class_probabilities: dict[str, float]
    classification_confidence: float
    risk_score: float
    risk_level: str
    model_used: str
    soil_moisture_note: str
    is_demo_mode: bool
    timestamp: str


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify backend service and model status."""
    is_loaded = predictor is not None
    model_mode = "Fusion (MobileNetV2 + LSTM)" if (is_loaded and predictor._fusion_model is not None) else "MobileNetV2 (Image Only)"
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        predictor_loaded=is_loaded,
        model_mode=model_mode,
        classes=CLASSES,
        timestamp=datetime.now(timezone.utc).isoformat(),
        esp32_connected=latest_telemetry["esp32_connected"],
        is_demo_mode=not latest_telemetry["is_real_hardware"]
    )


@app.get("/telemetry")
def get_telemetry():
    """Returns the latest environmental telemetry readings and connection state."""
    return latest_telemetry


class SensorDataPayload(BaseModel):
    device_id: str = Field("SAFE-ESP32-01", description="Device Identifier")
    temperature: float = Field(..., description="Ambient temperature in °C")
    humidity: float = Field(..., description="Relative humidity in %")
    soil_moisture_raw: Optional[int] = Field(None, description="Raw ADC 12-bit reading")
    soil_moisture_percent: Optional[float] = Field(None, description="Calibrated percentage")
    timestamp: Optional[int] = Field(None, description="Epoch or millis timestamp")

@app.post("/sensor-data")
def receive_sensor_data(data: SensorDataPayload):
    """
    Direct endpoint for ESP32-WROOM hardware firmware.
    Ingests temperature, humidity, raw and percentage soil moisture.
    """
    latest_telemetry["temperature"] = data.temperature
    latest_telemetry["humidity"] = data.humidity
    if data.soil_moisture_percent is not None:
        latest_telemetry["soil_moisture"] = data.soil_moisture_percent
    latest_telemetry["is_real_hardware"] = True
    latest_telemetry["source"] = f"ESP32 Hardware ({data.device_id})"
    latest_telemetry["esp32_connected"] = True
    latest_telemetry["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Append to sliding window
    reading = [latest_telemetry["npk_level"], data.temperature, data.humidity, latest_telemetry["ph_value"]]
    latest_telemetry["env_buffer"].append(reading)
    if len(latest_telemetry["env_buffer"]) > 60:
        latest_telemetry["env_buffer"].pop(0)

    logger.info(f"Received data from {data.device_id}: Temp={data.temperature}C, Hum={data.humidity}%, Soil={data.soil_moisture_percent}% (Raw: {data.soil_moisture_raw})")
    return {
        "status": "success",
        "message": "Sensor data received successfully",
        "device_id": data.device_id,
        "recorded_at": latest_telemetry["last_updated"]
    }

@app.post("/telemetry")
def update_telemetry(payload: TelemetryUpdate):
    """Endpoint for ESP32 hardware (or simulator) to push new sensor readings."""
    latest_telemetry["temperature"] = payload.temperature
    latest_telemetry["humidity"] = payload.humidity
    if payload.soil_moisture is not None:
        latest_telemetry["soil_moisture"] = payload.soil_moisture
    latest_telemetry["npk_level"] = payload.npk_level or 2
    latest_telemetry["ph_value"] = payload.ph_value or 68.0
    latest_telemetry["is_real_hardware"] = payload.is_real_hardware
    latest_telemetry["source"] = "ESP32 Hardware" if payload.is_real_hardware else "DEMO (Simulation)"
    latest_telemetry["esp32_connected"] = payload.is_real_hardware
    latest_telemetry["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Append to sliding window buffer (capped at 60)
    reading = [latest_telemetry["npk_level"], latest_telemetry["temperature"], latest_telemetry["humidity"], latest_telemetry["ph_value"]]
    latest_telemetry["env_buffer"].append(reading)
    if len(latest_telemetry["env_buffer"]) > 60:
        latest_telemetry["env_buffer"].pop(0)

    return {"status": "telemetry updated", "is_real_hardware": payload.is_real_hardware}


def _process_image(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        return img_arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")


latest_cam_image_bytes = None

@app.post("/upload-leaf")
async def upload_leaf_from_cam(image: UploadFile = File(...)):
    """
    Endpoint for ESP32-CAM to upload a photo directly over Wi-Fi.
    Automatically runs multimodal inference against current telemetry.
    """
    global latest_cam_image_bytes
    file_bytes = await image.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image received")
    
    latest_cam_image_bytes = file_bytes
    img_arr = _process_image(file_bytes)

    # Save to disk as latest_cam_capture.jpg
    save_path = ROOT / "sample_images" / "latest_cam_capture.jpg"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Run inference with current telemetry
    temp = latest_telemetry["temperature"]
    hum = latest_telemetry["humidity"]
    soil = latest_telemetry["soil_moisture"]
    npk = latest_telemetry["npk_level"]
    ph = latest_telemetry["ph_value"]

    base_step = np.array([npk, temp, hum, ph], dtype=np.float32)
    seq = np.tile(base_step, (SEQ_LEN, 1))

    raw_result = predictor.predict(img_arr, seq, soil_moisture=soil)
    return {
        "status": "success",
        "predicted_class": raw_result["predicted_class"],
        "confidence": max(raw_result["probabilities"].values()),
        "risk_score": raw_result["risk_score"],
        "risk_level": raw_result["risk_level"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/trigger-cam-capture")
def trigger_cam_capture(cam_ip: str = Query(..., description="Local IP of ESP32-CAM")):
    """
    Triggers ESP32-CAM by making a GET request to http://<cam_ip>/capture,
    downloads the image, and processes it through the multimodal AI pipeline.
    """
    global latest_cam_image_bytes
    url = f"http://{cam_ip}/capture"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ESP32-CAM returned HTTP {resp.status_code}")
        
        file_bytes = resp.content
        latest_cam_image_bytes = file_bytes

        save_path = ROOT / "sample_images" / "latest_cam_capture.jpg"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        img_arr = _process_image(file_bytes)

        temp = latest_telemetry["temperature"]
        hum = latest_telemetry["humidity"]
        soil = latest_telemetry["soil_moisture"]
        npk = latest_telemetry["npk_level"]
        ph = latest_telemetry["ph_value"]

        base_step = np.array([npk, temp, hum, ph], dtype=np.float32)
        seq = np.tile(base_step, (SEQ_LEN, 1))

        raw_result = predictor.predict(img_arr, seq, soil_moisture=soil)
        return {
            "status": "success",
            "predicted_class": raw_result["predicted_class"],
            "class_probabilities": raw_result["probabilities"],
            "classification_confidence": max(raw_result["probabilities"].values()),
            "risk_score": raw_result["risk_score"],
            "risk_level": raw_result["risk_level"],
            "model_used": raw_result["model_used"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch from ESP32-CAM at {cam_ip}: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: Optional[UploadFile] = File(None, description="Tomato leaf RGB image"),
    temperature: Optional[float] = Form(None),
    humidity: Optional[float] = Form(None),
    soil_moisture: Optional[float] = Form(None),
    npk_level: Optional[int] = Form(None),
    ph_value: Optional[float] = Form(None),
    demo_mode: bool = Form(False)
):
    """
    Multimodal crop disease prediction endpoint.
    Accepts:
      - Tomato leaf image (file upload)
      - Environmental variables: temperature, humidity, soil_moisture, npk, pH
    Returns predicted disease class, confidence, probabilities, and multimodal risk score.
    """
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction service unavailable: Model not loaded")

    # 1. Process Image
    if image is not None:
        file_bytes = await image.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty")
        img_arr = _process_image(file_bytes)
    else:
        # If no image provided, generate placeholder/demo crop
        logger.warning("No image provided. Using simulated leaf image.")
        img_arr = np.ones((*IMG_SIZE, 3), dtype=np.float32) * 0.5

    # 2. Build Environmental Sequence
    # Fallback to latest telemetry if values not provided in form
    temp = temperature if temperature is not None else latest_telemetry["temperature"]
    hum = humidity if humidity is not None else latest_telemetry["humidity"]
    npk = npk_level if npk_level is not None else latest_telemetry["npk_level"]
    ph = ph_value if ph_value is not None else latest_telemetry["ph_value"]
    soil = soil_moisture if soil_moisture is not None else latest_telemetry["soil_moisture"]

    # Construct 30-step temporal sequence
    # If telemetry buffer has history, use it; otherwise repeat with slight natural variance
    buffer = latest_telemetry.get("env_buffer", [])
    if len(buffer) >= SEQ_LEN:
        seq = np.array(buffer[-SEQ_LEN:], dtype=np.float32)
    else:
        # Build 30-step window around current values
        base_step = np.array([npk, temp, hum, ph], dtype=np.float32)
        noise = np.random.normal(0, [0.0, 0.3, 0.5, 0.2], size=(SEQ_LEN, 4)).astype(np.float32)
        seq = np.tile(base_step, (SEQ_LEN, 1)) + noise
        seq[:, 0] = np.clip(np.round(seq[:, 0]), 1, 3) # NPK 1-3

    # 3. Predict via SAFEPredictor
    try:
        raw_result = predictor.predict(
            image_array=img_arr,
            env_sequence_raw=seq,
            soil_moisture=soil
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    probs = raw_result["probabilities"]
    conf = float(max(probs.values()))

    return PredictionResponse(
        predicted_class=raw_result["predicted_class"],
        class_index=raw_result["class_index"],
        class_probabilities=probs,
        classification_confidence=conf,
        risk_score=raw_result["risk_score"],
        risk_level=raw_result["risk_level"],
        model_used=raw_result["model_used"],
        soil_moisture_note=raw_result["soil_moisture_note"],
        is_demo_mode=demo_mode or not latest_telemetry["is_real_hardware"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)





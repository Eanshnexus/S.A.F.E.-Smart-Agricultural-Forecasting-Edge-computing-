"""
src/inference/predictor.py
S.A.F.E. -- Reusable Inference Module

Provides SAFEPredictor: a single class that loads all trained models
and returns a structured prediction dict from an image + env inputs.

Usage (standalone test):
    python src/inference/predictor.py

Usage (from FastAPI):
    from src.inference.predictor import SAFEPredictor
    predictor = SAFEPredictor()
    result = predictor.predict(image_array, env_sequence)

Input spec:
    image_array  : np.ndarray, shape (64, 64, 3), dtype float32, range [0,1]
    env_sequence : np.ndarray, shape (30, 4), dtype float32
                   columns in order: [NPK Level, Temperature, Humidity, Ph Value]
                   Values should be raw (unscaled) -- predictor handles scaling.

    Optional (for future HW integration):
    soil_moisture : float or None
                    Accepted and logged, but NOT fed to model.
                    The current model was trained without this feature.
                    Include in future version when real paired data is available.

Output dict:
    predicted_class  : str   e.g. "bacterial_spot"
    class_index      : int
    probabilities    : dict  {class_name: float}
    risk_score       : float  (probability of any disease class, 0-1)
    risk_level       : str   "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    model_used       : str
    soil_moisture_note: str  (transparency about missing feature)
"""

from __future__ import annotations
import os, sys, json, warnings
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
import yaml
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)


MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
PROC_DIR   = ROOT / CFG["paths"]["processed_dir"]
CLASSES    = CFG["image"]["classes"]
IMG_SIZE   = tuple(CFG["image"]["input_size"])
SEQ_LEN    = CFG["env"]["sequence_length"]
N_FEATURES = 4   # NPK Level, Temperature, Humidity, Ph Value

RISK_THRESHOLDS = {
    "LOW":      0.30,
    "MODERATE": 0.55,
    "HIGH":     0.75,
    # above 0.75 -> CRITICAL
}

HEALTHY_IDX = CLASSES.index("healthy")


# -----------------------------------------------------------------------------
class SAFEPredictor:
    """
    Loads all trained S.A.F.E. models and scaler once, then serves predictions.
    Thread-safe for single-process use.
    """

    def __init__(self, use_fusion: bool = True):
        import tensorflow as tf
        from tensorflow import keras

        self._use_fusion = use_fusion
        self._classes    = CLASSES

        print("Loading S.A.F.E. models...")

        # Always load image model
        clf_path = MODELS_DIR / "mobilenetv2_classifier.keras"
        self._img_model = keras.models.load_model(str(clf_path))
        print(f"  OK: Image classifier loaded  ({clf_path.name})")

        # Load scaler for env features
        scaler_path = PROC_DIR / "env_scaler.pkl"
        self._scaler = joblib.load(str(scaler_path))
        print(f"  OK: Env scaler loaded         ({scaler_path.name})")

        # Optionally load fusion model
        if use_fusion:
            fusion_path = MODELS_DIR / "fusion_model.keras"
            if fusion_path.exists():
                self._fusion_model = keras.models.load_model(str(fusion_path))
                print(f"  OK: Fusion model loaded       ({fusion_path.name})")
            else:
                print(f"  WARN: Fusion model not found — falling back to image-only.")
                self._use_fusion = False
                self._fusion_model = None
        else:
            self._fusion_model = None

        print("SAFEPredictor ready.\n")

    # --------------------------------------------------------------------------
    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.75:
            return "CRITICAL"
        if risk_score >= 0.55:
            return "HIGH"
        if risk_score >= 0.30:
            return "MODERATE"
        return "LOW"

    def _compute_risk_score(self, probs: dict) -> float:
        """Risk = 1 - P(healthy)."""
        return round(1.0 - probs.get("healthy", 0.0), 4)

    # --------------------------------------------------------------------------
    def predict(
        self,
        image_array: np.ndarray,
        env_sequence_raw: Optional[np.ndarray] = None,
        soil_moisture: Optional[float] = None,
    ) -> dict:
        """
        Parameters
        ----------
        image_array       : (64, 64, 3) float32, range [0,1]
        env_sequence_raw  : (30, 4) float32 — raw unscaled values
                            [NPK Level, Temperature, Humidity, Ph Value]
                            If None, falls back to image-only prediction.
        soil_moisture     : float or None — accepted for HW compatibility
                            but NOT used by the current model.
        Returns
        -------
        dict with predicted_class, class_index, probabilities, risk_score,
             risk_level, model_used, soil_moisture_note
        """
        # -- Validate image ----------------------------------------------------
        if image_array.ndim == 3:
            img_batch = image_array[np.newaxis, ...]   # (1, 64, 64, 3)
        else:
            img_batch = image_array

        assert img_batch.shape[1:] == (*IMG_SIZE, 3), \
            f"Expected image shape (N, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3), got {img_batch.shape}"

        # -- Choose prediction mode ---------------------------------------------
        can_fuse = (
            self._use_fusion
            and self._fusion_model is not None
            and env_sequence_raw is not None
        )

        if can_fuse:
            # Scale env sequence
            seq = env_sequence_raw.reshape(-1, N_FEATURES)
            seq_scaled = self._scaler.transform(seq).reshape(1, SEQ_LEN, N_FEATURES)
            probs_arr = self._fusion_model.predict(
                [img_batch, seq_scaled], verbose=0
            )[0]
            model_used = "MobileNetV2 + LSTM Fusion"
        else:
            probs_arr = self._img_model.predict(img_batch, verbose=0)[0]
            model_used = "MobileNetV2 (image only)"
            if env_sequence_raw is None:
                model_used += " — env sequence not provided"

        probs_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs_arr)}
        cls_idx    = int(np.argmax(probs_arr))
        cls_name   = CLASSES[cls_idx]
        risk_score = self._compute_risk_score(probs_dict)
        risk_lvl   = self._risk_level(risk_score)

        soil_note = (
            "Soil moisture accepted but NOT used by current model. "
            "dataset.csv contained no soil moisture measurements. "
            "This feature is reserved for future model versions."
        ) if soil_moisture is not None else (
            "Soil moisture not provided. Not required by current model."
        )

        return {
            "predicted_class": cls_name,
            "class_index":     cls_idx,
            "probabilities":   probs_dict,
            "risk_score":      risk_score,
            "risk_level":      risk_lvl,
            "model_used":      model_used,
            "soil_moisture_note": soil_note,
        }

    # --------------------------------------------------------------------------
    def predict_from_paths(
        self,
        image_path: str,
        env_sequence_raw: Optional[np.ndarray] = None,
        soil_moisture: Optional[float] = None,
    ) -> dict:
        """Convenience wrapper: load image from file path."""
        from PIL import Image as PILImage
        img = PILImage.open(image_path).convert("RGB").resize(IMG_SIZE)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        return self.predict(img_arr, env_sequence_raw, soil_moisture)


# -----------------------------------------------------------------------------
def _self_test():
    """Run a quick end-to-end test with dummy inputs to verify load + inference."""
    print("=" * 60)
    print("  SAFEPredictor — Self Test")
    print("=" * 60)

    pred = SAFEPredictor(use_fusion=True)

    # Dummy image (random)
    dummy_img = np.random.rand(64, 64, 3).astype(np.float32)

    # Dummy env sequence (raw unscaled values in realistic ranges)
    dummy_env = np.column_stack([
        np.random.randint(1, 4,  30).astype(np.float32),   # NPK Level 1-3
        np.random.uniform(25, 40, 30).astype(np.float32),  # Temperature
        np.random.uniform(45, 70, 30).astype(np.float32),  # Humidity
        np.random.uniform(60, 80, 30).astype(np.float32),  # Ph Value
    ])  # shape (30, 4)

    print("\n[Test 1] Fusion prediction (image + env):")
    result = pred.predict(dummy_img, dummy_env)
    print(json.dumps(result, indent=2))

    print("\n[Test 2] Image-only prediction (no env sequence):")
    result2 = pred.predict(dummy_img, env_sequence_raw=None)
    print(json.dumps(result2, indent=2))

    print("\n[Test 3] With soil_moisture passthrough:")
    result3 = pred.predict(dummy_img, dummy_env, soil_moisture=42.5)
    print(f"  soil_moisture_note: {result3['soil_moisture_note'][:80]}...")

    print("\nOK: Self-test passed — predictor is operational.")


if __name__ == "__main__":
    _self_test()

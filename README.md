# S.A.F.E. — Smart Agricultural Forecasting & Edge-computing

**Proactive tomato crop-disease prediction using a CNN-LSTM multimodal architecture.**

---

## Project Overview

S.A.F.E. combines two data modalities to predict tomato leaf disease:

| Modality | Source | Model |
|---|---|---|
| RGB leaf images | PlantVillage Tomato Dataset | MobileNetV2 (frozen, ImageNet) |
| Environmental time-series | IoT sensor CSV (temp, humidity, NPK, pH) | Stacked LSTM |
| Fusion | Both above | Concatenation + Dense head |

**Target classes:** `bacterial_spot`, `early_blight`, `healthy`, `late_blight`

---

## Dataset Details

### Image Dataset (PlantVillage)
Pre-split into `Train/Validate/Test` folders:

| Split | bacterial_spot | early_blight | healthy | late_blight | Total |
|---|---|---|---|---|---|
| Train    | 300 | 300 | 300 | 300 | 1,200 |
| Validate | 64  | 64  | 64  | 64  | 256   |
| Test     | 64  | 300 | 64  | 64  | 492   |

> Note: `Test/early_blight` has 300 images. 64 are sampled for balanced evaluation.

### Environmental Dataset (`dataset.csv`)
- **3,551 rows**, 1-minute readings from 2023-11-27 10:00 to 2023-11-29 21:10
- **No missing values, no duplicates**
- **Features used:** `NPK Level` (1-3), `Temperature` (19-49degC), `Humidity` (38-70%), `Ph Value` (44-97)
- **Label:** `Diseased` (0 = Healthy, 1 = Diseased) — **binary only**, not 4-class
- **Soil moisture:** NOT present in dataset.csv. Not included in the model.

---

## Scientific Limitations (Read Before Citing)

> **CRITICAL:** PlantVillage images and `dataset.csv` are **INDEPENDENT, UNPAIRED** datasets.
> No plant-level correspondence exists between them.

For the multimodal fusion model, images are paired with environmental sequences
**by disease class** (disease images -> sequences where `Diseased==1`,
healthy images -> sequences where `Diseased==0`).
This is a **synthetic proof-of-concept pairing** for architectural demonstration only.
It does NOT constitute scientifically validated real-world crop disease forecasting.

**LSTM Performance Note:** The standalone LSTM achieved ~52% accuracy on the binary
disease label — essentially random-chance performance. This is an honest result:
the 4 environmental features in the CSV (NPK, Temp, Humidity, pH) do not contain
a clearly learnable temporal pattern for the binary disease label at a 30-minute
window. This is documented transparently and does not invalidate the fusion
architecture demonstration.

---

## Architecture

```
Image (64x64x3)                  Env Sequence (30-steps x 4-features)
       |                                       |
  MobileNetV2                         LSTM(64) -> LSTM(32)
  (ImageNet, frozen)                        |
  GlobalAvgPool                       Dense(64, relu)
  Dense(256, relu)                          |
       |                                     |
       |___________Concatenate_______________|
                         |
                   Dense(128, relu)
                   Dropout(0.3)
                   Dense(64, relu)
                   Dense(4, softmax)
```

---

## Results

| Model | Test Accuracy | Macro F1 | Notes |
|---|---|---|---|
| MobileNetV2 (image only) | **83.5%** | **0.81** | Frozen base, 30 epochs |
| LSTM (env only, binary) | 52.4% | 0.56 | Near-random; CSV features lack learnable temporal pattern |
| CNN-LSTM Fusion | TBD | TBD | Synthetic pairing — indicative only |

---

## Project Structure

```
SAFE/
|- config.yaml              # All hyperparameters and paths
|- requirements.txt         # Python dependencies
|- run_pipeline.py          # Master pipeline runner
|- src/
|  |- data/
|  |  |- prepare_image_data.py    # Image validation & generators
|  |  |- prepare_env_data.py      # CSV cleaning & sequence building
|  |- image_model/
|  |  |- train_image_model.py     # MobileNetV2 training
|  |- temporal_model/
|  |  |- train_lstm.py            # LSTM temporal training
|  |- fusion/
|  |  |- train_fusion.py          # Multimodal fusion training
|  |- evaluation/
|  |  |- evaluate_models.py       # Comparison report & charts
|  |- inference/
|     |- predictor.py             # SAFEPredictor class (for FastAPI)
|- models/                  # Saved .keras model files
|- data/processed/          # Sequences, scaler, class indices
|- reports/                 # Metrics JSON, plots, confusion matrices
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline from scratch
python run_pipeline.py

# Or run steps individually:
python src/data/prepare_image_data.py
python src/data/prepare_env_data.py
python src/image_model/train_image_model.py
python src/temporal_model/train_lstm.py
python src/fusion/train_fusion.py
python src/evaluation/evaluate_models.py
python src/inference/predictor.py    # self-test
```

---

## Inference API Usage

```python
from src.inference.predictor import SAFEPredictor
import numpy as np

predictor = SAFEPredictor(use_fusion=True)

# image_array: (64, 64, 3) float32, range [0, 1]
# env_sequence: (30, 4) float32 - raw values [NPK, Temperature, Humidity, Ph]
result = predictor.predict(image_array, env_sequence)

print(result["predicted_class"])  # e.g. "bacterial_spot"
print(result["risk_level"])       # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
print(result["probabilities"])    # {class: probability}
```

---

## Hardware Integration (ESP32)

The inference module accepts `soil_moisture` as a passthrough parameter
for ESP32 hardware compatibility:

```python
result = predictor.predict(image_array, env_sequence, soil_moisture=42.5)
```

Soil moisture is **not used by the current model** (absent from training data)
but is logged in the output for future model versions once paired data is collected.

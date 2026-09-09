import requests
import numpy as np
from pathlib import Path
from PIL import Image

# 1. Health check
print("Testing /health...")
r = requests.get("http://localhost:8000/health")
print("Health status:", r.status_code)
print(r.json())

# 2. Test /predict with sample image
print("\nTesting /predict with sample leaf image...")
sample_img = Path("c:/Users/gupta/Desktop/SAFE/sample_images/early_blight.jpg")
if sample_img.exists():
    with open(sample_img, "rb") as f:
        files = {"image": ("early_blight.jpg", f, "image/jpeg")}
        data = {
            "temperature": 29.5,
            "humidity": 82.0,
            "soil_moisture": 62.0,
            "npk_level": 2,
            "ph_value": 70.0
        }
        res = requests.post("http://localhost:8000/predict", files=files, data=data)
        print("Predict status:", res.status_code)
        print(res.json())
else:
    print("Sample image not found, skipping predict test.")

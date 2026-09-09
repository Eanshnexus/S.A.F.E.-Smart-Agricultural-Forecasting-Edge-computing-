"""
src/data/prepare_env_data.py
S.A.F.E. -- Step 2: Environmental Time-Series Data Preparation

Reads dataset.csv, cleans and normalises it, then constructs
30-step sliding-window sequences for LSTM training.

WHAT IS IN THE CSV:
  - 3,551 rows, 1-minute timestamps from 27-Nov-2023 10:00
    to 29-Nov-2023 21:10 (continuous, no gaps).
  - Columns: Timestamp, NPK Level, Temperature, Humidity, Ph Value, Diseased

WHAT IS NOT IN THE CSV:
  - No soil-moisture column. The LSTM will NOT include soil moisture.
    This is explicitly documented so as not to misrepresent the data.

TEMPORAL SPLIT STRATEGY:
  - Split is performed chronologically (no shuffle before split)
    to respect time-series causality:
      Train : first 70% of timesteps
      Val   : next  15%
      Test  : last  15%
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

SEED       = CFG["project"]["seed"]
CSV_PATH   = Path(CFG["paths"]["raw_csv"])
PROC_DIR   = ROOT / CFG["paths"]["processed_dir"]
REP_DIR    = ROOT / CFG["paths"]["reports_dir"]
SEQ_LEN    = CFG["env"]["sequence_length"]
STRIDE     = CFG["env"]["stride"]

FEATURE_COLS_RENAME = {
    "NPK Level": "NPK Level",
    "Humidity (%)": "Humidity",
    "Ph Value": "Ph Value",
}

np.random.seed(SEED)


# -----------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    """Load CSV, rename columns, parse timestamps."""
    # Detect actual temperature column name (has degree symbol)
    raw_cols = pd.read_csv(CSV_PATH, encoding="latin-1", nrows=0).columns.tolist()
    temp_col = next(c for c in raw_cols if "temp" in c.lower() or "temperature" in c.lower())
    humidity_col = next(c for c in raw_cols if "humid" in c.lower())

    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df = df.rename(columns={temp_col: "Temperature", humidity_col: "Humidity"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    print("=" * 60)
    print("  Environmental Dataset — Cleaned Summary")
    print("=" * 60)
    print(f"  Shape            : {df.shape}")
    print(f"  Timestamp range  : {df['Timestamp'].iloc[0]}  -->  {df['Timestamp'].iloc[-1]}")
    print(f"  Missing values   : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows   : {df.duplicated().sum()}")
    print(f"\n  Class distribution:")
    print(f"    Diseased=0 (Healthy)  : {(df['Diseased']==0).sum()} ({100*(df['Diseased']==0).mean():.1f}%)")
    print(f"    Diseased=1 (Diseased) : {(df['Diseased']==1).sum()} ({100*(df['Diseased']==1).mean():.1f}%)")
    print(f"\n  Feature statistics:")
    feat_cols = ["NPK Level", "Temperature", "Humidity", "Ph Value"]
    print(df[feat_cols].describe().to_string())
    print()
    print("  IMPORTANT: No soil-moisture column found in dataset.csv.")
    print("  The LSTM uses: NPK Level, Temperature, Humidity, Ph Value (4 features).")
    return df


def plot_env_timeseries(df: pd.DataFrame):
    """Plot the 4 environmental features over time."""
    feat_cols = ["Temperature", "Humidity", "NPK Level", "Ph Value"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(len(feat_cols), 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Environmental Time-Series — dataset.csv\n(S.A.F.E. Project)", fontsize=13, fontweight="bold")

    for ax, col, color in zip(axes, feat_cols, colors):
        ax.plot(df["Timestamp"], df[col], color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel(col, fontsize=9)
        ax.grid(True, alpha=0.25)
        # Overlay disease events
        diseased_ts = df[df["Diseased"] == 1]["Timestamp"]
        ax.scatter(diseased_ts, df[df["Diseased"] == 1][col],
                   color="red", s=1.5, alpha=0.25, label="Diseased=1")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    plt.xticks(rotation=30, fontsize=7)
    plt.tight_layout()

    out = REP_DIR / "env_timeseries.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"OK: Time-series plot saved -> {out}")


def build_sequences(df: pd.DataFrame):
    """
    Construct sliding-window sequences of length SEQ_LEN.
    Temporal split: 70% train / 15% val / 15% test (chronological).
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, scaler).
    """
    feat_cols = ["NPK Level", "Temperature", "Humidity", "Ph Value"]
    features  = df[feat_cols].values.astype(np.float32)
    labels    = df["Diseased"].values.astype(np.int32)

    # -- Temporal split on raw rows (before windowing) ------------------------
    n   = len(features)
    n_train = int(n * CFG["env"]["train_ratio"])
    n_val   = int(n * CFG["env"]["val_ratio"])

    train_feat, train_lab = features[:n_train],          labels[:n_train]
    val_feat,   val_lab   = features[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_feat,  test_lab  = features[n_train+n_val:],    labels[n_train+n_val:]

    # -- Fit scaler on train only (no data leakage) ---------------------------
    scaler = StandardScaler()
    train_feat_sc = scaler.fit_transform(train_feat)
    val_feat_sc   = scaler.transform(val_feat)
    test_feat_sc  = scaler.transform(test_feat)

    def make_windows(feat_sc, lab):
        X, y = [], []
        for i in range(0, len(feat_sc) - SEQ_LEN, STRIDE):
            X.append(feat_sc[i:i + SEQ_LEN])
            y.append(lab[i + SEQ_LEN - 1])   # label at last step of window
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    X_train, y_train = make_windows(train_feat_sc, train_lab)
    X_val,   y_val   = make_windows(val_feat_sc,   val_lab)
    X_test,  y_test  = make_windows(test_feat_sc,  test_lab)

    print(f"\n  Sequence shape (features={len(feat_cols)}, window={SEQ_LEN}):")
    print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}  pos={y_train.sum()}")
    print(f"    X_val  : {X_val.shape}  y_val  : {y_val.shape}")
    print(f"    X_test : {X_test.shape}  y_test : {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feat_cols


def save_artifacts(X_train, y_train, X_val, y_val, X_test, y_test, scaler, feat_cols):
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    npz_path = PROC_DIR / "env_sequences.npz"
    np.savez_compressed(npz_path,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val,
                        X_test=X_test,   y_test=y_test)
    print(f"OK: Sequences saved -> {npz_path}")

    # Save scaler
    scaler_path = PROC_DIR / "env_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"OK: Scaler saved -> {scaler_path}")

    # Save metadata
    meta = {
        "sequence_length": SEQ_LEN,
        "stride": STRIDE,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "splits": {
            "train": {"X": list(X_train.shape), "n_positive": int(y_train.sum())},
            "val":   {"X": list(X_val.shape),   "n_positive": int(y_val.sum())},
            "test":  {"X": list(X_test.shape),  "n_positive": int(y_test.sum())},
        },
        "soil_moisture_note": (
            "ABSENT from dataset.csv. The LSTM uses 4 real features: "
            "NPK Level, Temperature, Humidity, Ph Value. "
            "Soil moisture is NOT included and NOT synthetically generated."
        ),
        "pairing_note": (
            "Temporal split is chronological. "
            "No shuffling applied before splitting to respect time-series causality."
        ),
    }
    meta_path = PROC_DIR / "env_metadata.json"
    with open(meta_path, "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print(f"OK: Metadata saved -> {meta_path}")


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    plot_env_timeseries(df)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feat_cols = build_sequences(df)
    save_artifacts(X_train, y_train, X_val, y_val, X_test, y_test, scaler, feat_cols)

    print("\nOK: Environmental data preparation complete.")

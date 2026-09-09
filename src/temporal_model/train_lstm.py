"""
src/temporal_model/train_lstm.py
S.A.F.E. -- Step 4: LSTM Temporal Model Training

Uses preprocessed 30-step sliding windows from prepare_env_data.py.
Features: NPK Level, Temperature, Humidity, Ph Value (4 real features).
Target: Diseased 0/1 (binary).

Architecture:
  LSTM(64, return_sequences=True) -> Dropout
  LSTM(32)                        -> Dropout
  Dense(64, relu)                 -> name="lstm_features"
  Dense(1, sigmoid)

Saves:
  models/lstm_temporal.keras
  reports/lstm_metrics.json
  reports/training_history_lstm.png
"""

from __future__ import annotations
import os, sys, json, warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["project"]["seed"]
import random; random.seed(SEED)
import numpy as np; np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
REP_DIR    = ROOT / CFG["paths"]["reports_dir"]
PROC_DIR   = ROOT / CFG["paths"]["processed_dir"]
L_CFG      = CFG["lstm_model"]


# -----------------------------------------------------------------------------
def load_sequences():
    npz = np.load(PROC_DIR / "env_sequences.npz")
    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val,   y_val   = npz["X_val"],   npz["y_val"]
    X_test,  y_test  = npz["X_test"],  npz["y_test"]
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(seq_len: int, n_features: int) -> keras.Model:
    units      = L_CFG["units"]              # [64, 32]
    dropout    = L_CFG["dropout"]
    rec_drop   = L_CFG["recurrent_dropout"]
    dense_u    = L_CFG["dense_units"]        # 64

    inp = keras.Input(shape=(seq_len, n_features), name="env_sequence")
    x = layers.LSTM(units[0], return_sequences=True,
                    dropout=dropout, recurrent_dropout=rec_drop,
                    name="lstm_1")(inp)
    x = layers.LSTM(units[1], return_sequences=False,
                    dropout=dropout, recurrent_dropout=rec_drop,
                    name="lstm_2")(x)
    x = layers.Dense(dense_u, activation="relu", name="lstm_features")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="disease_prob")(x)

    model = keras.Model(inp, out, name="LSTM_Temporal_SAFE")
    model.compile(
        optimizer=keras.optimizers.Adam(L_CFG["training"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def plot_history(history, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history["accuracy"],     label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle("LSTM Temporal Model Training History — S.A.F.E.", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=100); plt.close()
    print(f"OK: History plot -> {path}")


def evaluate(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Healthy", "Diseased"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Diseased"],
                yticklabels=["Healthy","Diseased"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — LSTM (Test Set)\nS.A.F.E. Project")
    plt.tight_layout()
    plt.savefig(REP_DIR / "confusion_matrix_lstm.png", dpi=100); plt.close()
    print(f"OK: Confusion matrix -> {REP_DIR/'confusion_matrix_lstm.png'}")

    metrics = {
        "model": "LSTM_Temporal",
        "features": ["NPK Level", "Temperature", "Humidity", "Ph Value"],
        "soil_moisture": "NOT PRESENT in dataset.csv — not included",
        "sequence_length": int(X_test.shape[1]),
        "test_accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
    }
    out = REP_DIR / "lstm_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"OK: Metrics saved -> {out}")
    return metrics


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  S.A.F.E. — LSTM Temporal Model Training")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_sequences()
    seq_len, n_features = X_train.shape[1], X_train.shape[2]

    model = build_model(seq_len, n_features)

    t_cfg = L_CFG["training"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=t_cfg["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=t_cfg["reduce_lr_patience"],
                          factor=0.3, verbose=1, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=t_cfg["epochs"],
        batch_size=t_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    model_path = MODELS_DIR / "lstm_temporal.keras"
    model.save(model_path)
    print(f"\nOK: LSTM model saved -> {model_path}")

    plot_history(history, REP_DIR / "training_history_lstm.png")
    evaluate(model, X_test, y_test)

    print("\nOK: LSTM temporal model training complete.")

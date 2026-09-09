"""
src/evaluation/evaluate_models.py
S.A.F.E. -- Step 6: Unified Model Evaluation & Comparison Report

Loads all saved models, re-evaluates them on the test set, and
produces a final comparison report + summary table.

Outputs:
  reports/final_comparison.json
  reports/model_comparison_chart.png
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
import tensorflow as tf; tf.random.set_seed(SEED)

from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
REP_DIR    = ROOT / CFG["paths"]["reports_dir"]
PROC_DIR   = ROOT / CFG["paths"]["processed_dir"]
RAW_ROOT   = Path(CFG["paths"]["raw_image_root"])
CLASSES    = CFG["image"]["classes"]
IMG_SIZE   = tuple(CFG["image"]["input_size"])
HEALTHY_IDX = CLASSES.index("healthy")


# -----------------------------------------------------------------------------
def load_test_images(n_per_class: int = 64) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image as PILImage
    imgs, labels = [], []
    test_root = RAW_ROOT / "Test"
    for cls_idx, cls_name in enumerate(CLASSES):
        paths = sorted((test_root / cls_name).glob("*.jpg")) + \
                sorted((test_root / cls_name).glob("*.JPG")) + \
                sorted((test_root / cls_name).glob("*.png"))
        random.seed(SEED + cls_idx)
        paths = random.sample(paths, min(n_per_class, len(paths)))
        for p in paths:
            img = PILImage.open(p).convert("RGB").resize(IMG_SIZE)
            imgs.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(cls_idx)
    return np.array(imgs), np.array(labels)


def load_test_env_sequences(n_per_class: int = 64) -> tuple[np.ndarray, np.ndarray]:
    npz = np.load(PROC_DIR / "env_sequences.npz")
    X_test, y_test = npz["X_test"], npz["y_test"]
    diseased = X_test[y_test == 1]
    healthy  = X_test[y_test == 0]
    seqs = []
    for cls_idx in range(len(CLASSES)):
        pool = healthy if cls_idx == HEALTHY_IDX else diseased
        random.seed(SEED + cls_idx + 100)
        idxs = [random.randint(0, len(pool)-1) for _ in range(n_per_class)]
        seqs.extend([pool[i] for i in idxs])
    return np.array(seqs, dtype=np.float32)


def metrics_dict(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1":  round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall":    round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def plot_comparison(results: list[dict]):
    names   = [r["model"] for r in results]
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    labels  = ["Accuracy", "F1 (macro)", "Precision", "Recall"]
    colors  = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = [r[m] for r in results]
        bars = ax.bar(x + i*width, vals, width, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("S.A.F.E. — Model Comparison (Test Set)\n"
                 "Note: Fusion uses synthetic cross-dataset pairing — results are indicative",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = REP_DIR / "model_comparison_chart.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"OK: Comparison chart -> {out}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  S.A.F.E. — Final Model Evaluation & Comparison")
    print("=" * 60)

    print("\nLoading balanced test set (64 images per class)...")
    test_imgs, test_labels = load_test_images(64)
    test_seqs = load_test_env_sequences(64)

    results = []

    # -- 1. Image-only model --------------------------------------------------
    print("\n[1/2] Evaluating MobileNetV2 (image-only)...")
    img_model = keras.models.load_model(MODELS_DIR / "mobilenetv2_classifier.keras")
    y_pred_img = np.argmax(img_model.predict(test_imgs, verbose=0), axis=1)
    m_img = metrics_dict("MobileNetV2 (Image Only)", test_labels, y_pred_img)
    results.append(m_img)
    print(f"  Accuracy: {m_img['accuracy']}  F1: {m_img['macro_f1']}")

    # -- 2. Fusion model ------------------------------------------------------
    print("\n[2/2] Evaluating CNN-LSTM Fusion...")
    fusion_model = keras.models.load_model(MODELS_DIR / "fusion_model.keras")
    y_pred_fus = np.argmax(
        fusion_model.predict([test_imgs, test_seqs], verbose=0), axis=1
    )
    m_fus = metrics_dict("MobileNetV2 + LSTM Fusion", test_labels, y_pred_fus)
    results.append(m_fus)
    print(f"  Accuracy: {m_fus['accuracy']}  F1: {m_fus['macro_f1']}")

    # -- Summary --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON (Test Set, 64 imgs/class)")
    print("=" * 60)
    header = f"  {'Model':<35} {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}"
    print(header)
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['model']:<35} {r['accuracy']:>6.4f}  {r['macro_f1']:>6.4f}  "
              f"{r['macro_precision']:>6.4f}  {r['macro_recall']:>6.4f}")
    print()
    print("  SCIENTIFIC CAVEAT:")
    print("  Fusion result uses synthetic cross-dataset pairing.")
    print("  Do NOT present as a real paired-observation result.")

    # Save
    final = {
        "evaluation_note": (
            "64 images per class used for balanced evaluation. "
            "Fusion model uses synthetic pairing between PlantVillage images "
            "and dataset.csv sequences. These are INDEPENDENT datasets."
        ),
        "models": results,
    }
    out = REP_DIR / "final_comparison.json"
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nOK: Final comparison saved -> {out}")

    plot_comparison(results)
    print("\nOK: Evaluation complete.")

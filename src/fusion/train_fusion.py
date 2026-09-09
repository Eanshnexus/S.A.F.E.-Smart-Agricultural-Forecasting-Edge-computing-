"""
src/fusion/train_fusion.py
S.A.F.E. -- Step 5: Multimodal CNN-LSTM Fusion Model

ARCHITECTURE:
  Branch 1 — Image:
    Load frozen MobileNetV2 feature extractor
    Output: 256-D vector

  Branch 2 — Environment:
    Load frozen LSTM up to "lstm_features" layer
    Output: 64-D vector

  Fusion:
    Concatenate [256 + 64] = 320-D
    Dense(128, relu) -> Dropout(0.3)
    Dense(64, relu)
    Dense(4, softmax)  -- 4 disease classes

IMPORTANT SCIENTIFIC NOTE (also logged to reports):
  PlantVillage images and dataset.csv are INDEPENDENT datasets.
  No plant-level pairing exists between them.
  For the proof-of-concept fusion model:
    - Disease images (bacterial_spot, early_blight, late_blight)
      are paired with environmental sequences where Diseased == 1
    - Healthy images are paired with sequences where Diseased == 0
  This is a SYNTHETIC PAIRING for architectural demonstration only.
  It does NOT constitute scientifically validated real-world forecasting.
"""

from __future__ import annotations
import os, sys, json, warnings, random
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
random.seed(SEED)
import numpy as np; np.random.seed(SEED)
import tensorflow as tf; tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
REP_DIR    = ROOT / CFG["paths"]["reports_dir"]
PROC_DIR   = ROOT / CFG["paths"]["processed_dir"]
RAW_ROOT   = Path(CFG["paths"]["raw_image_root"])
CLASSES    = CFG["image"]["classes"]
IMG_SIZE   = tuple(CFG["image"]["input_size"])
F_CFG      = CFG["fusion_model"]
BATCH      = F_CFG["training"]["batch_size"]

# Class index: 0=bacterial_spot, 1=early_blight, 2=healthy, 3=late_blight
HEALTHY_IDX  = CLASSES.index("healthy")
DISEASE_IDXS = [i for i, c in enumerate(CLASSES) if c != "healthy"]

SPLITS = {
    "Train":    RAW_ROOT / "Train",
    "Validate": RAW_ROOT / "Validate",
    "Test":     RAW_ROOT / "Test",
}


# -----------------------------------------------------------------------------
def load_pretrained_branches():
    """Load frozen feature extractors from Stage 3 & 4."""
    img_ext = keras.models.load_model(MODELS_DIR / "image_feature_extractor.keras")
    img_ext.trainable = False

    lstm_full = keras.models.load_model(MODELS_DIR / "lstm_temporal.keras")
    lstm_ext  = keras.Model(
        inputs=lstm_full.input,
        outputs=lstm_full.get_layer("lstm_features").output,
        name="LSTM_FeatureExtractor",
    )
    lstm_ext.trainable = False

    print(f"  Image extractor  : {img_ext.output.shape}")
    print(f"  LSTM extractor   : {lstm_ext.output.shape}")
    return img_ext, lstm_ext


def build_fusion_model(img_ext, lstm_ext) -> keras.Model:
    dense_units = F_CFG["dense_units"]    # [128, 64]
    dropout     = F_CFG["dropout"]
    n_classes   = len(CLASSES)
    seq_len     = CFG["env"]["sequence_length"]
    n_feat      = 4                        # NPK, Temp, Humidity, Ph

    img_input = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    env_input = keras.Input(shape=(seq_len, n_feat), name="env_input")

    img_feats = img_ext(img_input, training=False)
    env_feats = lstm_ext(env_input, training=False)

    x = layers.Concatenate(name="fusion_concat")([img_feats, env_feats])
    x = layers.Dense(dense_units[0], activation="relu", name="fusion_dense1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units[1], activation="relu", name="fusion_dense2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fusion_output")(x)

    model = keras.Model([img_input, env_input], outputs, name="SAFE_FusionModel")
    model.compile(
        optimizer=keras.optimizers.Adam(F_CFG["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def make_fusion_dataset(split_name: str, env_X: np.ndarray, env_y: np.ndarray,
                         n_samples_per_class: int = None):
    """
    Creates (images_array, env_sequences_array, one_hot_labels) for a split.

    PAIRING LOGIC (SYNTHETIC — documented):
      disease images  -> randomly sampled env sequences where Diseased==1
      healthy images  -> randomly sampled env sequences where Diseased==0

    NOTE: This pairing is a proof-of-concept approximation.
    The two datasets are NOT from the same plants.
    """
    from PIL import Image as PILImage

    rescale = 1.0 / 255
    diseased_seqs = env_X[env_y == 1]
    healthy_seqs  = env_X[env_y == 0]

    all_imgs, all_seqs, all_labels = [], [], []

    for cls_idx, cls_name in enumerate(CLASSES):
        img_dir = SPLITS[split_name] / cls_name
        img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG")) + \
                    sorted(img_dir.glob("*.png"))

        if n_samples_per_class and len(img_paths) > n_samples_per_class:
            random.seed(SEED + cls_idx)
            img_paths = random.sample(img_paths, n_samples_per_class)

        pool = healthy_seqs if cls_idx == HEALTHY_IDX else diseased_seqs

        for img_path in img_paths:
            img = PILImage.open(img_path).convert("RGB").resize(IMG_SIZE)
            img_arr = np.array(img, dtype=np.float32) * rescale
            all_imgs.append(img_arr)

            seq_idx = random.randint(0, len(pool) - 1)
            all_seqs.append(pool[seq_idx])

            lbl = np.zeros(len(CLASSES), dtype=np.float32)
            lbl[cls_idx] = 1.0
            all_labels.append(lbl)

    imgs   = np.array(all_imgs,   dtype=np.float32)
    seqs   = np.array(all_seqs,   dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    # Shuffle
    idx = np.arange(len(imgs)); np.random.shuffle(idx)
    return imgs[idx], seqs[idx], labels[idx]


def plot_history(history, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip([ax1,ax2],["loss","accuracy"],["Loss","Accuracy"]):
        ax.plot(history.history[key],         label=f"Train {title}")
        ax.plot(history.history[f"val_{key}"],label=f"Val {title}")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Fusion Model Training History — S.A.F.E.", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
    print(f"OK: History -> {path}")


def evaluate_fusion(model, test_imgs, test_seqs, test_labels, split_label="Test"):
    y_pred_prob = model.predict([test_imgs, test_seqs], verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  [{split_label}] Accuracy:{acc:.4f}  Precision:{prec:.4f}  Recall:{rec:.4f}  F1:{f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix — Fusion Model ({split_label})\nS.A.F.E. Project")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(REP_DIR / f"confusion_matrix_fusion_{split_label.lower()}.png", dpi=100)
    plt.close()

    return {
        "model": "SAFE_FusionModel",
        "split": split_label,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "per_class": classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  S.A.F.E. — CNN-LSTM Fusion Model Training")
    print("=" * 60)
    print("\n  PAIRING NOTE: Images and env sequences are from INDEPENDENT")
    print("  datasets. Pairing is synthetic for architecture demonstration.\n")

    # Load env sequences
    npz = np.load(PROC_DIR / "env_sequences.npz")
    X_train_env, y_train_env = npz["X_train"], npz["y_train"]
    X_val_env,   y_val_env   = npz["X_val"],   npz["y_val"]
    X_test_env,  y_test_env  = npz["X_test"],  npz["y_test"]

    img_ext, lstm_ext = load_pretrained_branches()
    model = build_fusion_model(img_ext, lstm_ext)

    # Build datasets (test: sample 64/class for balanced eval)
    print("\nBuilding training data...")
    tr_imgs, tr_seqs, tr_lbls = make_fusion_dataset("Train", X_train_env, y_train_env)
    print(f"  Train: {tr_imgs.shape[0]} samples")
    print("Building validation data...")
    vl_imgs, vl_seqs, vl_lbls = make_fusion_dataset("Validate", X_val_env, y_val_env)
    print(f"  Val:   {vl_imgs.shape[0]} samples")
    print("Building test data (64 per class)...")
    ts_imgs, ts_seqs, ts_lbls = make_fusion_dataset("Test", X_test_env, y_test_env,
                                                      n_samples_per_class=64)
    print(f"  Test:  {ts_imgs.shape[0]} samples")

    t_cfg = F_CFG["training"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=t_cfg["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.3, verbose=1, min_lr=1e-6),
    ]

    history = model.fit(
        [tr_imgs, tr_seqs], tr_lbls,
        validation_data=([vl_imgs, vl_seqs], vl_lbls),
        epochs=t_cfg["epochs"],
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    fusion_path = MODELS_DIR / "fusion_model.keras"
    model.save(fusion_path)
    print(f"\nOK: Fusion model saved -> {fusion_path}")

    plot_history(history, REP_DIR / "training_history_fusion.png")
    metrics = evaluate_fusion(model, ts_imgs, ts_seqs, ts_lbls)

    # Load image-only metrics for comparison
    img_metrics_path = REP_DIR / "image_model_metrics.json"
    comparison = {"fusion_model": metrics}
    if img_metrics_path.exists():
        with open(img_metrics_path) as f:
            img_m = json.load(f)
        comparison["image_only_model"] = {
            "model": img_m["model"],
            "accuracy": img_m["test_accuracy"],
            "macro_f1": img_m["macro_f1"],
        }
        print("\n-- Model Comparison -------------------------------------")
        print(f"  Image-only  (MobileNetV2) : Acc={img_m['test_accuracy']:.4f}  F1={img_m['macro_f1']:.4f}")
        print(f"  Fusion (MobileNetV2+LSTM) : Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}")
        print("  NOTE: Fusion uses synthetic pairing — comparison is indicative only.")

    out = REP_DIR / "fusion_metrics.json"
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nOK: Fusion metrics saved -> {out}")
    print("\nOK: Fusion model training complete.")

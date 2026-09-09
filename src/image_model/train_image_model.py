"""
src/image_model/train_image_model.py
S.A.F.E. -- Step 3: MobileNetV2 Image Classifier Training

Architecture:
  MobileNetV2 (ImageNet, frozen base)
  -> GlobalAveragePooling2D
  -> Dense(256, relu) -- also used as feature extractor output
  -> Dropout(0.4)
  -> Dense(4, softmax)

Saves:
  models/mobilenetv2_classifier.keras   -- full 4-class model
  models/image_feature_extractor.keras  -- model up to Dense(256)
  reports/image_model_metrics.json      -- accuracy, F1, confusion matrix
  reports/training_history.png          -- loss/accuracy curves
  reports/confusion_matrix.png          -- test confusion matrix
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

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
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
RAW_ROOT   = Path(CFG["paths"]["raw_image_root"])
CLASSES    = CFG["image"]["classes"]
IMG_SIZE   = tuple(CFG["image"]["input_size"])   # (64, 64)
IMG_CFG    = CFG["image_model"]
AUG        = CFG["image"]["augmentation"]
BATCH      = IMG_CFG["training"]["batch_size"]

SPLITS = {
    "Train":    RAW_ROOT / "Train",
    "Validate": RAW_ROOT / "Validate",
    "Test":     RAW_ROOT / "Test",
}


# -----------------------------------------------------------------------------
def build_generators():
    aug = AUG
    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        horizontal_flip=aug["horizontal_flip"],
        zoom_range=aug["zoom_range"],
        brightness_range=aug["brightness_range"],
        fill_mode="nearest",
    )
    val_test_gen = ImageDataGenerator(rescale=1.0/255)

    kw = dict(target_size=IMG_SIZE, batch_size=BATCH, classes=CLASSES,
              class_mode="categorical", seed=SEED)

    train_flow = train_gen.flow_from_directory(str(SPLITS["Train"]), shuffle=True,  **kw)
    val_flow   = val_test_gen.flow_from_directory(str(SPLITS["Validate"]), shuffle=False, **kw)
    test_flow  = val_test_gen.flow_from_directory(str(SPLITS["Test"]),     shuffle=False, **kw)

    return train_flow, val_flow, test_flow


def build_model() -> keras.Model:
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False   # freeze during initial training

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu", name="image_features")(x)
    x = layers.Dropout(IMG_CFG["head"]["dropout"])(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="disease_class")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_SAFE")
    model.compile(
        optimizer=keras.optimizers.Adam(IMG_CFG["training"]["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_feature_extractor(full_model: keras.Model) -> keras.Model:
    """Return sub-model that outputs the 256-D feature vector."""
    return keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer("image_features").output,
        name="ImageFeatureExtractor",
    )


def plot_history(history, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],     label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"],     label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("MobileNetV2 Training History — S.A.F.E.", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"OK: Training history -> {out_path}")


def evaluate(model, test_flow):
    print("\n-- Evaluation on Test Set --")
    # For balanced eval: early_blight has 300 imgs in test folder.
    # We evaluate on ALL images but note the imbalance in the report.
    test_flow.reset()
    y_pred_prob = model.predict(test_flow, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = test_flow.classes

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"  Accuracy (macro) : {acc:.4f}")
    print(f"  Precision(macro) : {prec:.4f}")
    print(f"  Recall   (macro) : {rec:.4f}")
    print(f"  F1       (macro) : {f1:.4f}")
    print("\n" + classification_report(y_true, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — MobileNetV2 (Test Set)\nS.A.F.E. Project")
    plt.tight_layout()
    cm_path = REP_DIR / "confusion_matrix_image_model.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    print(f"OK: Confusion matrix -> {cm_path}")

    per_class = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    metrics = {
        "model": "MobileNetV2_classifier",
        "test_accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "note_test_class_imbalance": (
            "Test/early_blight contains 300 images vs 64 for other classes. "
            "Macro-averaged metrics partially correct for this."
        ),
    }
    out = REP_DIR / "image_model_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"OK: Metrics saved -> {out}")
    return metrics


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  S.A.F.E. — MobileNetV2 Image Classifier Training")
    print("=" * 60)

    train_flow, val_flow, test_flow = build_generators()
    model = build_model()

    t_cfg = IMG_CFG["training"]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=t_cfg["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=t_cfg["reduce_lr_factor"],
                          patience=t_cfg["reduce_lr_patience"], verbose=1, min_lr=1e-6),
        ModelCheckpoint(str(MODELS_DIR / "mobilenetv2_best.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=t_cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Save full classifier
    clf_path = MODELS_DIR / "mobilenetv2_classifier.keras"
    model.save(clf_path)
    print(f"\nOK: Full classifier saved -> {clf_path}")

    # Save feature extractor
    feat_ext = build_feature_extractor(model)
    feat_path = MODELS_DIR / "image_feature_extractor.keras"
    feat_ext.save(feat_path)
    print(f"OK: Feature extractor saved -> {feat_path}")

    plot_history(history, REP_DIR / "training_history_image_model.png")
    evaluate(model, test_flow)

    print("\nOK: Image model training complete.")

"""
src/data/prepare_image_data.py
S.A.F.E. -- Step 1: Image Dataset Validation & Preprocessing Verification
"""
from __future__ import annotations
import os, sys, json, random, warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["project"]["seed"]
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW_ROOT = Path(CFG["paths"]["raw_image_root"])
PROC_DIR = ROOT / CFG["paths"]["processed_dir"]
REP_DIR  = ROOT / CFG["paths"]["reports_dir"]
CLASSES  = CFG["image"]["classes"]
IMG_SIZE = tuple(CFG["image"]["input_size"])

SPLITS = {
    "Train":    RAW_ROOT / "Train",
    "Validate": RAW_ROOT / "Validate",
    "Test":     RAW_ROOT / "Test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_images(folder: Path) -> list:
    """Case-insensitive, dedup extension glob."""
    return [p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def count_images(split_root: Path) -> dict:
    return {cls: len(get_images(split_root / cls))
            if (split_root / cls).exists() else 0
            for cls in CLASSES}


def verify_dataset():
    print("=" * 60)
    print("  S.A.F.E. - Image Dataset Verification")
    print("=" * 60)
    report = {}
    for split_name, split_root in SPLITS.items():
        counts = count_images(split_root)
        report[split_name] = counts
        print(f"\n[{split_name}]")
        total = 0
        for cls, cnt in counts.items():
            note = ""
            if split_name == "Test" and cls == "early_blight" and cnt > 64:
                note = f"  (will sample 64 for balanced eval, found {cnt})"
            print(f"  {cls:<20} {cnt:>4}{note}")
            total += cnt
        print(f"  {'TOTAL':<20} {total:>4}")

    out = REP_DIR / "dataset_inspection.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nOK: Inspection report saved -> {out}")
    return report


def sample_grid(n_per_class: int = 4):
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                             figsize=(n_per_class * 2.5, len(CLASSES) * 2.5))
    fig.suptitle("Sample Training Images - S.A.F.E. Dataset", fontsize=13, fontweight="bold")

    for r, cls in enumerate(CLASSES):
        imgs = get_images(SPLITS["Train"] / cls)
        sample = random.sample(imgs, min(n_per_class, len(imgs)))
        for c, img_path in enumerate(sample):
            ax = axes[r, c]
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                ax.imshow(np.array(img))
            except Exception:
                ax.text(0.5, 0.5, "error", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls.replace("_", "\n"), fontsize=8, rotation=0,
                              labelpad=55, va="center")

    plt.tight_layout()
    out = REP_DIR / "sample_images_grid.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"OK: Sample grid saved -> {out}")


def build_data_generators():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    aug = CFG["image"]["augmentation"]
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        horizontal_flip=aug["horizontal_flip"],
        zoom_range=aug["zoom_range"],
        brightness_range=aug["brightness_range"],
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    kw = dict(target_size=IMG_SIZE,
              batch_size=CFG["image_model"]["training"]["batch_size"],
              classes=CLASSES, class_mode="categorical", seed=SEED)

    train_gen = train_datagen.flow_from_directory(str(SPLITS["Train"]), shuffle=True, **kw)
    val_gen   = val_test_datagen.flow_from_directory(str(SPLITS["Validate"]), shuffle=False, **kw)
    test_gen  = val_test_datagen.flow_from_directory(str(SPLITS["Test"]), shuffle=False, **kw)

    print(f"\nOK: Train   : {train_gen.samples} images, {train_gen.num_classes} classes")
    print(f"OK: Validate: {val_gen.samples} images")
    print(f"OK: Test    : {test_gen.samples} images")

    class_map_path = PROC_DIR / "class_indices.json"
    with open(class_map_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"OK: Class indices saved -> {class_map_path}")
    return train_gen, val_gen, test_gen


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    REP_DIR.mkdir(parents=True, exist_ok=True)
    verify_dataset()
    sample_grid()
    build_data_generators()
    print("\nOK: Image data preparation complete.")

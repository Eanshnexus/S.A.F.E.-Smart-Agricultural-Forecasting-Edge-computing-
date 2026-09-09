"""
run_pipeline.py
S.A.F.E. -- Master Pipeline Runner

Executes all training steps in order:
  1. Prepare image data (validate + generator test)
  2. Prepare environmental sequences
  3. Train MobileNetV2 image classifier
  4. Train LSTM temporal model
  5. Train CNN-LSTM fusion model
  6. Run final evaluation and comparison

Run from project root:
    python run_pipeline.py
"""
import sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
steps = [
    ("Image Data Validation",     ROOT / "src/data/prepare_image_data.py"),
    ("Environmental Data Prep",   ROOT / "src/data/prepare_env_data.py"),
    ("MobileNetV2 Training",      ROOT / "src/image_model/train_image_model.py"),
    ("LSTM Training",             ROOT / "src/temporal_model/train_lstm.py"),
    ("Fusion Model Training",     ROOT / "src/fusion/train_fusion.py"),
    ("Final Evaluation",          ROOT / "src/evaluation/evaluate_models.py"),
]

def run_step(name: str, script: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nFAIL: FAILED: {name}  (exit code {result.returncode})")
        return False
    print(f"\nOK: DONE: {name}  ({elapsed:.1f}s)")
    return True

if __name__ == "__main__":
    print("S.A.F.E. — Full ML Pipeline")
    failed = []
    for name, script in steps:
        if not run_step(name, script):
            failed.append(name)
            print("Aborting pipeline due to step failure.")
            break
    if not failed:
        print("\n" + "="*60)
        print("  ALL STEPS COMPLETED SUCCESSFULLY")
        print("  Models saved to: models/")
        print("  Reports saved to: reports/")
        print("="*60)
    else:
        print(f"\nFailed steps: {failed}")
        sys.exit(1)

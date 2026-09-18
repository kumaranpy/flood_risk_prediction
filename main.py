"""
End-to-End Pipeline Orchestrator for Flood Risk Prediction.

Executes all pipeline phases in sequence:
  Phase 1: Preprocessing, Empirical Target Binning, & Deterministic Splitting
  Phase 2: Exploratory Data Analysis & Visualizations
  Phase 3: Fold-Safe Pipeline Training, Tuning, & Probability Calibration
  Phase 4: Strict Held-out Test Split Evaluation & Report Generation
  Phase 5: Single-Instance Verified Inference Demonstration
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import run_data_preparation
from src.eda import run_eda
from src.train import run_training
from src.evaluate import run_evaluation
from src.predict import run_prediction


def main():
    start_time = time.time()
    print("=" * 70)
    print("=== REFACTORED PRODUCTION FLOOD RISK PREDICTION PIPELINE ===")
    print("=" * 70)

    try:
        p1_start = time.time()
        print("\n🚀 [Phase 1/5]: Data Preparation & Fold-Safe Splitting")
        run_data_preparation()
        print(f"⏱ Phase 1 completed in {time.time() - p1_start:.2f}s")

        p2_start = time.time()
        print("\n🚀 [Phase 2/5]: Exploratory Data Analysis & Plots")
        run_eda()
        print(f"⏱ Phase 2 completed in {time.time() - p2_start:.2f}s")

        p3_start = time.time()
        print("\n🚀 [Phase 3/5]: Fold-Safe Pipeline Training & Calibration")
        run_training()
        print(f"⏱ Phase 3 completed in {time.time() - p3_start:.2f}s")

        p4_start = time.time()
        print("\n🚀 [Phase 4/5]: Evaluation on Untouched Held-out Test Split")
        run_evaluation()
        print(f"⏱ Phase 4 completed in {time.time() - p4_start:.2f}s")

        p5_start = time.time()
        print("\n🚀 [Phase 5/5]: Single-Instance Verified Inference Demonstration")
        run_prediction()
        print(f"⏱ Phase 5 completed in {time.time() - p5_start:.2f}s")

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)!")
        print("=" * 70)
        print("\n📊 Launch the Interactive Web Dashboard:")
        print("   streamlit run app/dashboard.py")
        print("\n📁 Key Production Artifacts:")
        print("   - Calibrated Pipeline : models/best_pipeline.pkl")
        print("   - Checksums Registry  : models/checksums.json")
        print("   - Empirical Bins      : models/target_bins.json")
        print("   - Raw Split Data      : data/splits/ (train.csv, test.csv)")
        print("   - Evaluation Reports  : outputs/reports/ (model_comparison.csv, final_report.md)")
        print("   - Evaluation Plots    : outputs/plots/")

    except Exception as e:
        print(f"\n❌ PIPELINE EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
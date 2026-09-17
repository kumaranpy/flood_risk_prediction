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
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import run_preprocessing
from features import run_feature_engineering
from eda import run_eda
from train import run_training
from evaluate import run_evaluation
from predict import run_prediction


def main():
    start_time = time.time()
    print("=" * 70)
    print("=== END-TO-END FLOOD RISK PREDICTION PIPELINE ===")
    print("=" * 70)

    try:
        p1_start = time.time()
        print("\n🚀 [Phase 1/6]: Data Cleaning & Preprocessing (FR-01 - FR-06)")
        run_preprocessing()
        print(f"⏱ Phase 1 completed in {time.time() - p1_start:.2f}s")

        p2_start = time.time()
        print("\n🚀 [Phase 2/6]: Feature Engineering & Selection (FR-07 - FR-10)")
        run_feature_engineering()
        print(f"⏱ Phase 2 completed in {time.time() - p2_start:.2f}s")

        p3_start = time.time()
        print("\n🚀 [Phase 3/6]: Exploratory Data Analysis & Plots")
        run_eda()
        print(f"⏱ Phase 3 completed in {time.time() - p3_start:.2f}s")

        p4_start = time.time()
        print("\n🚀 [Phase 4/6]: Model Training & Tuning (FR-11 - FR-16)")
        run_training()
        print(f"⏱ Phase 4 completed in {time.time() - p4_start:.2f}s")

        p5_start = time.time()
        print("\n🚀 [Phase 5/6]: Model Evaluation & Reporting (FR-17 - FR-20)")
        run_evaluation()
        print(f"⏱ Phase 5 completed in {time.time() - p5_start:.2f}s")

        p6_start = time.time()
        print("\n🚀 [Phase 6/6]: Single-Instance Inference Demonstration")
        run_prediction()
        print(f"⏱ Phase 6 completed in {time.time() - p6_start:.2f}s")

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)!")
        print("=" * 70)
        print("\n📊 Launch the Interactive Web Dashboard:")
        print("   streamlit run app/dashboard.py")
        print("\n📁 Key Project Artifacts:")
        print("   - Trained Models : models/ (best_model.pkl, scaler.pkl, etc.)")
        print("   - Evaluation Plots: outputs/plots/")
        print("   - Comparison CSV : outputs/reports/model_comparison.csv")
        print("   - Final Report   : outputs/reports/final_report.md")
        print("   - Processed Data : data/processed/")

    except Exception as e:
        print(f"\n❌ PIPELINE EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
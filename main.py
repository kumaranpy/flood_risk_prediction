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

from src.data.preprocess import run_data_preparation
from src.eda import run_eda
from src.models.train import run_training
from src.models.evaluate import run_evaluation
from src.models.predict import run_prediction
from src.utils.logger import setup_logging


def main():
    setup_logging()
    import logging
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("=== REFACTORED PRODUCTION FLOOD RISK PREDICTION PIPELINE ===")
    logger.info("=" * 70)

    try:
        p1_start = time.time()
        logger.info("\n🚀 [Phase 1/5]: Data Preparation & Fold-Safe Splitting")
        run_data_preparation()
        logger.info(f"⏱ Phase 1 completed in {time.time() - p1_start:.2f}s")

        p2_start = time.time()
        logger.info("\n🚀 [Phase 2/5]: Exploratory Data Analysis & Plots")
        run_eda()
        logger.info(f"⏱ Phase 2 completed in {time.time() - p2_start:.2f}s")

        p3_start = time.time()
        logger.info("\n🚀 [Phase 3/5]: Fold-Safe Pipeline Training & Calibration")
        run_training()
        logger.info(f"⏱ Phase 3 completed in {time.time() - p3_start:.2f}s")

        p4_start = time.time()
        logger.info("\n🚀 [Phase 4/5]: Evaluation on Untouched Held-out Test Split")
        run_evaluation()
        logger.info(f"⏱ Phase 4 completed in {time.time() - p4_start:.2f}s")

        p5_start = time.time()
        logger.info("\n🚀 [Phase 5/5]: Single-Instance Verified Inference Demonstration")
        run_prediction()
        logger.info(f"⏱ Phase 5 completed in {time.time() - p5_start:.2f}s")

        total_elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)!")
        logger.info("=" * 70)
        logger.info("\n📊 Launch the Interactive Web Dashboard:")
        logger.info("   streamlit run app/dashboard.py")
        logger.info("\n📁 Key Production Artifacts:")
        logger.info("   - Calibrated Pipeline : models/v1/best_pipeline.pkl")
        logger.info("   - Checksums Registry  : models/checksums.json")
        logger.info("   - Empirical Bins      : models/target_bins.json")
        logger.info("   - Feature Medians     : models/feature_medians.json")
        logger.info("   - Model Registry      : models/registry.json")
        logger.info("   - Raw Split Data      : data/splits/ (train.csv, test.csv)")
        logger.info("   - Evaluation Reports  : outputs/reports/ (model_comparison.csv, final_report.md)")
        logger.info("   - Evaluation Plots    : outputs/plots/")

    except Exception as e:
        logger.error(f"\n❌ PIPELINE EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
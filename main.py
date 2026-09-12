import sys
from pathlib import Path

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
    print("=" * 70)
    print("=== FLOOD RISK PREDICTION PIPELINE ===")
    print("=" * 70)

    try:
        print("\n🚀 Starting Phase 1: Preprocessing")
        run_preprocessing()
        
        print("\n🚀 Starting Phase 2: Feature Engineering")
        run_feature_engineering()
        
        print("\n🚀 Starting Phase 3: EDA Plots")
        run_eda()
        
        print("\n🚀 Starting Phase 4: Model Training")
        run_training()
        
        print("\n🚀 Starting Phase 5: Model Evaluation")
        run_evaluation()
        
        print("\n🚀 Starting Phase 6: Sample Prediction")
        run_prediction()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print("\n📊 To view the dashboard, run:")
        print("   streamlit run app/dashboard.py")
        print("\n📁 Outputs saved to:")
        print("   - Models: models/")
        print("   - Plots: outputs/plots/")
        print("   - Reports: outputs/reports/")
        print("   - Processed data: data/processed/")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
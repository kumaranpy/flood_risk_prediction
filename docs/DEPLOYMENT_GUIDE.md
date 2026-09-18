# Deployment Guide — Flood Risk Prediction System

## Prerequisites

- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for training)
- Git

---

## 1. Local Development Setup

### Clone & Install
```bash
git clone <your-repo-url>
cd flood_risk_prediction

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
# Full pipeline (preprocess → train → evaluate → explain)
make all

# Or step by step:
make data      # Preprocess & split
make train     # Train all models + calibrate
make eval      # Evaluate on held-out test set
make explain   # SHAP explainability plots
make test      # Run test suite
```

### Launch Dashboard
```bash
streamlit run app/dashboard.py
# → Opens at http://localhost:8501
```

### Launch API Server
```bash
uvicorn api.main:app --reload --port 8000
# → API docs at http://localhost:8000/docs
# → Health check: curl http://localhost:8000/health
```

---

## 2. Streamlit Cloud Deployment

### Setup
1. Push project to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set **Main file**: `app/dashboard.py`
5. Set **Python version**: 3.11

### Environment Notes
- Streamlit Cloud has a 1 GB memory limit; KNN model may be excluded during deployment
- Pre-train models locally and commit `models/v1/best_pipeline.pkl` to Git LFS or use DVC
- Alternatively, add a startup script to run `python src/models/train.py` before launching

### secrets.toml (if needed)
```toml
# .streamlit/secrets.toml
# Add any API keys or environment variables here
```

---

## 3. Docker Deployment

### Build Image
```bash
# Build Docker image
make docker-build
# or
docker build -t flood_risk_app:latest .
```

### Run Container
```bash
docker run -p 8501:8501 -p 8000:8000 flood_risk_app:latest
```

### Run with Pre-trained Models (recommended)
```bash
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -p 8501:8501 \
  -p 8000:8000 \
  flood_risk_app:latest
```

### Multi-Service with Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run app/dashboard.py --server.port=8501 --server.address=0.0.0.0

  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Monitoring Setup

### Drift Detection
Run the drift detection script against new inference data:
```python
import pandas as pd
import yaml
from src.monitoring.drift import check_all_features

config = yaml.safe_load(open('src/config/config.yaml'))
train_df = pd.read_csv('data/splits/train.csv')
live_df = pd.read_csv('path/to/live_inference_data.csv')

results = check_all_features(train_df, live_df, config)
drifted = [k for k, v in results.items() if v.get('drifted')]
print(f"Drifted features: {drifted}")
```

### Prediction Logging
All predictions are logged via structured logging. Configure a log aggregator (e.g., CloudWatch, Datadog) to capture logs from the `flood_risk.monitoring` logger namespace.

### Retraining Trigger
Retrain when:
- PSI > 0.20 on any feature (significant drift)
- KS statistic > 0.20 on 3+ features
- F1-Weighted drops > 5% on a monitoring dataset

---

## 5. Troubleshooting

### `ModelNotFoundError: Model not found`
- Run `make train` to generate `models/v1/best_pipeline.pkl`
- Verify `models/checksums.json` exists

### `SecurityError: SHA-256 mismatch`
- Model file was tampered or corrupted
- Re-run `make train` to regenerate clean artifacts

### `InconsistentVersionWarning` from scikit-learn
- Model was saved with a different sklearn version
- Re-run `make train` with current environment to regenerate pkl files

### Dashboard shows blank / no data
- Ensure `outputs/reports/model_comparison.csv` exists (run `make eval`)
- Ensure `models/v1/best_pipeline.pkl` exists (run `make train`)

### API returns 503
- Model not loaded; check that `models/v1/best_pipeline.pkl` exists
- Check API logs for `ModelNotFoundError`

### Out of Memory During Training
- Reduce `RandomForest` `n_estimators` from 300 to 100 in `src/models/train.py`
- Skip the KNN model (it stores all training data)
- Set `n_jobs=1` instead of `-1`

---

## 6. Production Checklist

- [ ] Run `pytest tests/ -v` — all tests passing
- [ ] Verify `models/checksums.json` exists and is current
- [ ] Verify `models/v1/registry.json` has correct version metadata
- [ ] Test `/health` endpoint returns 200
- [ ] Test `/predict` with a valid payload
- [ ] Verify UNCERTAIN response for borderline inputs (confidence < 65%)
- [ ] Confirm no `print()` statements in `src/` or `api/` modules
- [ ] Confirm no plaintext secrets or credentials in codebase

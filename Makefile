.PHONY: all setup data train eval test dashboard api clean docker-build lint help

all: setup data train eval test

setup:
	pip install -r requirements.txt

data:
	python src/data/preprocess.py

train:
	python src/models/train.py

eval:
	python src/models/evaluate.py

test:
	pytest tests/ -v --tb=short

dashboard:
	streamlit run app/dashboard.py

api:
	uvicorn api.main:app --reload --port 8000

explain:
	python src/explain/shap_explainer.py

robustness:
	python src/monitoring/robustness.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -f outputs/plots/*.png outputs/reports/*.csv outputs/reports/*.md outputs/reports/*.json
	find . -name "*.pyc" -delete

docker-build:
	docker build -t flood_risk_app:latest .

docker-run:
	docker run -p 8501:8501 -p 8000:8000 flood_risk_app:latest

lint:
	ruff check src/ app/ api/ tests/ --fix
	black --check src/ app/ api/ tests/

format:
	black src/ app/ api/ tests/
	ruff check --fix src/ app/ api/ tests/

help:
	@echo "Available commands:"
	@echo "  make setup       - Install dependencies"
	@echo "  make data        - Run data preparation"
	@echo "  make train       - Train model pipeline"
	@echo "  make eval        - Evaluate on test set"
	@echo "  make test        - Run test suite"
	@echo "  make dashboard   - Launch Streamlit dashboard"
	@echo "  make api         - Launch FastAPI server"
	@echo "  make explain     - Run SHAP explainability"
	@echo "  make robustness  - Run robustness tests"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Auto-format code"
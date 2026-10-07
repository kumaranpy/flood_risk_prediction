# 🚀 AquaSense Deployment Guide: Streamlit Community Cloud

This guide provides step-by-step instructions for deploying **AquaSense: Intelligent Flood Risk Intelligence** to [Streamlit Community Cloud](https://streamlit.io/cloud) in under 3 minutes.

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure:
- [x] All runtime dependencies are in `requirements.txt` (development tools moved to `requirements-dev.txt`).
- [x] Model artifact (`models/best_pipeline.pkl`) and metadata (`models/checksums.json`, `models/target_bins.json`, `models/registry.json`) are committed to Git.
- [x] Data splits (`data/splits/train.csv`, `data/splits/test.csv`) are tracked in Git.
- [x] Evaluation plots (`outputs/plots/*.png`) and reports (`outputs/reports/*.csv`) are tracked in Git.
- [x] All file sizes are well under GitHub's 100 MB hard limit (`best_pipeline.pkl` is ~10 KB).
- [x] Secrets configuration is gitignored (`.streamlit/secrets.toml`).

---

## 🌐 Step-by-Step Streamlit Community Cloud Deployment

### Step 1: Push Repository to GitHub
Ensure all recent changes are pushed to your remote repository on GitHub:
```bash
git add .
git commit -m "deploy: AquaSense ready for Streamlit Cloud"
git push origin main
```

*(Optional GitHub Rebrand)*: If desired, go to **Settings** → **General** → **Repository name**, enter `aquasense`, and click **Rename**.

---

### Step 2: Create New App on Streamlit Cloud
1. Navigate to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click the **"New app"** button in the upper-right corner.
3. Fill in the deployment form:
   - **Repository**: `<your-github-username>/aquasense` (or `flood_risk_prediction`)
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
   - **App URL** (optional custom slug): `aquasense` $\rightarrow$ `https://aquasense.streamlit.app`

---

### Step 3: Configure Advanced Settings
Click **"Advanced settings..."** before deploying:
- **Python Version**: Select **`3.11`** (matches the verified `scikit-learn==1.5.1` environment).
- **Secrets** (Optional): If you have external telemetry or keys, paste TOML secrets here:
  ```toml
  [app]
  environment = "production"
  ```
  *(AquaSense runs out-of-the-box without requiring any mandatory secrets).*

---

### Step 4: Deploy & Verify
1. Click **"Deploy!"**.
2. Streamlit Cloud will:
   - Provision an isolated Python 3.11 container.
   - Install dependencies from `requirements.txt`.
   - Launch `app/dashboard.py` with multi-page navigation from `app/pages/`.
3. Expected build time: **~90 seconds**.
4. Expected RAM usage: **~220 MB** (well within Streamlit Cloud's 1 GB free-tier limit).

---

## 🔍 How AquaSense Ensures Cloud Compatibility

| Challenge | AquaSense Solution |
|---|---|
| **RAM Limits (1 GB limit)** | Heavy development tools (`pytest`, `black`, `ruff`) separated into `requirements-dev.txt`. Models use lightweight regularized pipelines (< 10 KB). |
| **Path Invariance** | All asset and data paths use `Path(__file__).resolve().parents[...]` — zero hardcoded paths, runs identically from any working directory. |
| **Deserialization Security** | SHA-256 hash verified against `checksums.json` before deserializing `best_pipeline.pkl`. |
| **Multi-Page Sidebar** | Subpages inside `app/pages/` (1 to 5) are natively discovered and styled by AquaSense's custom sidebar component. |
| **Theme Consistency** | `.streamlit/config.toml` enforces the Deep-Ocean Dark palette directly in Streamlit's engine. |

---

## 🛠️ Local Verification Command

To test the exact Streamlit Cloud startup command locally:
```bash
# In your virtual environment
streamlit run app/dashboard.py --server.port 8501 --server.headless true
```
Open `http://localhost:8501` to confirm all 6 pages render smoothly.

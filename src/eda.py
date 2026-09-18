"""
Exploratory Data Analysis (EDA) Module for Flood Risk Prediction.

Analyzes the training partition (data/splits/train.csv) and generates
publication-quality visualizations in outputs/plots/.
"""

from pathlib import Path
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import DomainFeatureAdder

warnings.filterwarnings("ignore")

TRAIN_SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
ORDER = ["Low", "Medium", "High"]


def run_eda():
    """
    Executes Exploratory Data Analysis on the raw training split and domain features.
    """
    print("=" * 60)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (TRAINING PARTITION)")
    print("=" * 60)

    if not TRAIN_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Training split not found: {TRAIN_SPLIT_PATH}. Run preprocess.py first.")

    print(f"Loading training split from: {TRAIN_SPLIT_PATH}")
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)

    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_raw = train_df[feature_cols].copy()
    y = train_df["RiskLevel"].astype(str)

    # Apply domain feature adder
    adder = DomainFeatureAdder()
    X = adder.fit_transform(X_raw)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Target: 'RiskLevel', Classes: {sorted(y.unique())}")
    print(f"Total features with domain features: {len(numeric_cols)}")

    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Class Distribution Bar Chart
    print("\n--- 1. Generating Class Distribution Plot ---")
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = y.value_counts().reindex(ORDER, fill_value=0)
    total_samples = len(y)
    colors = [PALETTE.get(c, "#3498DB") for c in ORDER]
    bars = ax.bar(ORDER, class_counts.values, color=colors, edgecolor="black", linewidth=1.2, width=0.55)

    for bar, count in zip(bars, class_counts.values):
        pct = (count / total_samples) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.015 * total_samples),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_title("Flood Risk Class Distribution (Training Partition)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Risk Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    plt.tight_layout()
    dist_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dist_path}")

    # 2. Correlation Heatmap (Top 12 Features)
    print("\n--- 2. Generating Correlation Heatmap ---")
    selector = SelectKBest(f_classif, k=min(12, len(numeric_cols)))
    selector.fit(X[numeric_cols], y)
    top12_cols = X[numeric_cols].columns[selector.get_support()].tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = X[top12_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.2,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix (Top 12 Features)", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {heatmap_path}")

    # 3. Numeric Feature Histograms (Top 8 Features)
    print("\n--- 3. Generating Numeric Feature Histograms ---")
    top8_cols = top12_cols[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, col in enumerate(top8_cols):
        ax = axes[i]
        for cls in ORDER:
            cls_data = X[y == cls][col]
            ax.hist(cls_data, bins=25, alpha=0.55, label=cls, density=True, color=PALETTE.get(cls, "#3498DB"))
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Raw Factor Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Feature Distributions Across Flood Risk Categories", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    hist_path = PLOTS_DIR / "histograms_numeric.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {hist_path}")

    # 4. Boxplots: Top 6 Features by Target
    print("\n--- 4. Generating Boxplots of Top 6 Features ---")
    top6 = top12_cols[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    plot_df = X[top6].copy()
    plot_df["RiskLevel"] = y.values

    for i, col in enumerate(top6):
        ax = axes[i]
        sns.boxplot(data=plot_df, x="RiskLevel", y=col, order=ORDER, palette=PALETTE, ax=ax)
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Raw Value", fontsize=9)
        ax.set_xlabel("Risk Level", fontsize=9)

    plt.suptitle("Top 6 Predictive Features by Risk Level", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    boxplot_path = PLOTS_DIR / "boxplots_top6_by_target.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {boxplot_path}")

    # 5. Pairplot of Top 4 Features
    print("\n--- 5. Generating Pairplot (Top 4 Features) ---")
    top4 = top12_cols[:4]
    sample_size = min(1200, len(X))
    sample_indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
    pair_df = X.iloc[sample_indices][top4].copy()
    pair_df["RiskLevel"] = y.iloc[sample_indices].values

    g = sns.pairplot(
        pair_df,
        hue="RiskLevel",
        hue_order=ORDER,
        palette=PALETTE,
        diag_kind="kde",
        plot_kws={"alpha": 0.45, "s": 15},
    )
    g.fig.suptitle("Multivariate Separation Across Top 4 Predictors", y=1.02, fontsize=13, fontweight="bold")
    pairplot_path = PLOTS_DIR / "pairplot_top5.png"
    plt.savefig(pairplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pairplot_path}")

    print("=" * 60)
    print("PHASE 2: EDA PLOTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Standard risk palette and ordering
PALETTE = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
ORDER = ["Low", "Medium", "High"]


def auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column: prefers FloodProbability, RiskLevel, or last column."""
    priority_cols = ["RiskLevel", "FloodProbability", "target", "label", "risk_level"]
    for col in priority_cols:
        if col in df.columns:
            return col
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        col_lower = col.lower()
        if "score" in col_lower or "index" in col_lower or "vulnerability" in col_lower:
            continue
        if any(kw in col_lower for kw in target_keywords):
            return col
    return df.columns[-1]


def run_eda():
    """
    Executes Phase 3: Exploratory Data Analysis, generating 6 publication-ready plots in outputs/plots/.
    """
    print("=" * 60)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) STARTED")
    print("=" * 60)

    print(f"\nLoading features from: {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Target: '{target_col}', Classes: {sorted(y.unique())}")
    print(f"Features ({len(numeric_cols)}): {numeric_cols}")

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

    ax.set_title("Flood Risk Class Distribution (Target)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Risk Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(class_counts.values) * 1.18)
    plt.tight_layout()
    dist_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {dist_path}")

    # 2. Correlation Heatmap
    print("\n--- 2. Generating Correlation Heatmap ---")
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
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
    ax.set_title("Feature Correlation Matrix (Top 12 Selected Features)", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {heatmap_path}")

    # 3. Numeric Feature Histograms
    print("\n--- 3. Generating Numeric Feature Histograms ---")
    n_features = len(numeric_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for cls in ORDER:
            cls_data = X[y == cls][col]
            ax.hist(cls_data, bins=25, alpha=0.55, label=cls, density=True, color=PALETTE.get(cls, "#3498DB"))
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Scaled Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions Across Flood Risk Categories", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    hist_path = PLOTS_DIR / "histograms_numeric.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {hist_path}")

    # 4. Boxplots: Top 6 Features by Target
    print("\n--- 4. Generating Boxplots of Top 6 Features ---")
    selector6 = SelectKBest(f_classif, k=min(6, len(numeric_cols)))
    selector6.fit(X[numeric_cols], y)
    top6 = X[numeric_cols].columns[selector6.get_support()].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, col in enumerate(top6):
        ax = axes[i]
        sns.boxplot(data=df, x=target_col, y=col, order=ORDER, palette=PALETTE, ax=ax)
        ax.set_title(f"{col}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Value", fontsize=9)
        ax.set_xlabel("Risk Level", fontsize=9)

    plt.suptitle("Top 6 Predictive Features by Risk Level", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    boxplot_path = PLOTS_DIR / "boxplots_top6_by_target.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {boxplot_path}")

    # 5. Pairplot of Top 4 Features (Sampled for speed & clarity)
    print("\n--- 5. Generating Pairplot (Top 4 Features) ---")
    selector4 = SelectKBest(f_classif, k=min(4, len(numeric_cols)))
    selector4.fit(X[numeric_cols], y)
    top4 = X[numeric_cols].columns[selector4.get_support()].tolist()

    sample_indices = np.random.RandomState(42).choice(len(X), size=min(1500, len(X)), replace=False)
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
    g.fig.suptitle("Multivariate Separation Across Top 4 Predictors", y=1.02, fontsize=14, fontweight="bold")
    pairplot_path = PLOTS_DIR / "pairplot_top5.png"
    plt.savefig(pairplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pairplot_path}")

    # 6. Flood Vulnerability Score Distribution
    print("\n--- 6. Generating Vulnerability Distribution Plot ---")
    if "Flood_Vulnerability_Score" in X.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cls in ORDER:
            sns.kdeplot(
                X[y == cls]["Flood_Vulnerability_Score"],
                label=f"{cls} Risk",
                color=PALETTE.get(cls, "#3498DB"),
                fill=True,
                alpha=0.35,
                linewidth=2,
                ax=ax,
            )
        ax.set_title("Distribution of Engineered Flood Vulnerability Score by Risk Tier", fontsize=13, fontweight="bold")
        ax.set_xlabel("Flood Vulnerability Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.legend(title="Risk Level", fontsize=10)
        plt.tight_layout()
        vuln_path = PLOTS_DIR / "vulnerability_distribution.png"
        plt.savefig(vuln_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {vuln_path}")

    print("=" * 60)
    print("PHASE 3: EDA PLOTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
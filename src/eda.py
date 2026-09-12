import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def auto_detect_target(df: pd.DataFrame) -> str:
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def run_eda():
    print("=" * 60)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (PLOTS)")
    print("=" * 60)

    print(f"\nLoading features from: {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Target: {target_col}, Classes: {sorted(y.unique())}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")

    class_names = sorted(y.unique().astype(str))
    n_classes = len(class_names)

    print("\n--- 1. HISTOGRAMS FOR NUMERIC FEATURES ---")
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for cls in class_names:
            cls_data = X[y.astype(str) == cls][col]
            ax.hist(cls_data, bins=30, alpha=0.5, label=f"Class {cls}", density=True)
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    hist_path = PLOTS_DIR / "histograms_numeric.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {hist_path}")

    print("\n--- 2. CORRELATION HEATMAP ---")
    corr = X.corr()
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                annot_kws={"size": 7})
    plt.title("Feature Correlation Heatmap (Lower Triangle)", fontsize=14)
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {heatmap_path}")

    print("\n--- 3. BOXPLOTS: TOP 6 FEATURES BY TARGET ---")
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=min(6, len(numeric_cols)))
    selector.fit(X[numeric_cols], y)
    top6 = X[numeric_cols].columns[selector.get_support()].tolist()
    
    n_cols = 3
    n_rows = int(np.ceil(len(top6) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    for i, col in enumerate(top6):
        ax = axes[i]
        data_to_plot = [X[y.astype(str) == cls][col].values for cls in class_names]
        bp = ax.boxplot(data_to_plot, labels=class_names, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f"{col} by {target_col}")
        ax.set_ylabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    boxplot_path = PLOTS_DIR / "boxplots_top6_by_target.png"
    plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {boxplot_path}")

    print("\n--- 4. CLASS DISTRIBUTION BAR CHART ---")
    class_counts = y.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(c) for c in class_counts.index], class_counts.values, 
                   color=plt.cm.Set2(np.linspace(0, 1, n_classes)))
    plt.title(f"Class Distribution: {target_col}", fontsize=14)
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(count), ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    class_dist_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(class_dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {class_dist_path}")

    print("\n--- 5. PAIRPLOT OF TOP 5 FEATURES ---")
    selector5 = SelectKBest(f_classif, k=min(5, len(numeric_cols)))
    selector5.fit(X[numeric_cols], y)
    top5 = X[numeric_cols].columns[selector5.get_support()].tolist()
    pairplot_df = X[top5].copy()
    pairplot_df[target_col] = y.astype(str).values
    
    g = sns.pairplot(pairplot_df, hue=target_col, palette="Set2", 
                     diag_kind="hist", plot_kws={"alpha": 0.6, "s": 20})
    g.fig.suptitle("Pairplot of Top 5 Features by Risk Level", y=1.02, fontsize=14)
    pairplot_path = PLOTS_DIR / "pairplot_top5.png"
    plt.savefig(pairplot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pairplot_path}")

    print("\n" + "=" * 60)
    print("PHASE 3: EDA PLOTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
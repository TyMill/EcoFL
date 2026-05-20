"""
EcoFL — Visualization & Figure Generation
==========================================
Generates all publication-quality figures from benchmark results.

Figures produced
----------------
Fig 1. Architecture diagram (see paper_draft.md)
Fig 2. Pareto frontier: Accuracy vs Energy
Fig 3. Bar chart: ML metrics per model × configuration
Fig 4. Bar chart: Energy consumption comparison
Fig 5. Heatmap: all metrics × model × configuration
Fig 6. Line plot: FL convergence (accuracy vs rounds)
Fig 7. Bar chart: Communication overhead
Fig 8. Latency comparison
Fig 9. EcoFL vs FedAvg energy reduction (%)
Fig 10. Scalability: rounds run by EcoFL vs FedAvg

Usage
-----
    python experiments/visualize_results.py
    python experiments/visualize_results.py --results results/aggregated_results.json
    python experiments/visualize_results.py --format pdf   # or png
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────

PALETTE = {
    "centralized": "#2196F3",   # blue
    "fedavg":      "#FF9800",   # orange
    "ecofl":       "#4CAF50",   # green
}
MODEL_ORDER = [
    "LogisticRegression", "RandomForest",
    "XGBoost", "MLP", "IsolationForest",
]
MODEL_SHORT = {
    "LogisticRegression": "LR",
    "RandomForest":       "RF",
    "XGBoost":            "XGB",
    "MLP":                "MLP",
    "IsolationForest":    "IF",
}
CONFIG_ORDER  = ["centralized", "fedavg", "ecofl"]
CONFIG_LABELS = {"centralized": "Centralized", "fedavg": "FedAvg", "ecofl": "EcoFL"}

def set_style():
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "figure.dpi":       150,
    })


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        ml = r["ml_metrics"]

        def _val(x):
            return x["mean"] if isinstance(x, dict) else x

        def _std(x):
            return x.get("std", 0.0) if isinstance(x, dict) else 0.0

        row = {
            "Model":         r["model_name"],
            "ModelShort":    MODEL_SHORT.get(r["model_name"], r["model_name"]),
            "Config":        r["configuration"],
            "ConfigLabel":   CONFIG_LABELS.get(r["configuration"], r["configuration"]),
            "Accuracy":      _val(ml["accuracy"]),
            "Accuracy_std":  _std(ml["accuracy"]),
            "F1":            _val(ml["f1"]),
            "F1_std":        _std(ml["f1"]),
            "ROCAUC":        _val(ml["roc_auc"]),
            "ROCAUC_std":    _std(ml["roc_auc"]),
            "Precision":     _val(ml["precision"]),
            "Recall":        _val(ml["recall"]),
            "Energy_mJ":     _val(r["total_energy_mj"]),
            "Energy_std":    _std(r["total_energy_mj"]),
            "Latency_ms":    _val(r["inference_latency_mean_ms"]),
            "Latency_std":   _std(r["inference_latency_mean_ms"]),
            "RAM_MB":        _val(r["peak_ram_mb"]),
            "RAM_std":       _std(r["peak_ram_mb"]),
            "CommKB":        _val(r["communication_kb"]),
            "CommKB_std":    _std(r["communication_kb"]),
            "Rounds":        _val(r["n_rounds"]),
            "Rounds_std":    _std(r["n_rounds"]),
            "TrainTime_s":   _val(r["training_time_s"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure categorical order
    df["Model"]  = pd.Categorical(df["Model"],  categories=MODEL_ORDER,  ordered=True)
    df["Config"] = pd.Categorical(df["Config"], categories=CONFIG_ORDER, ordered=True)
    return df.sort_values(["Model", "Config"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────

def save_fig(fig, name: str, out_dir: str, fmt: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────
# Fig 2: Pareto Frontier
# ─────────────────────────────────────────────

def fig_pareto(df: pd.DataFrame, out_dir: str, fmt: str):
    """
    Pareto frontier: F1-Score vs Training Energy (log X axis).
    Single panel, log scale, clean label positioning.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom label offsets per model per config to avoid overlaps
    # format: (model, config) -> (dx, dy)
    label_offsets = {
        ("LogisticRegression", "centralized"): ( -26,   7),
        ("LogisticRegression", "fedavg"):      (   6,   6),
        ("LogisticRegression", "ecofl"):       (   6, -13),
        ("RandomForest",       "centralized"): (   6,   6),
        ("RandomForest",       "fedavg"):      (   6, -13),
        ("RandomForest",       "ecofl"):       ( -26,   6),
        ("XGBoost",            "centralized"): ( -26, -13),
        ("XGBoost",            "fedavg"):      (   6,   6),
        ("XGBoost",            "ecofl"):       (   6, -13),
        ("MLP",                "centralized"): (   6,   6),
        ("MLP",                "fedavg"):      (   6, -13),
        ("MLP",                "ecofl"):       ( -30,   6),
        ("IsolationForest",    "centralized"): ( -26, -13),
        ("IsolationForest",    "fedavg"):      (   6,   6),
        ("IsolationForest",    "ecofl"):       (   6,  -13),
    }

    for cfg in CONFIG_ORDER:
        sub   = df[df["Config"] == cfg]
        color = PALETTE[cfg]

        ax.scatter(
            sub["Energy_mJ"], sub["F1"],
            c=color, s=120, zorder=4,
            edgecolors="white", linewidths=0.9,
            label=CONFIG_LABELS[cfg],
        )

        # X-only error bars — cap at 40% of value on log scale
        xerr = np.minimum(sub["Energy_std"].values, sub["Energy_mJ"].values * 0.4)
        xerr = np.clip(xerr, 0.01, None)
        ax.errorbar(
            sub["Energy_mJ"], sub["F1"],
            xerr=xerr,
            fmt="none", ecolor=color, alpha=0.20,
            capsize=2, elinewidth=0.7, zorder=3,
        )

        for _, row in sub.iterrows():
            dx, dy = label_offsets.get(
                (row["Model"], cfg), (6, 5)
            )
            ax.annotate(
                row["ModelShort"],
                (row["Energy_mJ"], row["F1"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5, color=color, fontweight="bold",
            )

    # ── Highlight EcoFL dominance vs FedAvg with arrows ──────────
    for model in MODEL_ORDER:
        fa = df[(df["Model"] == model) & (df["Config"] == "fedavg")]
        ec = df[(df["Model"] == model) & (df["Config"] == "ecofl")]
        if fa.empty or ec.empty:
            continue
        ax.annotate(
            "",
            xy    =(ec["Energy_mJ"].values[0], ec["F1"].values[0]),
            xytext=(fa["Energy_mJ"].values[0], fa["F1"].values[0]),
            arrowprops=dict(
                arrowstyle="->",
                color="grey", lw=0.8, alpha=0.45,
            ),
            zorder=2,
        )

    # ── Axes ─────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlabel("Total Training Energy (mJ, log scale)", fontsize=11)
    ax.set_ylabel("F1-Score (macro)", fontsize=11)
    ax.set_title("Pareto Frontier: F1-Score vs Training Energy\n"
                 "(arrows show FedAvg → EcoFL shift per model)", fontsize=11)
    ax.set_ylim(0.50, 0.72)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(framealpha=0.92, fontsize=10, loc="upper right")

    fig.tight_layout()
    save_fig(fig, "fig2_pareto_frontier", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 3: ML Metrics grouped bar chart
# ─────────────────────────────────────────────

def fig_ml_metrics(df: pd.DataFrame, out_dir: str, fmt: str):
    """Grouped bar chart: Accuracy and F1 per model × configuration."""
    models  = [m for m in MODEL_ORDER if m in df["Model"].values]
    n_model = len(models)
    n_cfg   = len(CONFIG_ORDER)
    width   = 0.25
    x       = np.arange(n_model)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("Accuracy", "Accuracy",    "Predictive Accuracy"),
        ("F1",       "F1 (macro)",  "F1-Score (macro)"),
    ]):
        ax = axes[ax_idx]
        for ci, cfg in enumerate(CONFIG_ORDER):
            vals = []
            errs = []
            for model in models:
                row = df[(df["Model"] == model) & (df["Config"] == cfg)]
                vals.append(row[metric].values[0] if len(row) else 0.0)
                errs.append(row[f"{metric}_std"].values[0] if len(row) else 0.0)

            offset = (ci - 1) * width
            bars   = ax.bar(
                x + offset, vals, width,
                label=CONFIG_LABELS[cfg],
                color=PALETTE[cfg],
                alpha=0.85,
                yerr=errs,
                capsize=3,
                error_kw={"elinewidth": 1.2},
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_SHORT[m] for m in models], fontsize=10
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if ax_idx == 1:
            ax.legend(framealpha=0.9)

    fig.suptitle("ML Performance: Centralized vs FedAvg vs EcoFL", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig3_ml_metrics", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 4: Energy comparison bar chart
# ─────────────────────────────────────────────

def fig_energy(df: pd.DataFrame, out_dir: str, fmt: str):
    """Energy consumption per model × configuration."""
    models  = [m for m in MODEL_ORDER if m in df["Model"].values]
    n_model = len(models)
    width   = 0.25
    x       = np.arange(n_model)

    fig, ax = plt.subplots(figsize=(10, 5))

    for ci, cfg in enumerate(CONFIG_ORDER):
        vals = []
        errs = []
        for model in models:
            row = df[(df["Model"] == model) & (df["Config"] == cfg)]
            vals.append(row["Energy_mJ"].values[0] if len(row) else 0.0)
            errs.append(row["Energy_std"].values[0] if len(row) else 0.0)

        offset = (ci - 1) * width
        ax.bar(
            x + offset, vals, width,
            label=CONFIG_LABELS[cfg],
            color=PALETTE[cfg],
            alpha=0.85,
            yerr=errs,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.set_ylabel("Total Training Energy (mJ, log scale)")
    ax.set_title("Energy Consumption: Centralized vs FedAvg vs EcoFL")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig4_energy", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 5: Heatmap — all metrics
# ─────────────────────────────────────────────

def fig_heatmap(df: pd.DataFrame, out_dir: str, fmt: str):
    """Heatmap of all key metrics across models × configurations."""
    metrics = ["Accuracy", "F1", "ROCAUC", "Energy_mJ",
               "Latency_ms", "RAM_MB", "CommKB"]
    metric_labels = ["Accuracy", "F1", "ROC-AUC", "Energy\n(mJ)",
                     "Latency\n(ms)", "RAM\n(MB)", "Comm.\n(KB)"]

    rows_idx  = []
    data_rows = []

    for model in MODEL_ORDER:
        for cfg in CONFIG_ORDER:
            sub = df[(df["Model"] == model) & (df["Config"] == cfg)]
            if len(sub) == 0:
                continue
            rows_idx.append(f"{MODEL_SHORT[model]}-{CONFIG_LABELS[cfg][:3]}")
            data_rows.append([sub[m].values[0] for m in metrics])

    mat = np.array(data_rows, dtype=float)

    # Normalise each column [0, 1] for colour mapping
    mat_norm = mat.copy()
    for j in range(mat.shape[1]):
        col_min = mat[:, j].min()
        col_max = mat[:, j].max()
        rng     = col_max - col_min
        if rng > 0:
            mat_norm[:, j] = (mat[:, j] - col_min) / rng

    # Invert energy / latency / RAM / comm (lower = better)
    for j in [3, 4, 5, 6]:
        mat_norm[:, j] = 1 - mat_norm[:, j]

    fig, ax = plt.subplots(figsize=(11, max(6, len(rows_idx) * 0.5)))
    im = ax.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Annotate with raw values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = f"{v:.3f}" if j <= 2 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(len(rows_idx)))
    ax.set_yticklabels(rows_idx, fontsize=9)
    ax.set_title("Metric Heatmap (green = better, normalised per column)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalised score")

    fig.tight_layout()
    save_fig(fig, "fig5_heatmap", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 6: Convergence curves (from round history)
# ─────────────────────────────────────────────

def fig_convergence(history_path: str, out_dir: str, fmt: str):
    """FL convergence: accuracy vs round for FedAvg and EcoFL."""
    if not os.path.exists(history_path):
        print(f"  ⚠ Round history not found at {history_path}, skipping Fig 6.")
        return

    with open(history_path) as f:
        history = json.load(f)

    models  = [m for m in MODEL_ORDER if any(m in k for k in history.keys())]
    n_plots = len(models)
    if n_plots == 0:
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=False)
    if n_plots == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]

        for cfg, color in [("fedavg", PALETTE["fedavg"]),
                           ("ecofl",  PALETTE["ecofl"])]:
            key = f"{model}_{cfg}"
            if key not in history:
                continue
            rounds = history[key]
            rnd_nums = [r["round"]    for r in rounds]
            accs     = [r["accuracy"] for r in rounds]
            ax.plot(rnd_nums, accs, marker="o", markersize=4,
                    color=color, label=CONFIG_LABELS[cfg], linewidth=1.8)

        ax.set_title(MODEL_SHORT.get(model, model))
        ax.set_xlabel("FL Round")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("FL Convergence: Accuracy vs Communication Round", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig6_convergence", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 7: Communication overhead
# ─────────────────────────────────────────────

def fig_communication(df: pd.DataFrame, out_dir: str, fmt: str):
    """Total communication overhead (KB) for FL configurations."""
    fl_df = df[df["Config"].isin(["fedavg", "ecofl"])]

    models = [m for m in MODEL_ORDER if m in fl_df["Model"].values]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cfg in enumerate(["fedavg", "ecofl"]):
        vals = []
        for model in models:
            row = fl_df[(fl_df["Model"] == model) & (fl_df["Config"] == cfg)]
            vals.append(row["CommKB"].values[0] if len(row) else 0.0)
        offset = (ci - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=CONFIG_LABELS[cfg], color=PALETTE[cfg], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.set_ylabel("Total Communication Overhead (KB, log scale)")
    ax.set_title("Communication Cost: FedAvg vs EcoFL")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig7_communication", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 8: Inference latency
# ─────────────────────────────────────────────

def fig_latency(df: pd.DataFrame, out_dir: str, fmt: str):
    """Inference latency comparison across models and configs."""
    models = [m for m in MODEL_ORDER if m in df["Model"].values]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for ci, cfg in enumerate(CONFIG_ORDER):
        vals = []
        errs = []
        for model in models:
            row = df[(df["Model"] == model) & (df["Config"] == cfg)]
            vals.append(row["Latency_ms"].values[0] if len(row) else 0.0)
            errs.append(row["Latency_std"].values[0] if len(row) else 0.0)
        offset = (ci - 1) * width
        ax.bar(x + offset, vals, width,
               label=CONFIG_LABELS[cfg], color=PALETTE[cfg],
               alpha=0.85, yerr=errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.set_ylabel("Inference Latency (ms, mean over 5 runs)")
    ax.set_title("Inference Latency per Model and Configuration")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig8_latency", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 9: EcoFL energy reduction %
# ─────────────────────────────────────────────

def fig_energy_reduction(df: pd.DataFrame, out_dir: str, fmt: str):
    """
    Percentage energy reduction of EcoFL vs FedAvg, per model.
    Negative = EcoFL uses more (unexpected); positive = savings.
    """
    models   = [m for m in MODEL_ORDER if m in df["Model"].values]
    savings  = []
    for model in models:
        fa_row = df[(df["Model"] == model) & (df["Config"] == "fedavg")]
        ec_row = df[(df["Model"] == model) & (df["Config"] == "ecofl")]
        if len(fa_row) and len(ec_row):
            fa_e = fa_row["Energy_mJ"].values[0]
            ec_e = ec_row["Energy_mJ"].values[0]
            pct  = (fa_e - ec_e) / (fa_e + 1e-8) * 100
            savings.append(pct)
        else:
            savings.append(0.0)

    colors = [PALETTE["ecofl"] if s > 0 else "#F44336" for s in savings]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        [MODEL_SHORT[m] for m in models],
        savings,
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Energy Saving vs FedAvg (%)")
    ax.set_title("EcoFL Energy Reduction Relative to Standard FedAvg")

    for bar, val in zip(bars, savings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (1 if val >= 0 else -3),
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig9_energy_reduction", out_dir, fmt)


# ─────────────────────────────────────────────
# Fig 10: Rounds comparison
# ─────────────────────────────────────────────

def fig_rounds(df: pd.DataFrame, out_dir: str, fmt: str):
    """Number of FL rounds: FedAvg vs EcoFL (early termination effect)."""
    fl_df  = df[df["Config"].isin(["fedavg", "ecofl"])]
    models = [m for m in MODEL_ORDER if m in fl_df["Model"].values]
    x      = np.arange(len(models))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cfg in enumerate(["fedavg", "ecofl"]):
        vals = []
        errs = []
        for model in models:
            row = fl_df[(fl_df["Model"] == model) & (fl_df["Config"] == cfg)]
            vals.append(row["Rounds"].values[0] if len(row) else 0.0)
            errs.append(row["Rounds_std"].values[0] if len(row) else 0.0)
        offset = (ci - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=CONFIG_LABELS[cfg], color=PALETTE[cfg],
               alpha=0.85, yerr=errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models])
    ax.set_ylabel("Number of FL Rounds")
    ax.set_title("FL Rounds: FedAvg (fixed) vs EcoFL (adaptive termination)")
    ax.legend(framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "fig10_rounds", out_dir, fmt)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EcoFL Visualization")
    p.add_argument("--results",
                   default="results/aggregated_results.json",
                   help="Path to aggregated_results.json")
    p.add_argument("--history",
                   default="results/seed_42/round_history.json",
                   help="Path to round_history.json")
    p.add_argument("--out",
                   default="results/figures",
                   help="Output directory for figures")
    p.add_argument("--format", choices=["pdf", "png", "svg"],
                   default="pdf", help="Output format (default: pdf)")
    return p.parse_args()


def main():
    args = parse_args()
    set_style()

    print(f"\n[EcoFL Viz] Loading results from {args.results}")
    df = load_data(args.results)

    print(f"[EcoFL Viz] Generating figures → {args.out}/")
    fig_pareto(df, args.out, args.format)
    fig_ml_metrics(df, args.out, args.format)
    fig_energy(df, args.out, args.format)
    fig_heatmap(df, args.out, args.format)
    fig_convergence(args.history, args.out, args.format)
    fig_communication(df, args.out, args.format)
    fig_latency(df, args.out, args.format)
    fig_energy_reduction(df, args.out, args.format)
    fig_rounds(df, args.out, args.format)

    print(f"\n[EcoFL Viz] ✓ All figures saved to {args.out}/")


if __name__ == "__main__":
    main()

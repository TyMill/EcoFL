"""
EcoFL — Statistical Testing
============================
Wilcoxon signed-rank test comparing EcoFL vs FedAvg
across 5 seeds for each model family.

Tests:
  H0: EcoFL F1 = FedAvg F1  (two-sided)
  H0: EcoFL Energy = FedAvg Energy  (one-sided, EcoFL < FedAvg)

Correction: Holm-Bonferroni for multiple comparisons.

Usage
-----
    python3 experiments/statistical_tests.py \
        --results_dir results/

    Reads: results/seed_*/benchmark_results.json
"""

import sys, os, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

MODELS = ["LogisticRegression","RandomForest","XGBoost","MLP","IsolationForest"]
SHORT  = {"LogisticRegression":"LR","RandomForest":"RF","XGBoost":"XGB",
          "MLP":"MLP","IsolationForest":"IF"}

def load_seeds(results_dir):
    """Load per-seed results from seed_*/benchmark_results.json."""
    seed_dirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")))
    all_seeds = []
    for sd in seed_dirs:
        p = os.path.join(sd, "benchmark_results.json")
        if os.path.exists(p):
            with open(p) as f:
                all_seeds.append(json.load(f))
    return all_seeds


def extract_metric(all_seeds, model, config, metric):
    vals = []
    for seed_data in all_seeds:
        for r in seed_data:
            if r["model_name"] == model and r["configuration"] == config:
                if metric in ["f1","accuracy","roc_auc"]:
                    vals.append(r["ml_metrics"][metric])
                else:
                    vals.append(r[metric])
    return np.array(vals)


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.zeros(n)
    for i, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - i))
    # Enforce monotonicity
    for i in range(1, n):
        corrected[order[i]] = max(corrected[order[i]], corrected[order[i-1]])
    return corrected


def bootstrap_ci(x, y, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap 95% CI for the mean difference (x - y)."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        xi = rng.choice(x, len(x), replace=True)
        yi = rng.choice(y, len(y), replace=True)
        diffs.append(np.mean(xi) - np.mean(yi))
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    all_seeds = load_seeds(args.results_dir)
    if not all_seeds:
        print(f"No seed results found in {args.results_dir}/seed_*/")
        sys.exit(1)

    n_seeds = len(all_seeds)
    print(f"\n[EcoFL Statistical Tests] n_seeds = {n_seeds}\n")

    # ── F1: EcoFL vs FedAvg ──────────────────────────────────────
    print("=" * 75)
    print("  F1-SCORE: EcoFL vs FedAvg  (Wilcoxon signed-rank, two-sided)")
    print("  H0: EcoFL F1 = FedAvg F1")
    print("=" * 75)
    print(f"  {'Model':6}  {'FA mean':>9}  {'EC mean':>9}  {'Δ mean':>9}  "
          f"{'W stat':>8}  {'p-val':>8}  {'95% CI (Δ)':>20}")
    print(f"  {'-'*73}")

    p_vals_f1 = []
    rows_f1   = []
    for model in MODELS:
        fa = extract_metric(all_seeds, model, "fedavg", "f1")
        ec = extract_metric(all_seeds, model, "ecofl",  "f1")
        if len(fa) < 3 or len(ec) < 3:
            continue
        try:
            stat, p = stats.wilcoxon(ec, fa, alternative="two-sided")
        except Exception:
            stat, p = 0.0, 1.0
        lo, hi = bootstrap_ci(ec, fa)
        p_vals_f1.append(p)
        rows_f1.append((model, fa, ec, stat, p, lo, hi))

    # Holm correction
    p_corr_f1 = holm_bonferroni(np.array(p_vals_f1))

    for i, (model, fa, ec, stat, p, lo, hi) in enumerate(rows_f1):
        sig = "***" if p_corr_f1[i] < 0.001 else "**" if p_corr_f1[i] < 0.01 \
              else "*" if p_corr_f1[i] < 0.05 else "n.s."
        print(f"  {SHORT[model]:6}  {np.mean(fa):>9.4f}  {np.mean(ec):>9.4f}  "
              f"{np.mean(ec)-np.mean(fa):>+9.4f}  {stat:>8.1f}  "
              f"{p_corr_f1[i]:>8.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")

    # ── ENERGY: EcoFL vs FedAvg (one-sided, EcoFL < FedAvg) ─────
    print(f"\n{'='*75}")
    print("  ENERGY: EcoFL vs FedAvg  (Wilcoxon signed-rank, one-sided)")
    print("  H0: EcoFL energy >= FedAvg energy")
    print(f"{'='*75}")
    print(f"  {'Model':6}  {'FA (mJ)':>12}  {'EC (mJ)':>12}  "
          f"{'Saving%':>9}  {'p-val':>8}  {'Sig':>5}")
    print(f"  {'-'*60}")

    p_vals_e = []
    rows_e   = []
    for model in MODELS:
        fa = extract_metric(all_seeds, model, "fedavg", "total_energy_mj")
        ec = extract_metric(all_seeds, model, "ecofl",  "total_energy_mj")
        if len(fa) < 3:
            continue
        try:
            stat, p = stats.wilcoxon(ec, fa, alternative="less")
        except Exception:
            stat, p = 0.0, 1.0
        p_vals_e.append(p)
        rows_e.append((model, fa, ec, p))

    p_corr_e = holm_bonferroni(np.array(p_vals_e))
    for i, (model, fa, ec, p) in enumerate(rows_e):
        saving = (np.mean(fa) - np.mean(ec)) / np.mean(fa) * 100
        sig = "***" if p_corr_e[i] < 0.001 else "**" if p_corr_e[i] < 0.01 \
              else "*" if p_corr_e[i] < 0.05 else "n.s."
        print(f"  {SHORT[model]:6}  {np.mean(fa):>12.1f}  {np.mean(ec):>12.1f}  "
              f"{saving:>8.1f}%  {p_corr_e[i]:>8.4f}  {sig:>5}")

    # ── ROUNDS ───────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  ROUNDS: EcoFL vs FedAvg")
    print(f"{'='*75}")
    print(f"  {'Model':6}  {'FA rounds':>10}  {'EC rounds':>10}  {'Reduction%':>12}")
    print(f"  {'-'*44}")
    for model in MODELS:
        fa = extract_metric(all_seeds, model, "fedavg", "n_rounds")
        ec = extract_metric(all_seeds, model, "ecofl",  "n_rounds")
        red = (np.mean(fa) - np.mean(ec)) / np.mean(fa) * 100
        print(f"  {SHORT[model]:6}  {np.mean(fa):>10.1f}  {np.mean(ec):>10.1f}  {red:>11.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────
    out = {
        "n_seeds": n_seeds,
        "f1_tests": [
            {
                "model": r[0],
                "fedavg_mean": float(np.mean(r[1])),
                "ecofl_mean":  float(np.mean(r[2])),
                "delta_mean":  float(np.mean(r[2]) - np.mean(r[1])),
                "wilcoxon_stat": float(r[3]),
                "p_raw":  float(r[4]),
                "p_holm": float(p_corr_f1[i]),
                "ci_95_lo": float(r[5]),
                "ci_95_hi": float(r[6]),
            }
            for i, r in enumerate(rows_f1)
        ],
        "energy_tests": [
            {
                "model": r[0],
                "fedavg_mean_mj": float(np.mean(r[1])),
                "ecofl_mean_mj":  float(np.mean(r[2])),
                "saving_pct": float((np.mean(r[1])-np.mean(r[2]))/np.mean(r[1])*100),
                "p_holm": float(p_corr_e[i]),
            }
            for i, r in enumerate(rows_e)
        ]
    }
    os.makedirs("results", exist_ok=True)
    with open("results/statistical_tests.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[Stats] ✓ Saved → results/statistical_tests.json")
    print("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("Correction: Holm-Bonferroni (family-wise error rate control)")


if __name__ == "__main__":
    main()

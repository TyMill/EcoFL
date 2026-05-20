"""
EcoFL — Main Experiment Runner
================================
Runs the full EcoFL benchmark across all models and
configurations, saves results to JSON files.

Usage
-----
    cd ecofl_project
    python experiments/run_experiments.py

    # Optional flags:
    python experiments/run_experiments.py --quick     # 3 models, 10 rounds
    python experiments/run_experiments.py --profile jetson_nano
    python experiments/run_experiments.py --seeds 3

Hardware emulation
------------------
To emulate Raspberry Pi 4 constraints via cgroups:
    systemd-run --scope -p CPUQuota=25% -p MemoryMax=1G \
        python experiments/run_experiments.py

(CPUQuota=25% ≈ 1 core on a 4-core host)
"""

import sys
import os
import argparse
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecofl.benchmark.pipeline import BenchmarkPipeline


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="EcoFL Benchmark Runner")
    parser.add_argument(
        "--profile",
        choices=["raspberry_pi4", "jetson_nano", "desktop"],
        default="raspberry_pi4",
        help="Hardware profile for energy estimation (default: raspberry_pi4)",
    )
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Maximum FL rounds (default: 20)",
    )
    parser.add_argument(
        "--clients", type=int, default=10,
        help="Number of federated clients (default: 10)",
    )
    parser.add_argument(
        "--samples", type=int, default=50_000,
        help="Dataset size (default: 50000)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds for averaging (default: 5, seeds 42–46)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3 models, 5 rounds, 1 seed, 10k samples",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["LogisticRegression", "RandomForest", "XGBoost", "MLP", "IsolationForest"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results (default: results/)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_seed(
    seed:     int,
    args,
    models:   list,
    out_dir:  str,
) -> list:
    """Run one full benchmark with a given random seed."""
    print(f"\n{'#'*70}")
    print(f"  EcoFL Benchmark — Seed {seed}")
    print(f"  Profile : {args.profile}")
    print(f"  Clients : {args.clients}  Rounds : {args.rounds}  Samples : {args.samples:,}")
    print(f"  Models  : {models}")
    print(f"{'#'*70}")

    pipeline = BenchmarkPipeline(
        n_clients=args.clients,
        n_rounds=args.rounds,
        n_samples=args.samples,
        hardware_profile=args.profile,
        alpha=0.5,
        random_state=seed,
        verbose=True,
    )

    results = pipeline.run_all(model_names=models)

    # Save per-seed results
    seed_out = os.path.join(out_dir, f"seed_{seed}")
    pipeline.save_results(os.path.join(seed_out, "benchmark_results.json"))
    pipeline.save_round_history(os.path.join(seed_out, "round_history.json"))

    return results


def aggregate_seeds(all_seed_results: list, seeds: list) -> list:
    """
    Average ML metrics and system metrics across seeds.
    Returns list of dicts with mean ± std for each metric.
    """
    import numpy as np
    from collections import defaultdict

    # Group by (model_name, configuration)
    groups = defaultdict(list)
    for seed_results in all_seed_results:
        for r in seed_results:
            key = (r["model_name"], r["configuration"])
            groups[key].append(r)

    aggregated = []
    for (model_name, configuration), runs in groups.items():
        def mean_std(vals):
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        entry = {
            "model_name":    model_name,
            "configuration": configuration,
            "n_seeds":       len(runs),
            "ml_metrics": {
                metric: mean_std([r["ml_metrics"][metric] for r in runs])
                for metric in ["accuracy", "f1", "roc_auc", "precision", "recall"]
            },
            "total_energy_mj":           mean_std([r["total_energy_mj"] for r in runs]),
            "inference_latency_mean_ms": mean_std([r["inference_latency_mean_ms"] for r in runs]),
            "peak_ram_mb":               mean_std([r["peak_ram_mb"] for r in runs]),
            "communication_kb":          mean_std([r["communication_kb"] for r in runs]),
            "n_rounds":                  mean_std([r["n_rounds"] for r in runs]),
            "training_time_s":           mean_std([r["training_time_s"] for r in runs]),
            "hardware_profile":          runs[0]["hardware_profile"],
        }
        aggregated.append(entry)

    return aggregated


def print_final_table(aggregated: list):
    """Print a summary table of final results."""
    print("\n" + "="*90)
    print("  FINAL RESULTS (mean ± std across seeds)")
    print("="*90)
    print(f"  {'Model':20s} {'Config':12s} {'Acc':>8s} {'F1':>8s} "
          f"{'AUC':>8s} {'Energy(mJ)':>12s} {'Rounds':>8s}")
    print("-"*90)

    for r in sorted(aggregated, key=lambda x: (x["model_name"], x["configuration"])):
        ml  = r["ml_metrics"]
        acc = f"{ml['accuracy']['mean']:.3f}±{ml['accuracy']['std']:.3f}"
        f1  = f"{ml['f1']['mean']:.3f}±{ml['f1']['std']:.3f}"
        auc = f"{ml['roc_auc']['mean']:.3f}±{ml['roc_auc']['std']:.3f}"
        eng = f"{r['total_energy_mj']['mean']:.1f}±{r['total_energy_mj']['std']:.1f}"
        rnd = f"{r['n_rounds']['mean']:.1f}±{r['n_rounds']['std']:.1f}"
        print(f"  {r['model_name']:20s} {r['configuration']:12s} "
              f"{acc:>8s} {f1:>8s} {auc:>8s} {eng:>12s} {rnd:>8s}")

    print("="*90)


def main():
    args   = parse_args()
    t_wall = time.time()

    # Quick mode overrides
    if args.quick:
        args.models  = ["LogisticRegression", "XGBoost", "MLP"]
        args.rounds  = 5
        args.seeds   = 1
        args.samples = 10_000
        print("⚡ Quick mode: 3 models, 5 rounds, 1 seed, 10k samples")

    os.makedirs(args.output, exist_ok=True)
    seeds = list(range(42, 42 + args.seeds))

    all_seed_results = []
    for seed in seeds:
        results = run_seed(seed, args, args.models, args.output)
        all_seed_results.append(results)

    # Aggregate across seeds
    aggregated = aggregate_seeds(all_seed_results, seeds)

    # Save aggregated results
    agg_path = os.path.join(args.output, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=float)
    print(f"\n[EcoFL] Aggregated results → {agg_path}")

    print_final_table(aggregated)

    elapsed = time.time() - t_wall
    print(f"\n[EcoFL] Total wall time: {elapsed/60:.1f} min")
    print("[EcoFL] ✓ Done. Run experiments/visualize_results.py to generate figures.")


if __name__ == "__main__":
    main()

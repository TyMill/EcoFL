"""
EcoFL — Ablation Study
======================
Evaluates sensitivity of EcoFL scheduler to:
  - convergence_tolerance (δ): [0.001, 0.005, 0.010, 0.020]
  - patience (p):              [2, 3, 5, 7]
  - cpu_threshold (θ_CPU):    [0.70, 0.80, 0.90, 1.00]

For each configuration: runs XGBoost EcoFL (fast model)
across 3 seeds, reports F1, energy, rounds.

Usage
-----
    cd ecofl_project/
    python3 experiments/ablation_study.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ecofl.data.generator import make_dataset
from ecofl.federated.client import EcoFLClient
from ecofl.federated.server import FederatedServer
from ecofl.federated.scheduler import SchedulerConfig
from sklearn.metrics import f1_score, accuracy_score

SEEDS        = [42, 43, 44]
MODEL        = "XGBoost"   # fast model for ablation
HW_PROFILE   = "raspberry_pi4"
N_SAMPLES    = 20_000
N_CLIENTS    = 10
N_ROUNDS_MAX = 20

def run_one(delta, patience, cpu_thresh, seed):
    data = make_dataset(n_samples=N_SAMPLES, n_clients=N_CLIENTS,
                        random_state=seed)
    clients = [
        EcoFLClient(i, data["client_partitions"][i][0],
                    data["client_partitions"][i][1],
                    MODEL, HW_PROFILE, energy_budget_mj=500.0)
        for i in range(N_CLIENTS)
    ]
    cfg = SchedulerConfig(
        convergence_tolerance=delta,
        patience=patience,
        cpu_threshold=cpu_thresh,
        max_rounds=N_ROUNDS_MAX,
    )
    server = FederatedServer(MODEL, mode="ecofl", scheduler_config=cfg)
    res = server.run(clients, data["X_test"], data["y_test"],
                     n_rounds=N_ROUNDS_MAX, verbose=False)
    rounds  = len(res["rounds"])
    energy  = res["total_training_energy_mj"]
    final   = res["final_metrics"]
    f1      = final.get("f1", 0.0)
    acc     = final.get("accuracy", 0.0)
    return rounds, energy, f1, acc


def ablation(param_name, param_vals, fixed):
    print(f"\n{'='*70}")
    print(f"  Ablation: {param_name}")
    print(f"  Fixed: {fixed}")
    print(f"{'='*70}")
    print(f"  {'Value':>8}  {'Rounds':>8}  {'Energy(mJ)':>12}  {'F1':>8}  {'Acc':>8}")
    print(f"  {'-'*56}")

    results = []
    for val in param_vals:
        kw = dict(fixed)
        kw[param_name] = val
        seed_results = []
        for seed in SEEDS:
            r = run_one(kw["delta"], kw["patience"], kw["cpu_thresh"], seed)
            seed_results.append(r)
        rounds  = np.mean([x[0] for x in seed_results])
        energy  = np.mean([x[1] for x in seed_results])
        f1      = np.mean([x[2] for x in seed_results])
        f1_std  = np.std ([x[2] for x in seed_results])
        acc     = np.mean([x[3] for x in seed_results])
        print(f"  {str(val):>8}  {rounds:>8.1f}  {energy:>12.1f}  "
              f"{f1:>6.3f}±{f1_std:.3f}  {acc:>6.3f}")
        results.append({
            param_name: val, "rounds": rounds, "energy_mj": energy,
            "f1_mean": f1, "f1_std": f1_std, "accuracy": acc
        })
    return results


def main():
    fixed_base = {"delta": 0.005, "patience": 3, "cpu_thresh": 0.80}
    all_results = {}

    # 1. δ ablation
    all_results["delta"] = ablation(
        "delta",
        [0.001, 0.005, 0.010, 0.020],
        fixed_base
    )

    # 2. patience ablation
    all_results["patience"] = ablation(
        "patience",
        [2, 3, 5, 7],
        fixed_base
    )

    # 3. cpu_threshold ablation
    all_results["cpu_thresh"] = ablation(
        "cpu_thresh",
        [0.70, 0.80, 0.90, 1.00],
        fixed_base
    )

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\n[Ablation] ✓ Saved → results/ablation_results.json")


if __name__ == "__main__":
    main()

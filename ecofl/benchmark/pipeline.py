"""
EcoFL Benchmark Pipeline
=========================
Orchestrates the complete experimental evaluation:

  For each model × {Centralized, FedAvg, EcoFL}:
    - Train with full system resource monitoring
    - Measure ML performance metrics
    - Measure system-level metrics (CPU, RAM, Energy, Latency)
    - Log structured results

Output: list of result dicts → JSON for downstream analysis
        and visualisation (experiments/visualize_results.py).
"""

import time
import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional

from ecofl.data.generator import make_dataset
from ecofl.energy.monitor import SystemMonitor
from ecofl.models.lightweight import (
    MODEL_CONFIGS, create_model, get_model_size_kb,
)
from ecofl.federated.client import EcoFLClient
from ecofl.federated.server import FederatedServer
from ecofl.federated.scheduler import SchedulerConfig


class BenchmarkPipeline:
    """
    Full EcoFL benchmarking pipeline.

    Parameters
    ----------
    n_clients       : int   — number of federated edge nodes
    n_rounds        : int   — max FL rounds (hard limit)
    n_samples       : int   — total dataset size
    hardware_profile: str   — 'raspberry_pi4' | 'jetson_nano' | 'desktop'
    alpha           : float — Dirichlet concentration (non-IID degree)
    random_state    : int
    verbose         : bool
    """

    def __init__(
        self,
        n_clients:        int   = 10,
        n_rounds:         int   = 20,
        n_samples:        int   = 50_000,
        hardware_profile: str   = "raspberry_pi4",
        alpha:            float = 0.5,
        random_state:     int   = 42,
        verbose:          bool  = True,
    ):
        self.n_clients        = n_clients
        self.n_rounds         = n_rounds
        self.n_samples        = n_samples
        self.hw_profile       = hardware_profile
        self.alpha            = alpha
        self.random_state     = random_state
        self.verbose          = verbose
        self.results:         List[Dict] = []

        self.tdp_watts = EcoFLClient.HARDWARE_PROFILES[hardware_profile]["tdp_watts"]

    # ────────────────────────────────────────────────────────────
    # Data preparation
    # ────────────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        """Generate, split, scale, and partition dataset."""
        if self.verbose:
            print("\n[EcoFL] Generating IoT telemetry dataset …")

        data = make_dataset(
            n_samples=self.n_samples,
            n_clients=self.n_clients,
            alpha=self.alpha,
            random_state=self.random_state,
        )

        self.X_train_all       = data["X_train_all"]
        self.y_train_all       = data["y_train_all"]
        self.X_test            = data["X_test"]
        self.y_test            = data["y_test"]
        self.client_partitions = data["client_partitions"]
        self.meta              = data["meta"]

        if self.verbose:
            m = self.meta
            print(f"  Samples : {m['n_samples']:,}  "
                  f"Features: {m['n_features']}  "
                  f"Anomaly rate: {m['anomaly_rate']:.1%}")
            print(f"  Train: {m['n_train']:,}  Test: {m['n_test']:,}")
            print(f"  Clients: {m['n_clients']}  α={m['dirichlet_alpha']}  "
                  f"Hardware: {self.hw_profile} (TDP={self.tdp_watts} W)")

    # ────────────────────────────────────────────────────────────
    # Centralized baseline
    # ────────────────────────────────────────────────────────────

    def run_centralized(self, model_name: str) -> Dict:
        """Train model on full pooled training set."""
        if self.verbose:
            print(f"\n  ── [Centralized] {model_name}")

        model   = create_model(model_name)
        monitor = SystemMonitor(tdp_watts=self.tdp_watts)

        # ── Training ─────────────────────────────────────────────
        monitor.start()
        t0 = time.perf_counter()

        if model_name == "IsolationForest":
            X_normal = self.X_train_all[self.y_train_all == 0]
            model.fit(X_normal if len(X_normal) > 5 else self.X_train_all)
        else:
            model.fit(self.X_train_all, self.y_train_all)

        t_train = time.perf_counter() - t0
        sys_rep = monitor.stop()

        # For very fast operations the background sampler may miss the window;
        # fall back to wall-clock energy estimate directly.
        energy_mj = sys_rep.estimated_energy_mj
        if energy_mj < 1e-6:
            energy_mj = self.tdp_watts * 0.5 * t_train * 1000.0  # assume 50% CPU

        # ── Inference latency ─────────────────────────────────────
        latencies = []
        for _ in range(10):
            t = time.perf_counter()
            if model_name == "IsolationForest":
                raw  = model.predict(self.X_test)
                preds = (raw == -1).astype(int)
            else:
                preds = model.predict(self.X_test)
            latencies.append((time.perf_counter() - t) * 1000.0)

        # ── Probabilities ─────────────────────────────────────────
        proba = self._get_proba(model, model_name, self.X_test)

        ml  = self._ml_metrics(self.y_test, preds, proba)
        kbs = get_model_size_kb(model, model_name)

        result = {
            "configuration":             "centralized",
            "model_name":                model_name,
            "ml_metrics":                ml,
            "training_time_s":           t_train,
            "inference_latency_mean_ms": float(np.mean(latencies)),
            "inference_latency_p95_ms":  float(np.percentile(latencies, 95)),
            "mean_cpu_percent":          sys_rep.mean_cpu_percent,
            "peak_ram_mb":               sys_rep.peak_ram_mb,
            "total_energy_mj":           energy_mj,
            "energy_per_round_mj":       energy_mj,
            "communication_kb":          0.0,
            "n_rounds":                  1,
            "model_size_kb":             kbs,
            "hardware_profile":          self.hw_profile,
        }

        if self.verbose:
            print(f"     Acc={ml['accuracy']:.4f}  F1={ml['f1']:.4f}  "
                  f"Energy={sys_rep.estimated_energy_mj:.2f} mJ  "
                  f"Latency={np.mean(latencies):.2f} ms")
        return result

    # ────────────────────────────────────────────────────────────
    # Federated (FedAvg or EcoFL)
    # ────────────────────────────────────────────────────────────

    def run_federated(
        self,
        model_name: str,
        mode:       str = "fedavg",
    ) -> Dict:
        """Run federated training in either FedAvg or EcoFL mode."""
        cfg_name = "ecofl" if mode == "ecofl" else "fedavg"
        if self.verbose:
            print(f"\n  ── [{cfg_name.upper():6s}] {model_name}")

        # ── Create fresh clients ──────────────────────────────────
        clients = [
            EcoFLClient(
                client_id=i,
                X_local=self.client_partitions[i][0],
                y_local=self.client_partitions[i][1],
                model_name=model_name,
                hardware_profile=self.hw_profile,
                energy_budget_mj=500.0 if mode == "ecofl" else None,
            )
            for i in range(self.n_clients)
        ]

        # ── Run FL loop ───────────────────────────────────────────
        server  = FederatedServer(model_name=model_name, mode=mode)
        fl_res  = server.run(
            clients=clients,
            X_test=self.X_test,
            y_test=self.y_test,
            n_rounds=self.n_rounds,
            verbose=self.verbose,
        )

        # ── Final inference latency ───────────────────────────────
        fitted   = [c for c in clients if c._is_fitted]
        latencies = []
        if fitted:
            sample = self.X_test[:200]
            for _ in range(5):
                t = time.perf_counter()
                fitted[0].inference(sample, n_runs=1)
                latencies.append((time.perf_counter() - t) * 1000.0)

        final      = fl_res.get("final_metrics", {})
        rounds_run = len(fl_res["rounds"])

        # CPU/RAM summaries from client histories.
        # Earlier versions derived CPU from energy/energy, which was only a placeholder.
        cpu_values = [
            h["system_metrics"].get("mean_cpu_percent", 0.0)
            for c in clients for h in c.training_history
            if "system_metrics" in h
        ]
        peak_rams = [
            h["system_metrics"].get("peak_ram_mb", 0.0)
            for c in clients for h in c.training_history
            if "system_metrics" in h
        ]

        result = {
            "configuration":             cfg_name,
            "model_name":                model_name,
            "ml_metrics":                {
                k: final.get(k, 0.0)
                for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]
            },
            "training_time_s":           fl_res["total_time_s"],
            "inference_latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "inference_latency_p95_ms":  float(np.percentile(latencies, 95)) if latencies else 0.0,
            "mean_cpu_percent":          float(np.mean(cpu_values)) if cpu_values else 0.0,
            "peak_ram_mb":               float(max(peak_rams)) if peak_rams else 0.0,
            "total_energy_mj":           fl_res["total_training_energy_mj"],
            "energy_per_round_mj":       (
                fl_res["total_training_energy_mj"] / rounds_run
                if rounds_run > 0 else 0.0
            ),
            "communication_kb":          fl_res["total_communication_kb"],
            "comm_per_round_kb":         (
                fl_res["total_communication_kb"] / rounds_run
                if rounds_run > 0 else 0.0
            ),
            "n_rounds":                  rounds_run,
            "round_history":             fl_res["rounds"],
            "hardware_profile":          self.hw_profile,
        }

        if mode == "ecofl":
            result["scheduler_stats"] = fl_res.get("scheduler_stats", {})

        if self.verbose:
            ml = result["ml_metrics"]
            print(f"     Acc={ml['accuracy']:.4f}  F1={ml['f1']:.4f}  "
                  f"Energy={result['total_energy_mj']:.2f} mJ  "
                  f"Rounds={rounds_run}")

        return result

    # ────────────────────────────────────────────────────────────
    # Full benchmark runner
    # ────────────────────────────────────────────────────────────

    def run_all(
        self,
        model_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Run all models × all configurations."""
        if model_names is None:
            model_names = list(MODEL_CONFIGS.keys())

        self.prepare_data()

        for model_name in model_names:
            if self.verbose:
                print(f"\n{'='*65}")
                print(f"  Model: {model_name}")
                print(f"{'='*65}")

            r_c  = self.run_centralized(model_name)
            r_fa = self.run_federated(model_name, mode="fedavg")
            r_ec = self.run_federated(model_name, mode="ecofl")

            self.results.extend([r_c, r_fa, r_ec])

            if self.verbose:
                print(f"\n  Summary — {model_name}:")
                for r in [r_c, r_fa, r_ec]:
                    ml = r["ml_metrics"]
                    print(
                        f"    {r['configuration']:12s}  "
                        f"Acc={ml['accuracy']:.3f}  "
                        f"F1={ml['f1']:.3f}  "
                        f"Energy={r['total_energy_mj']:.1f} mJ  "
                        f"Rounds={r['n_rounds']}"
                    )

        return self.results

    # ────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────

    def save_results(self, path: str = "results/benchmark_results.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Strip round_history for compact summary file
        clean = [
            {k: v for k, v in r.items() if k != "round_history"}
            for r in self.results
        ]
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=float)
        if self.verbose:
            print(f"\n[EcoFL] Summary saved → {path}")

    def save_round_history(
        self, path: str = "results/round_history.json"
    ) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        history = {
            f"{r['model_name']}_{r['configuration']}": r.get("round_history", [])
            for r in self.results
            if "round_history" in r
        }
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=float)
        if self.verbose:
            print(f"[EcoFL] Round history saved → {path}")

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _ml_metrics(y_true, y_pred, y_prob=None) -> Dict:
        m = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "f1":        float(f1_score(y_true, y_pred,
                                        average="macro", zero_division=0)),
            "precision": float(precision_score(y_true, y_pred,
                                               average="macro", zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred,
                                            average="macro", zero_division=0)),
            "roc_auc":   0.0,
        }
        if y_prob is not None:
            try:
                m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                pass
        return m

    @staticmethod
    def _get_proba(model, model_name: str, X: np.ndarray):
        try:
            if model_name == "IsolationForest":
                scores = model.score_samples(X)
                return -(scores - scores.min()) / (
                    scores.max() - scores.min() + 1e-8
                )
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None

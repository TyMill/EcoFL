"""
EcoFL Federated Learning Client
=================================
Simulates a single IoT edge node participating in federated
learning. Handles local training, inference, resource
monitoring, and exposes resource status to the scheduler.

Hardware profiles
-----------------
raspberry_pi4 : TDP=6.4W, 1 core, 1 GB RAM
jetson_nano   : TDP=10.0W, 2 cores, 4 GB RAM
desktop       : TDP=65.0W, 8 cores, 32 GB RAM (for testing)
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Any, Dict, List, Optional, Tuple

from ecofl.energy.monitor import SystemMonitor
from ecofl.models.lightweight import (
    MODEL_CONFIGS,
    create_model,
    get_model_parameters,
    set_model_parameters,
    get_model_size_kb,
)


class EcoFLClient:
    """
    Simulated edge IoT node for federated learning.

    Parameters
    ----------
    client_id : int
    X_local, y_local : local dataset partition
    model_name : str — one of MODEL_CONFIGS keys
    hardware_profile : 'raspberry_pi4' | 'jetson_nano' | 'desktop'
    energy_budget_mj : float | None
        Total energy budget in millijoules.
        None = unlimited (FedAvg baseline).
    """

    HARDWARE_PROFILES = {
        "raspberry_pi4": {"tdp_watts": 6.4,  "cpu_cores": 1, "ram_gb": 1},
        "jetson_nano":   {"tdp_watts": 10.0, "cpu_cores": 2, "ram_gb": 4},
        "desktop":       {"tdp_watts": 65.0, "cpu_cores": 8, "ram_gb": 32},
    }

    def __init__(
        self,
        client_id:        int,
        X_local:          np.ndarray,
        y_local:          np.ndarray,
        model_name:       str,
        hardware_profile: str   = "raspberry_pi4",
        energy_budget_mj: Optional[float] = None,
    ):
        self.client_id        = client_id
        self.model_name       = model_name
        self.hw               = self.HARDWARE_PROFILES[hardware_profile]
        self.energy_budget_mj = energy_budget_mj
        self.energy_spent_mj  = 0.0
        self._is_fitted       = False

        # ── Local train / val split ──────────────────────────────
        classes = np.unique(y_local)
        if len(classes) > 1 and len(y_local) >= 10:
            try:
                (self.X_train, self.X_val,
                 self.y_train, self.y_val) = train_test_split(
                    X_local, y_local,
                    test_size=0.2,
                    stratify=y_local,
                    random_state=42,
                )
            except ValueError:
                self.X_train = self.X_val = X_local
                self.y_train = self.y_val = y_local
        else:
            self.X_train = self.X_val = X_local
            self.y_train = self.y_val = y_local

        # ── Model ────────────────────────────────────────────────
        self.model: Any = create_model(model_name)

        # ── Monitor ──────────────────────────────────────────────
        self._monitor = SystemMonitor(
            sampling_interval=0.3,
            tdp_watts=self.hw["tdp_watts"],
        )

        # ── History ──────────────────────────────────────────────
        self.training_history: List[Dict] = []

    # ────────────────────────────────────────────────────────────
    # Properties
    # ────────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self.X_train)

    @property
    def federated_strategy(self) -> str:
        return MODEL_CONFIGS[self.model_name]["federated_strategy"]

    # ────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────

    def train(self, global_params: Optional[Dict] = None) -> Dict:
        """
        Local training with full resource profiling.

        If global_params is provided and strategy is parameter_avg,
        the global weights are injected before local fine-tuning.

        Returns
        -------
        dict with model_params, n_samples, local_accuracy,
        training_time_s, model_size_kb, system_metrics.
        """
        # Inject global parameters (warm-start for LR / MLP)
        if (global_params is not None
                and self.federated_strategy == "parameter_avg"
                and self._is_fitted):
            set_model_parameters(self.model, global_params, self.model_name)

        # ── Train ────────────────────────────────────────────────
        self._monitor.start()
        t0 = time.perf_counter()

        if self.model_name == "IsolationForest":
            X_normal = self.X_train[self.y_train == 0]
            fit_data  = X_normal if len(X_normal) > 5 else self.X_train
            self.model.fit(fit_data)
        else:
            try:
                self.model.fit(self.X_train, self.y_train)
            except ValueError:
                # Degenerate partition: recreate model, disable early_stopping
                self.model = create_model(self.model_name)
                if hasattr(self.model, "early_stopping"):
                    self.model.early_stopping = False
                try:
                    self.model.fit(self.X_train, self.y_train)
                except Exception:
                    pass

        t_train   = time.perf_counter() - t0
        sys_rep   = self._monitor.stop()
        # Check if model actually fitted
        params = get_model_parameters(self.model, self.model_name)
        if params is None:
            # Model failed to fit (degenerate partition) — skip
            return None

        self._is_fitted = True

        # Track cumulative energy
        self.energy_spent_mj += sys_rep.estimated_energy_mj

        # ── Metrics ──────────────────────────────────────────────
        local_acc    = self._local_accuracy()
        model_size   = get_model_size_kb(self.model, self.model_name)

        result = {
            "client_id":       self.client_id,
            "model_params":    params,
            "n_samples":       self.n_samples,
            "local_accuracy":  local_acc,
            "training_time_s": t_train,
            "model_size_kb":   model_size,
            "system_metrics":  sys_rep.to_dict(),
        }
        self.training_history.append(result)
        return result

    # ────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────

    def inference(self, X: np.ndarray, n_runs: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Run inference with latency profiling.

        Returns
        -------
        predictions : np.ndarray
        metrics     : dict  (latency_mean_ms, latency_p95_ms, system_metrics)
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Client {self.client_id}: model not fitted yet."
            )

        latencies   = []
        predictions = None

        for _ in range(n_runs):
            t = time.perf_counter()
            if self.model_name == "IsolationForest":
                raw = self.model.predict(X)
                predictions = (raw == -1).astype(int)
            else:
                predictions = self.model.predict(X)
            latencies.append((time.perf_counter() - t) * 1000.0)

        # System metrics sampled separately (don't inflate latency)
        self._monitor.start()
        if self.model_name == "IsolationForest":
            raw = self.model.predict(X)
        else:
            _ = self.model.predict(X)
        sys_rep = self._monitor.stop()

        inf_metrics = {
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_p95_ms":  float(np.percentile(latencies, 95)),
            "system_metrics":  sys_rep.to_dict(),
        }
        return predictions, inf_metrics

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return class probability estimates if available."""
        if not self._is_fitted:
            return None
        try:
            if self.model_name == "IsolationForest":
                scores = self.model.score_samples(X)
                # Invert and normalise → higher = more anomalous
                proba  = -(scores - scores.min()) / (
                    scores.max() - scores.min() + 1e-8
                )
                return np.column_stack([1 - proba, proba])
            return self.model.predict_proba(X)
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────
    # Resource status (called by scheduler)
    # ────────────────────────────────────────────────────────────

    def get_resource_status(self) -> Dict:
        """
        Current resource snapshot for scheduler decisions.
        """
        status = self._monitor.get_current_status()
        remaining = (
            self.energy_budget_mj - self.energy_spent_mj
            if self.energy_budget_mj is not None
            else float("inf")
        )
        status["energy_budget_remaining_mj"] = remaining
        status["energy_spent_mj"]             = self.energy_spent_mj
        return status

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    def _local_accuracy(self) -> float:
        try:
            if self.model_name == "IsolationForest":
                raw  = self.model.predict(self.X_val)
                pred = (raw == -1).astype(int)
            else:
                pred = self.model.predict(self.X_val)
            return float(accuracy_score(self.y_val, pred))
        except Exception:
            return 0.0

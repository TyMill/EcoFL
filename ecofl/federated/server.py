"""
EcoFL Federated Learning Server
=================================
Central coordinator for both FedAvg and EcoFL training modes.

Responsibilities
----------------
- Distribute global model parameters to participating clients
- Trigger local training
- Aggregate updated parameters (FedAvg or prediction ensemble)
- Evaluate global model on held-out test set after each round
- Delegate round/client scheduling to EnergyAwareScheduler (EcoFL)
- Log per-round system metrics
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)
from typing import Dict, List, Optional, TYPE_CHECKING

from ecofl.models.lightweight import (
    MODEL_CONFIGS,
    aggregate_parameters,
    set_model_parameters,
)
from ecofl.federated.scheduler import EnergyAwareScheduler, SchedulerConfig

if TYPE_CHECKING:
    from ecofl.federated.client import EcoFLClient


class FederatedServer:
    """
    FL server supporting FedAvg and EcoFL operation modes.

    Parameters
    ----------
    model_name : str
    mode       : 'fedavg' | 'ecofl'
    scheduler_config : SchedulerConfig | None
        Only used when mode='ecofl'.
    """

    def __init__(
        self,
        model_name:       str,
        mode:             str = "fedavg",
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        self.model_name        = model_name
        self.mode              = mode
        self.fed_strategy      = MODEL_CONFIGS[model_name]["federated_strategy"]
        self._round_history:   List[Dict] = []

        self.scheduler: Optional[EnergyAwareScheduler] = (
            EnergyAwareScheduler(scheduler_config or SchedulerConfig())
            if mode == "ecofl" else None
        )

    # ────────────────────────────────────────────────────────────
    # Main training loop
    # ────────────────────────────────────────────────────────────

    def run(
        self,
        clients:  List["EcoFLClient"],
        X_test:   np.ndarray,
        y_test:   np.ndarray,
        n_rounds: int  = 20,
        verbose:  bool = True,
    ) -> Dict:
        """
        Execute the federated training loop.

        Returns
        -------
        dict with keys:
          mode, model_name, rounds (list of per-round dicts),
          total_communication_kb, total_training_energy_mj,
          total_time_s, final_metrics, scheduler_stats (EcoFL only)
        """
        results = {
            "mode":                     self.mode,
            "model_name":               self.model_name,
            "rounds":                   [],
            "total_communication_kb":   0.0,
            "total_training_energy_mj": 0.0,
            "n_clients":                len(clients),
        }

        t_start        = time.perf_counter()
        global_params  = None   # updated after first round (param_avg)

        for rnd in range(1, n_rounds + 1):

            # ── Client selection ─────────────────────────────────
            if self.mode == "ecofl" and self.scheduler:
                participating, excluded_ids = self.scheduler.select_clients(
                    clients, rnd
                )
            else:
                participating = clients
                excluded_ids  = []

            if verbose:
                tag = f"[{self.mode.upper():6s}]"
                print(
                    f"  {tag} Round {rnd:2d}/{n_rounds} | "
                    f"Clients {len(participating)}/{len(clients)} | "
                    f"Excl: {excluded_ids}"
                )

            # ── Local training ───────────────────────────────────
            round_params:  List[Dict] = []
            round_weights: List[float] = []
            round_energy:  float      = 0.0
            round_comm:    float      = 0.0

            for client in participating:
                tr = client.train(global_params=global_params)
                if tr is None:
                    continue
                round_params.append(tr["model_params"])
                round_weights.append(float(tr["n_samples"]))
                round_energy += tr["system_metrics"]["estimated_energy_mj"]
                round_comm   += tr["model_size_kb"]

            if not round_params:
                continue  # no clients trained this round

            # ── Aggregation ──────────────────────────────────────
            if self.fed_strategy == "parameter_avg":
                global_params = aggregate_parameters(
                    round_params, round_weights, self.model_name
                )
                # Broadcast to all clients (including non-participating)
                for c in clients:
                    if c._is_fitted:
                        set_model_parameters(
                            c.model, global_params, self.model_name
                        )

            # ── Evaluation ───────────────────────────────────────
            metrics = self._evaluate(participating, X_test, y_test)

            round_result = {
                "round":          rnd,
                "n_participating": len(participating),
                "n_excluded":      len(excluded_ids),
                "excluded_ids":    excluded_ids,
                "energy_mj":       round_energy,
                "comm_kb":         round_comm,
                **metrics,
            }
            results["rounds"].append(round_result)
            results["total_training_energy_mj"] += round_energy
            results["total_communication_kb"]   += round_comm
            self._round_history.append(round_result)

            if verbose:
                print(
                    f"           ↳ Acc={metrics['accuracy']:.4f}  "
                    f"F1={metrics['f1']:.4f}  "
                    f"E={round_energy:.1f} mJ  "
                    f"Comm={round_comm:.1f} KB"
                )

            # ── EcoFL early termination ──────────────────────────
            if (self.mode == "ecofl"
                    and self.scheduler
                    and self.scheduler.should_terminate(
                        metrics["accuracy"], rnd
                    )):
                if verbose:
                    print(f"  → EcoFL: early termination at round {rnd}.")
                break

        results["total_time_s"] = time.perf_counter() - t_start
        results["final_metrics"] = (
            results["rounds"][-1] if results["rounds"] else {}
        )

        if self.mode == "ecofl" and self.scheduler:
            results["scheduler_stats"] = self.scheduler.get_stats()

        return results

    # ────────────────────────────────────────────────────────────
    # Evaluation helpers
    # ────────────────────────────────────────────────────────────

    def _evaluate(
        self,
        clients: List["EcoFLClient"],
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> Dict:
        """Evaluate current global model on test set."""
        fitted = [c for c in clients if c._is_fitted]
        if not fitted:
            return self._empty_metrics()

        try:
            if self.fed_strategy == "parameter_avg":
                # All clients share the same global parameters
                preds, _ = fitted[0].inference(X_test, n_runs=1)
                proba     = self._get_proba(fitted[0], X_test)

            else:
                # Weighted prediction ensemble
                preds, proba = self._ensemble_predict(fitted, X_test)

            return self._compute_metrics(y_test, preds, proba)

        except Exception as e:
            print(f"    ⚠ Evaluation error: {e}")
            return self._empty_metrics()

    def _ensemble_predict(
        self,
        clients: List["EcoFLClient"],
        X:       np.ndarray,
    ):
        """Weighted majority-vote ensemble for tree models."""
        total_weight = sum(c.n_samples for c in clients)
        vote_matrix  = np.zeros((len(X), 2), dtype=float)

        for client in clients:
            pred, _ = client.inference(X, n_runs=1)
            w        = client.n_samples / total_weight
            for i, p in enumerate(pred):
                vote_matrix[i, int(p)] += w

        preds = np.argmax(vote_matrix, axis=1)
        proba = vote_matrix[:, 1]
        return preds, proba

    @staticmethod
    def _get_proba(client: "EcoFLClient", X: np.ndarray):
        proba_2d = client.predict_proba(X)
        if proba_2d is not None:
            return proba_2d[:, 1]
        return None

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_prob=None) -> Dict:
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
    def _empty_metrics() -> Dict:
        return {
            "accuracy": 0.0, "f1": 0.0,
            "precision": 0.0, "recall": 0.0, "roc_auc": 0.0,
        }

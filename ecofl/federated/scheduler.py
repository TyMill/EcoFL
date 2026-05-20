"""
EcoFL Energy-Aware Communication Scheduler
==========================================
Core novelty of EcoFL.

Implements two complementary mechanisms to reduce total FL
energy consumption without commensurate accuracy loss:

1. Dynamic client selection
   At each round boundary, each client is evaluated against
   three resource thresholds (CPU, RAM, energy budget).
   Clients failing any threshold are excluded for that round.

2. Adaptive round termination
   Federation is terminated early when global model accuracy
   improvement stagnates below tolerance δ for p consecutive
   rounds, avoiding wasted computation and communication.

Together these constitute the EcoFL energy-aware strategy,
tested against the standard FedAvg baseline which runs all
clients for a fixed number of rounds unconditionally.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ecofl.federated.client import EcoFLClient


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class SchedulerConfig:
    """
    EcoFL scheduler hyper-parameters (Table X in paper).

    Attributes
    ----------
    cpu_threshold : float
        Maximum allowed CPU utilisation fraction [0, 1].
        Clients exceeding this are excluded.
    ram_threshold : float
        Maximum allowed RAM utilisation fraction [0, 1].
    min_energy_budget_mj : float
        Minimum remaining energy (mJ) required to participate.
    convergence_tolerance : float (δ)
        Minimum per-round accuracy improvement to count as
        "meaningful progress".
    patience : int (p)
        Number of consecutive rounds without improvement
        before early termination.
    min_clients_per_round : int (K_min)
        Minimum clients required to run an aggregation round.
    max_rounds : int
        Hard upper limit on number of rounds.
    """
    cpu_threshold:         float = 0.80
    ram_threshold:         float = 0.85
    min_energy_budget_mj:  float = 10.0
    convergence_tolerance: float = 0.005
    patience:              int   = 3
    min_clients_per_round: int   = 2
    max_rounds:            int   = 20


# ─────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────

class EnergyAwareScheduler:
    """
    Energy-aware communication scheduler for EcoFL.

    Usage
    -----
    scheduler = EnergyAwareScheduler(SchedulerConfig())

    for round_num in range(1, max_rounds + 1):
        eligible, excluded = scheduler.select_clients(clients, round_num)
        if not eligible:
            break
        # ... run round with eligible clients ...
        acc = evaluate(global_model, X_test, y_test)
        if scheduler.should_terminate(acc, round_num):
            break

    stats = scheduler.get_stats()
    """

    def __init__(self, config: SchedulerConfig | None = None):
        self.cfg = config or SchedulerConfig()

        # Internal state
        self._accuracy_history:         List[float] = []
        self._rounds_without_improvement: int       = 0
        self._total_excluded:             int       = 0
        self._exclusion_log:              List[dict] = []

    # ────────────────────────────────────────────────────────────
    # Client selection
    # ────────────────────────────────────────────────────────────

    def select_clients(
        self,
        clients:      List["EcoFLClient"],
        current_round: int,
    ) -> Tuple[List["EcoFLClient"], List[int]]:
        """
        Select clients eligible for the current aggregation round.

        Each client is tested against:
          (1)  cpu_percent   ≤ θ_CPU  × 100
          (2)  ram_percent   ≤ θ_RAM  × 100
          (3)  energy_remaining ≥ E_min

        Returns
        -------
        eligible      : list of EcoFLClient — admitted to this round
        excluded_ids  : list of int         — client IDs excluded
        """
        eligible:     List["EcoFLClient"] = []
        excluded_ids: List[int]           = []
        exclusion_reasons: dict           = {}

        for client in clients:
            status = client.get_resource_status()

            cpu_ok    = status["cpu_percent"] <= self.cfg.cpu_threshold * 100
            ram_ok    = status["ram_percent"] <= self.cfg.ram_threshold * 100
            energy_ok = (
                status["energy_budget_remaining_mj"]
                >= self.cfg.min_energy_budget_mj
            )

            if cpu_ok and ram_ok and energy_ok:
                eligible.append(client)
            else:
                excluded_ids.append(client.client_id)
                self._total_excluded += 1
                exclusion_reasons[client.client_id] = {
                    "cpu_ok":    cpu_ok,
                    "ram_ok":    ram_ok,
                    "energy_ok": energy_ok,
                    "cpu_pct":   status["cpu_percent"],
                    "ram_pct":   status["ram_percent"],
                    "energy_rem": status["energy_budget_remaining_mj"],
                }

        # ── Fallback: enforce K_min ──────────────────────────────
        if len(eligible) < self.cfg.min_clients_per_round:
            # Not enough eligible → include all (degrade gracefully)
            eligible = clients
            excluded_ids = []

        self._exclusion_log.append({
            "round":        current_round,
            "n_eligible":   len(eligible),
            "n_excluded":   len(excluded_ids),
            "excluded_ids": excluded_ids,
            "reasons":      exclusion_reasons,
        })

        return eligible, excluded_ids

    # ────────────────────────────────────────────────────────────
    # Adaptive termination
    # ────────────────────────────────────────────────────────────

    def should_terminate(
        self,
        current_accuracy: float,
        current_round:    int,
    ) -> bool:
        """
        Decide whether to terminate federation early.

        Criteria:
          • Hard limit  : current_round >= max_rounds
          • Convergence : |A(t) - A(t-1)| < δ for p consecutive rounds

        Returns True → terminate, False → continue.
        """
        if current_round >= self.cfg.max_rounds:
            return True

        self._accuracy_history.append(current_accuracy)

        if len(self._accuracy_history) < 2:
            return False

        improvement = abs(
            self._accuracy_history[-1] - self._accuracy_history[-2]
        )

        if improvement < self.cfg.convergence_tolerance:
            self._rounds_without_improvement += 1
        else:
            self._rounds_without_improvement = 0

        return self._rounds_without_improvement >= self.cfg.patience

    # ────────────────────────────────────────────────────────────
    # Statistics
    # ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Summary statistics for post-experiment analysis."""
        return {
            "rounds_run":                  len(self._accuracy_history),
            "total_excluded_events":       self._total_excluded,
            "rounds_without_improvement":  self._rounds_without_improvement,
            "final_accuracy":              (
                self._accuracy_history[-1]
                if self._accuracy_history else 0.0
            ),
            "accuracy_history":            self._accuracy_history,
            "exclusion_log":               self._exclusion_log,
        }

    def reset(self):
        """Reset state for a new experiment run."""
        self._accuracy_history         = []
        self._rounds_without_improvement = 0
        self._total_excluded           = 0
        self._exclusion_log            = []

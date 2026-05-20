"""
EcoFL System Monitor
====================
Background thread-based system resource monitor.
Tracks CPU utilization, RAM consumption, and estimates
energy consumption during FL training and inference operations.

Energy estimation model:
    E_hat = P_TDP * alpha_CPU * t_duration

where:
    P_TDP    = device thermal design power (W)
    alpha_CPU = mean CPU utilization fraction [0, 1]
    t_duration = operation duration (s)
"""

import psutil
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class SystemSnapshot:
    """Single point-in-time resource sample."""
    timestamp: float
    cpu_percent: float
    ram_mb: float


@dataclass
class SystemReport:
    """Aggregated resource report for one operation."""
    mean_cpu_percent: float
    peak_ram_mb: float
    mean_ram_mb: float
    duration_s: float
    estimated_energy_mj: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mean_cpu_percent":    round(self.mean_cpu_percent, 2),
            "peak_ram_mb":         round(self.peak_ram_mb, 2),
            "mean_ram_mb":         round(self.mean_ram_mb, 2),
            "duration_s":          round(self.duration_s, 4),
            "estimated_energy_mj": round(self.estimated_energy_mj, 4),
            "n_samples":           self.n_samples,
        }

    @classmethod
    def empty(cls) -> "SystemReport":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)


# ─────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────

class SystemMonitor:
    """
    Non-blocking background resource monitor.

    Usage
    -----
    monitor = SystemMonitor(tdp_watts=6.4)
    monitor.start()
    ... # operation under measurement
    report = monitor.stop()   # returns SystemReport
    print(report.to_dict())

    Args
    ----
    sampling_interval : float
        Seconds between resource samples (default 0.5 s).
    tdp_watts : float
        Device thermal design power in watts.
        Raspberry Pi 4  → 6.4 W
        Jetson Nano     → 10.0 W
        Generic desktop → 65.0 W
    """

    # TDP presets for supported hardware profiles
    TDP_PRESETS = {
        "raspberry_pi4": 6.4,
        "jetson_nano":   10.0,
        "desktop":       65.0,
    }

    def __init__(
        self,
        sampling_interval: float = 0.5,
        tdp_watts: float = 6.4,
    ):
        self.sampling_interval = sampling_interval
        self.tdp_watts = tdp_watts

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshots: List[SystemSnapshot] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start background sampling. Resets previous measurements."""
        with self._lock:
            self._snapshots = []
        self._running = True
        self._start_time = time.time()

        # Use process-level measurements for reproducible profiling.
        # Host-level RAM can be misleading on development machines and CI runners;
        # RSS better reflects the memory footprint of this benchmark process.
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # warm-up: first call returns 0

        self._thread = threading.Thread(
            target=self._sample_loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> SystemReport:
        """Stop sampling and return aggregated SystemReport."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._compute_report()

    def get_current_status(self) -> dict:
        """
        Instantaneous resource status for scheduler decisions.
        Blocks for ~0.1 s to get a stable CPU reading.
        """
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return {
            "cpu_percent":        cpu,
            "ram_percent":        mem.percent,
            "ram_available_mb":   mem.available / (1024 * 1024),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_loop(self):
        process = getattr(self, "_process", psutil.Process())
        while self._running:
            try:
                cpu = process.cpu_percent(interval=None)
                ram_mb = process.memory_info().rss / (1024 * 1024)
            except psutil.Error:
                cpu = psutil.cpu_percent(interval=None)
                ram_mb = psutil.virtual_memory().used / (1024 * 1024)

            snap = SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu,
                ram_mb=ram_mb,
            )
            with self._lock:
                self._snapshots.append(snap)
            time.sleep(self.sampling_interval)

    def _compute_report(self) -> SystemReport:
        with self._lock:
            snaps = list(self._snapshots)

        if not snaps:
            return SystemReport.empty()

        cpu_vals = [s.cpu_percent for s in snaps]
        ram_vals = [s.ram_mb for s in snaps]

        duration = snaps[-1].timestamp - self._start_time
        mean_cpu = float(np.mean(cpu_vals))
        # Energy = P_TDP * CPU_fraction * duration → convert J to mJ
        energy_mj = self.tdp_watts * (mean_cpu / 100.0) * duration * 1000.0

        return SystemReport(
            mean_cpu_percent=mean_cpu,
            peak_ram_mb=float(np.max(ram_vals)),
            mean_ram_mb=float(np.mean(ram_vals)),
            duration_s=float(duration),
            estimated_energy_mj=energy_mj,
            n_samples=len(snaps),
        )

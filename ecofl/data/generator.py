"""
EcoFL Data Generator
=====================
Generates synthetic multivariate IoT sensor telemetry data
and partitions it into non-IID client splits for federated
learning simulation.

Dataset characteristics
-----------------------
- 12 sensor features (temperature, humidity, vibration,
  current, voltage, pressure, light, CO₂, 4 derived stats)
- Binary anomaly labels (0 = normal, 1 = anomaly)
- Configurable anomaly rate (default 5%)
- Non-IID client splits via Dirichlet distribution
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple


# ─────────────────────────────────────────────
# Dataset generation
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "temperature_c",
    "humidity_pct",
    "vibration_g",
    "current_a",
    "voltage_v",
    "pressure_hpa",
    "light_lux",
    "co2_ppm",
    "temp_rolling_mean",
    "temp_rolling_std",
    "vibration_rolling_mean",
    "vibration_rolling_std",
]


def generate_iot_telemetry(
    n_samples:    int   = 50_000,
    n_features:   int   = 12,
    anomaly_rate: float = 0.05,
    noise_level:  float = 0.1,
    random_state: int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multivariate IoT sensor telemetry.

    Anomalies are introduced via:
      1. Gaussian distribution shift (mean offset)
      2. Contextual spike injection (per-feature outliers)

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — binary labels
    """
    rng = np.random.RandomState(random_state)
    n_features = min(n_features, len(FEATURE_NAMES))

    n_anomaly = int(n_samples * anomaly_rate)
    n_normal  = n_samples - n_anomaly

    # Nominal sensor means
    base_means = np.array([
        22.0, 55.0, 0.05, 2.5, 220.0,
        1013.0, 300.0, 400.0,
        21.5, 0.8, 0.04, 0.01
    ])[:n_features]

    # Build positive-definite covariance with sensor correlations
    cov = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = rng.uniform(0.05, 0.35)
            cov[i, j] = r
            cov[j, i] = r
    cov += np.eye(n_features) * 1.5          # ensure PD

    # --- Normal samples ---
    X_normal = rng.multivariate_normal(base_means, cov, n_normal)
    X_normal += rng.normal(0, noise_level, X_normal.shape)

    # --- Anomaly samples: very subtle shift, high overlap with normal ---
    # Shift only 3–8% of mean on a minority of features
    shift = rng.uniform(0.03, 0.08, n_features)
    active_features = rng.choice(n_features, size=max(2, n_features // 4), replace=False)
    shift_vec = np.zeros(n_features)
    shift_vec[active_features] = shift[active_features]
    anomaly_means = base_means * (1 + shift_vec)

    # Nearly same covariance → strong class overlap
    X_anomaly = rng.multivariate_normal(anomaly_means, cov * 1.15, n_anomaly)
    X_anomaly += rng.normal(0, noise_level * 2.0, X_anomaly.shape)

    # Spikes on only ~15% of anomaly samples
    n_spikes = max(1, int(n_anomaly * 0.15))
    spike_idx  = rng.choice(n_anomaly, size=n_spikes, replace=False)
    spike_feat = rng.choice(n_features, size=n_spikes)
    for idx, feat in zip(spike_idx, spike_feat):
        sigma = np.std(X_normal[:, feat]) + 1e-8
        X_anomaly[idx, feat] += rng.uniform(1.2, 2.2) * sigma

    # Concatenate and shuffle
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([
        np.zeros(n_normal,  dtype=int),
        np.ones (n_anomaly, dtype=int),
    ])

    # --- Label noise: flip ~8% of labels to simulate sensor mis-labelling ---
    n_flip = int(n_samples * 0.08)
    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float32), y[perm]


# ─────────────────────────────────────────────
# Partitioning strategies
# ─────────────────────────────────────────────

def partition_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients:    int = 10,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data into equal IID splits."""
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in splits]


def partition_noniid_dirichlet(
    X:            np.ndarray,
    y:            np.ndarray,
    n_clients:    int   = 10,
    alpha:        float = 0.5,
    random_state: int   = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID partition via per-class Dirichlet sampling.

    Lower alpha  → more heterogeneous (non-IID)
    Higher alpha → more homogeneous (IID-like)

    alpha = 0.5  used in EcoFL experiments (moderately non-IID)
    """
    rng     = np.random.RandomState(random_state)
    classes = np.unique(y)
    client_buckets: List[List[np.ndarray]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        # Dirichlet proportions over clients
        props = rng.dirichlet(alpha * np.ones(n_clients))
        props = (props * len(cls_idx)).astype(int)

        # Fix rounding so sum == len(cls_idx)
        deficit = len(cls_idx) - props.sum()
        for i in range(abs(deficit)):
            if deficit > 0:
                props[i % n_clients] += 1
            else:
                props[i % n_clients] = max(1, props[i % n_clients] - 1)

        props = np.maximum(props, 1)

        splits = np.split(cls_idx, np.cumsum(props[:-1]))
        for client_id, split in enumerate(splits):
            if len(split) > 0:
                client_buckets[client_id].append(split)

    partitions = []
    for client_id in range(n_clients):
        if client_buckets[client_id]:
            idx = np.concatenate(client_buckets[client_id])
            rng.shuffle(idx)
        else:
            idx = np.array([], dtype=int)
        partitions.append((X[idx], y[idx]))

    return partitions


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def preprocess(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, apply to both splits."""
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)
    return X_tr_sc, X_te_sc, scaler


def make_dataset(
    n_samples:    int   = 50_000,
    n_clients:    int   = 10,
    anomaly_rate: float = 0.05,
    alpha:        float = 0.5,
    test_size:    float = 0.20,
    random_state: int   = 42,
) -> dict:
    """
    Convenience wrapper: generate, split, scale, partition.

    Returns
    -------
    dict with keys:
        X_train_all, y_train_all,
        X_test, y_test,
        client_partitions,
        scaler,
        meta  (dataset statistics)
    """
    X, y = generate_iot_telemetry(
        n_samples=n_samples,
        anomaly_rate=anomaly_rate,
        random_state=random_state,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_tr_sc, X_te_sc, scaler = preprocess(X_tr, X_te)

    partitions = partition_noniid_dirichlet(
        X_tr_sc, y_tr,
        n_clients=n_clients,
        alpha=alpha,
        random_state=random_state,
    )

    meta = {
        "n_samples":       n_samples,
        "n_features":      X.shape[1],
        "n_train":         len(y_tr),
        "n_test":          len(y_te),
        "anomaly_rate":    float(y.mean()),
        "n_clients":       n_clients,
        "dirichlet_alpha": alpha,
        "client_sizes":    [len(p[1]) for p in partitions],
    }

    return {
        "X_train_all":       X_tr_sc,
        "y_train_all":       y_tr,
        "X_test":            X_te_sc,
        "y_test":            y_te,
        "client_partitions": partitions,
        "scaler":            scaler,
        "meta":              meta,
    }

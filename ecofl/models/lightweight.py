"""
EcoFL Lightweight Model Registry
=================================
Factory, serialization, and FedAvg aggregation for all
lightweight model families evaluated in EcoFL.

Federated strategies
--------------------
parameter_avg   : FedAvg weighted averaging of model weights.
                  Supported by: Logistic Regression, MLP.

prediction_ensemble : Clients train independently; server
                      aggregates via weighted majority vote.
                      Used for: Random Forest, XGBoost, Isolation Forest
                      (tree structures cannot be meaningfully averaged).
"""

import pickle
import numpy as np
from typing import Any, Dict, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, dict] = {
    "LogisticRegression": {
        "class":              LogisticRegression,
        "params":             {
            "max_iter": 1000,
            "C":        1.0,
            "solver":   "lbfgs",
            "random_state": 42,
        },
        "federated_strategy": "parameter_avg",
        "description":        "Linear baseline with L2 regularisation",
    },
    "RandomForest": {
        "class":              RandomForestClassifier,
        "params":             {
            "n_estimators": 50,
            "max_depth":    8,
            "n_jobs":       1,
            "random_state": 42,
        },
        "federated_strategy": "prediction_ensemble",
        "description":        "50-tree ensemble; depth-capped for RAM compliance",
    },
    "XGBoost": {
        "class":              XGBClassifier,
        "params":             {
            "n_estimators":  50,
            "max_depth":     4,
            "learning_rate": 0.1,
            "n_jobs":        1,
            "random_state":  42,
            "eval_metric":   "logloss",
            "verbosity":     0,
            "use_label_encoder": False,
        },
        "federated_strategy": "prediction_ensemble",
        "description":        "Gradient-boosted trees; CPU-optimised",
    },
    "MLP": {
        "class":              MLPClassifier,
        "params":             {
            "hidden_layer_sizes": (64, 32),
            "activation":         "relu",
            "solver":             "adam",
            "max_iter":           200,
            "random_state":       42,
            "early_stopping":     False,
            "n_iter_no_change":   10,
            "tol":                1e-4,
        },
        "federated_strategy": "parameter_avg",
        "description":        "2-layer MLP: 64→32→output",
    },
    "IsolationForest": {
        "class":              IsolationForest,
        "params":             {
            "n_estimators":  50,
            "contamination": 0.05,
            "random_state":  42,
        },
        "federated_strategy": "prediction_ensemble",
        "description":        "Unsupervised anomaly detection baseline",
    },
}


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_model(model_name: str) -> Any:
    """Instantiate a fresh model by name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    cfg = MODEL_CONFIGS[model_name]
    # Strip unknown kwargs for older sklearn versions
    params = {k: v for k, v in cfg["params"].items()
              if k != "use_label_encoder"}
    return cfg["class"](**params)


# ─────────────────────────────────────────────
# Parameter extraction / injection (FedAvg)
# ─────────────────────────────────────────────

def get_model_parameters(model: Any, model_name: str) -> Dict:
    """
    Extract model parameters for federated aggregation.
    For tree models, returns a serialised pickle payload.
    """
    if model_name == "LogisticRegression":
        if not hasattr(model, "coef_"):
            return None
        return {
            "coef":      model.coef_.copy(),
            "intercept": model.intercept_.copy(),
            "classes":   model.classes_.copy(),
        }
    elif model_name == "MLP":
        if not hasattr(model, "coefs_"):
            return None
        return {
            "coefs":      [c.copy() for c in model.coefs_],
            "intercepts": [i.copy() for i in model.intercepts_],
        }
    else:
        # Tree models: full model pickle
        return {"model_bytes": pickle.dumps(model)}


def set_model_parameters(model: Any, params: Dict, model_name: str) -> Any:
    """Inject aggregated parameters back into a model instance."""
    if model_name == "LogisticRegression":
        model.coef_      = params["coef"]
        model.intercept_ = params["intercept"]
        model.classes_   = params["classes"]
    elif model_name == "MLP":
        model.coefs_      = params["coefs"]
        model.intercepts_ = params["intercepts"]
    # Tree models: no injection (prediction ensemble strategy)
    return model


def aggregate_parameters(
    params_list: List[Dict],
    weights: List[float],
    model_name: str,
) -> Dict:
    """
    FedAvg weighted aggregation.

    For parameter_avg models  → weighted mean of weight matrices.
    For prediction_ensemble   → returns params from the largest client
                                (used only as a fallback reference).
    """
    total = sum(weights)
    w = [wk / total for wk in weights]

    if model_name == "LogisticRegression":
        coef      = sum(wi * p["coef"]      for wi, p in zip(w, params_list))
        intercept = sum(wi * p["intercept"] for wi, p in zip(w, params_list))
        return {"coef": coef, "intercept": intercept,
                "classes": params_list[0]["classes"]}

    elif model_name == "MLP":
        n_layers  = len(params_list[0]["coefs"])
        coefs     = [
            sum(wi * p["coefs"][i] for wi, p in zip(w, params_list))
            for i in range(n_layers)
        ]
        intercepts = [
            sum(wi * p["intercepts"][i] for wi, p in zip(w, params_list))
            for i in range(n_layers)
        ]
        return {"coefs": coefs, "intercepts": intercepts}

    else:
        # Return params from highest-weight client
        best = int(np.argmax(weights))
        return params_list[best]


# ─────────────────────────────────────────────
# Communication overhead estimation
# ─────────────────────────────────────────────

def get_model_size_kb(model: Any, model_name: str) -> float:
    """
    Estimate per-round communication cost as serialised
    parameter size in kilobytes.
    """
    params = get_model_parameters(model, model_name)
    return len(pickle.dumps(params)) / 1024.0

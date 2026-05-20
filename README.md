[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20314541.svg)](https://doi.org/10.5281/zenodo.20314541)


# EcoFL: Energy-Conscious Federated Learning

EcoFL is a reproducible Python framework for benchmarking energy-aware federated learning on resource-constrained IoT edge devices. It accompanies the manuscript:

**Can Federated Learning Go Green? EcoFL: A System-Level Energy-Aware Benchmark for IoT Edge Intelligence**

The framework compares centralized training, standard FedAvg, and EcoFL's energy-aware federated configuration across lightweight model families under emulated Raspberry Pi 4 / Jetson Nano hardware profiles.

## Main features

- Synthetic IoT telemetry generator with 50,000 samples, 12 features, 5% anomaly injection, 8% label noise, and Dirichlet non-IID client partitioning.
- Five lightweight model families: Logistic Regression, Random Forest, XGBoost, Multilayer Perceptron, and Isolation Forest.
- Three training configurations: centralized, FedAvg, and EcoFL.
- TDP-based computation-side energy estimation: `E = P_TDP × mean CPU utilization × training duration`.
- Energy-aware scheduler with CPU/RAM/energy thresholds and adaptive round termination.
- Reproducible experiment scripts, statistical tests, ablation study, and figure generation.

## Repository structure

```text
ecofl/
├── ecofl/
│   ├── benchmark/       # End-to-end benchmark pipeline
│   ├── data/            # Synthetic IoT telemetry generator and non-IID partitioning
│   ├── energy/          # System monitoring and TDP-based energy model
│   ├── federated/       # Client, server, and EcoFL scheduler
│   └── models/          # Lightweight model registry and aggregation helpers
├── experiments/         # Main experiment, ablation, statistics, and visualization scripts
├── results/             # Reproducibility artefacts from the paper experiments
├── tests/               # Minimal smoke tests
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
└── reproduce.sh
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Reproducing the paper experiments

The default full run uses five seeds (`42–46`), 50,000 samples, 10 clients, and 20 maximum FL rounds:

```bash
python experiments/run_experiments.py --profile raspberry_pi4 --seeds 5 --rounds 20 --clients 10 --samples 50000
python experiments/statistical_tests.py
python experiments/ablation_study.py
python experiments/visualize_results.py
```

A convenience script is also provided:

```bash
bash reproduce.sh
```

For a fast smoke run:

```bash
python experiments/run_experiments.py --quick
```

## Optional hardware emulation with cgroups

To emulate Raspberry Pi 4-like constraints on Linux:

```bash
systemd-run --scope -p CPUQuota=25% -p MemoryMax=1G \
  python experiments/run_experiments.py --profile raspberry_pi4 --seeds 5
```

## Precomputed results

The `results/` directory contains JSON result artefacts and PDF figures generated from the paper experiments. These files are included to support inspection and reproduction of the reported results.

## Important implementation note

The MLP configuration in the released code uses `max_iter=200` to reproduce the archived result files. If the manuscript states `max_iter=50`, update the manuscript table or regenerate all results after changing the model configuration.

## Data and code availability statement

The EcoFL source code, synthetic IoT telemetry data generator, configuration scripts, experiment outputs, statistical tests, and figure-generation scripts are publicly available in this repository. A permanent archived version should be deposited in Zenodo after GitHub release and cited using the generated DOI.

## License

This project is released under the MIT License.

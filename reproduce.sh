#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiments.py --profile raspberry_pi4 --seeds 5 --rounds 20 --clients 10 --samples 50000
python experiments/statistical_tests.py
python experiments/ablation_study.py
python experiments/visualize_results.py

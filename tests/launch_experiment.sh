#!/bin/bash
# Launcher: activates conda env and runs experiment under nohup.
# The 'exec' replaces this shell with python, so nohup on bash = nohup on python.
# Usage: nohup bash tests/launch_experiment.sh > logs/launch.txt 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"
mkdir -p logs

source /home/jwlee/miniconda3/etc/profile.d/conda.sh
conda activate fpga

echo "[$(date +%T)] Conda activated: $(python --version)"
echo "[$(date +%T)] Starting experiment_full_comparison.py ..."
exec python -u tests/experiment_full_comparison.py

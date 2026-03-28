#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/starter_ecoli_heatshock.yaml}"
python run_script.py src/01_audit_dataset.py "$CFG"
python run_script.py src/02_build_state_table.py "$CFG"
python run_script.py src/03_map_geometry.py "$CFG"
python run_script.py src/04_fit_residual_field.py "$CFG"
python run_script.py src/05_build_recoverability.py "$CFG"
python run_script.py src/06_trajectory_velocity.py "$CFG"
python run_script.py src/07_validation.py "$CFG"
python run_script.py src/08_variance_scaling.py "$CFG"

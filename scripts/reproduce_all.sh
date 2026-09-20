#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CENTRAL_ENV=${SCHUR_CENTRAL_ENV:-schur-central}
RAY_ENV=${SCHUR_RAY_ENV:-schur-ray}
cd "${ROOT}"

conda run --no-capture-output -n "${CENTRAL_ENV}" scripts/run_centralized.sh
conda run --no-capture-output -n "${RAY_ENV}" scripts/run_distributed.sh
conda run --no-capture-output -n "${CENTRAL_ENV}" \
  python -m experiments.plot_main --result-root results/main --output-dir figures
conda run --no-capture-output -n "${CENTRAL_ENV}" scripts/run_randomness.sh


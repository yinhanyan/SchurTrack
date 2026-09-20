#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
CPU=${SCHUR_CPU:-0}
CPU_SET=${SCHUR_RANDOMNESS_CPUS:-0-15}
WORKERS=${SCHUR_RANDOMNESS_WORKERS:-8}
DATA=${RANDOM_NOISY_DATA:-"${ROOT}/data/random-noisy-10000x500.npy"}
RESULT=${ROOT}/results/attp-randomness

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

test -f "${DATA}" || { echo "Missing ${DATA}; run prepare_datasets.py"; exit 1; }
cd "${ROOT}"

# Timing is serialized and pinned exactly like the centralized main runs.
taskset -c "${CPU}" "${PYTHON}" -m experiments.run_attp_randomness \
  --result-dir "${RESULT}" --dataset "${DATA}" --rows 1500 \
  --qs 1,2,4,8,10 --seeds 100 --seed-start 130000 \
  --workers 1 --calibration-repeats 3 --calibrate-only

# Only the error-distribution trials are parallelized; each worker gets one CPU.
taskset -c "${CPU_SET}" "${PYTHON}" -m experiments.run_attp_randomness \
  --result-dir "${RESULT}" --dataset "${DATA}" --rows 1500 \
  --qs 1,2,4,8,10 --seeds 100 --seed-start 130000 \
  --workers "${WORKERS}" --calibration-repeats 3

"${PYTHON}" -m experiments.render_attp_table \
  --result-dir "${RESULT}" --output-prefix "${RESULT}/table"
"${PYTHON}" -m experiments.plot_attp_randomness \
  --result-dir "${RESULT}" --output "figures/attp-randomness-quality.png"


#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CPU=${SCHUR_CPU:-0}
PYTHON=${PYTHON:-python}
GLOVE=${GLOVE_DATA:-"${ROOT}/data/glove-10000x300.npy"}
RANDOM_NOISY=${RANDOM_NOISY_DATA:-"${ROOT}/data/random-noisy-16000x500.npy"}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

test -f "${GLOVE}" || { echo "Missing ${GLOVE}; run prepare_datasets.py"; exit 1; }
test -f "${RANDOM_NOISY}" || { echo "Missing ${RANDOM_NOISY}; run prepare_datasets.py"; exit 1; }
cd "${ROOT}"

for ell in 5 10 20 30 50 100 125 140; do
  echo "[SW] ell=${ell}"
  taskset -c "${CPU}" "${PYTHON}" -m experiments.run_main \
    --tasks sw --input "${GLOVE}" --rows 10000 --window 5000 \
    --query-step 20 --ell "${ell}" --seed 0 \
    --output-dir "results/main/sw/ell-${ell}"
done

for ell in 2 5 10 20 30 50 100 150 200 240; do
  echo "[ATTP] ell=${ell}"
  taskset -c "${CPU}" "${PYTHON}" -m experiments.run_main \
    --tasks attp --input "${RANDOM_NOISY}" --rows 3000 \
    --query-step 20 --ell "${ell}" --seed 0 \
    --output-dir "results/main/attp/ell-${ell}"
done


#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CPU_SET=${SCHUR_DISTRIBUTED_CPUS:-0-4}
PYTHON=${PYTHON:-python}
RANDOM_NOISY=${RANDOM_NOISY_DATA:-"${ROOT}/data/random-noisy-16000x500.npy"}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export SCHUR_RAY_NUM_CPUS=5

test -f "${RANDOM_NOISY}" || { echo "Missing ${RANDOM_NOISY}; run prepare_datasets.py"; exit 1; }
cd "${ROOT}"

for ell in 5 10 50 100 200 240; do
  echo "[Dist] ell=${ell}"
  taskset -c "${CPU_SET}" "${PYTHON}" -m experiments.run_main \
    --tasks dist --distributed-backend ray --input "${RANDOM_NOISY}" \
    --rows 10000 --sites 4 --query-step 20 --ell "${ell}" --seed 0 \
    --output-dir "results/main/dist/ell-${ell}"
done

for ell in 5 10 50 120 240; do
  echo "[Distributed window] ell=${ell}"
  taskset -c "${CPU_SET}" "${PYTHON}" -m experiments.run_main \
    --tasks dsw --distributed-backend ray --input "${RANDOM_NOISY}" \
    --rows 16000 --window 6000 --sites 4 --query-step 20 \
    --ell "${ell}" --seed 0 \
    --output-dir "results/main/dsw/ell-${ell}"
done


# SchurTrack covariance-tracking experiments

This directory is the self-contained artifact for the covariance-tracking
experiments of SchurTrack. It reproduces the artifacts used
by the experimental section:

1. the sixteen panels in the four-task performance comparison;
2. the deterministic-versus-probabilistic summary table; and
3. the two-panel error-distribution figure.

## Repository layout

- `base/`: Frequent Directions, AeroSketch primitives, and the SchurTrack core.
- `algorithm/sw/`: centralized sliding-window methods, including the baselines, Fast-DS-FD, and AeroSketch.
- `algorithm/attp/`: PFD, AeroSketch, SchurTrack, and the instrumented
  AeroSketch used in the deterministic-versus-probabilistic study.
- `algorithm/distributed/`: Ray implementations for distributed prefix
  tracking. 
- `algorithm/dsw/`: Ray implementations for DA2,
  AeroSketch, and SchurTrack. 
- `experiments/run_main.py`: common measurement protocol for all four tasks.
- `experiments/plot_main.py`: generates the 4-by-4 paper figure grid.
- `experiments/run_attp_randomness.py`: update-time calibration and the
  100-seed ATTP error study.
- `experiments/render_attp_table.py` and
  `experiments/plot_attp_randomness.py`: generate the ATTP table and figure.
- `scripts/prepare_datasets.py`: prepares compact NumPy input matrices.

## Environments

The distributed experiments require Ray 2.46.0 and therefore use its
compatible Python 3.9 environment. The centralized experiments do not
require Ray and use Python 3.12. Recreate both environments:

```bash
conda env create -f environment-centralized.yml
conda env create -f environment-distributed.yml
```

All timing scripts force OpenMP, OpenBLAS, MKL, and NumExpr to one thread.
Centralized runs are pinned to logical CPU 0. Distributed runs use logical
CPUs 0--4 for one coordinator plus four site actors. Override these defaults
with `SCHUR_CPU`, `SCHUR_DISTRIBUTED_CPUS`, or
`SCHUR_RANDOMNESS_CPUS` only when reproducing on a different machine.

## Data preparation

The preparation script automatically downloads the official
[GloVe 6B archive](https://nlp.stanford.edu/data/glove.6B.zip) and extracts
only `glove.6B.300d.txt`. The archive and extracted text file are cached under
`data/raw/`, so subsequent runs reuse these cached files instead of downloading
the archive again. The random-noisy workload is regenerated from the model in
the paper using a fixed random seed:

```bash
conda activate schur-central
python scripts/prepare_datasets.py --seed 0
```

The script writes `data/glove-10000x300.npy`,
`data/random-noisy-10000x500.npy`, and SHA-256 metadata. 

## Reproduce the main 4-by-4 figure grid

Run the centralized tasks in the centralized environment:

```bash
conda activate schur-central
./scripts/run_centralized.sh
```

This reruns Fast-DS-FD, PFD, AeroSketch, and SchurTrack for every width parameter.

Run the two distributed tasks in the Ray environment:

```bash
conda activate schur-ray
./scripts/run_distributed.sh
```

This starts a fresh local Ray instance for every width and reruns P2, DA2,
AeroSketch, and SchurTrack. 

After both runs, generate all sixteen PDF and PNG panels:

```bash
conda activate schur-central
python -m experiments.plot_main \
  --result-root results/main --output-dir figures
```

The output names match the manuscript, for example
`figures/sw-time-avg-error.pdf`, `figures/attp-cost-max-error.pdf`, and
`figures/dsw-cost-avg-error.pdf`. The update-time axis is logarithmic. The
blue curve is AeroSketch time divided by SchurTrack time at matched empirical
error. The green curve is SchurTrack/AeroSketch memory or communication at
matched empirical error.

## Reproduce the ATTP determinism study

```bash
conda activate schur-central
./scripts/run_randomness.sh
```

It evaluates the same 1,500-row random-noisy stream at every historical
prefix for SchurTrack and 100 common random seeds for each AeroSketch setting
`q_PI=q_SI` in `{1,2,4,8,10}` in parallel. By default eight workers share CPUs 0--15; set
`SCHUR_RANDOMNESS_WORKERS` to a lower value if the machine has fewer physical
cores.

Outputs are:

- `results/attp-randomness/quality.json` and `.csv`: raw per-seed results;
- `results/attp-randomness/timing.json`: isolated update calibration;
- `results/attp-randomness/table.tsv` and `.md`: the table summary in
  machine-readable and Markdown formats; the same table is printed to the
  terminal;
- `figures/attp-randomness-quality.pdf` and `.png`: the paper figure.

The JSONL checkpoint permits an interrupted seed sweep to resume. Remove the
result directory before changing the dataset, width, query protocol, or seed
range.

## One command

After preparing the data and creating both named conda environments, the full
artifact can be launched with:

```bash
./scripts/reproduce_all.sh
```

Use `SCHUR_CENTRAL_ENV` and `SCHUR_RAY_ENV` if the environment names differ.
# Vial defect classifier – MLOps project (group Yap-IT)

A small CNN that looks at a photo of a pharmaceutical glass vial and says whether it is **clean (0)** or **defective (1)**.

The teachers gave us the model as two Jupyter notebooks. Our job is to turn them into a production pipeline: Python modules in a standard project structure, data versioned with DVC, experiments tracked with MLflow, a Dagger pipeline in Go run by GitHub Actions, and the model served as an API.

The original assignment is in [docs/assignment.md](docs/assignment.md). The dataset is described in [docs/pharmaceutical-vial-dataset.md](docs/pharmaceutical-vial-dataset.md).

## Team

| Member | Part |
| --- | --- |
| Ruben Coloma Garcia | Project structure, data pipeline, drift detection |
| Elena Linares Leal | Model, training and evaluation |
| Diego Plaza Vinuesa | Dagger pipeline and GitHub Actions |
| Andrea Cela | Serving API and deployment |

We work with short branches, semantic commit messages and pull requests that another member reviews before merging.

## Project structure

The layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/):

```
├── data/
│   ├── images/
│   │   ├── metadata.csv.dvc   <- which image is clean/defective (versioned with DVC)
│   │   └── raw/               <- the 730 images (downloaded, not in git)
│   ├── test/                  <- 10 extra test images + their metadata (DVC)
│   └── processed/             <- train/val/test split csv files (generated, not in git)
├── docs/                      <- assignment, dataset description, baseline metrics, diagrams
├── notebooks/                 <- the original notebooks from the teachers (kept for comparison)
├── src/
│   ├── config.py              <- all paths and settings in one place
│   ├── dataset.py             <- step 1: download the images
│   ├── features.py            <- step 2: validate the data, split it, transforms + PyTorch Dataset
│   ├── monitoring.py          <- drift detection (baseline + KS tests)
│   └── modeling/              <- model, training, evaluation, prediction
├── tests/                     <- pytest tests
├── pyproject.toml             <- project info and dependencies
└── uv.lock                    <- exact versions of every dependency
```

## Setup

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 for you).

```bash
git clone https://github.com/elenalinares/itu-dsip-project.git
cd itu-dsip-project
uv sync            # creates .venv with the exact versions from uv.lock
uv run dvc pull    # gets the metadata csv files tracked by DVC
```

## How to run

Run every step as a module (`python -m src.<step>`) from the project root.

### 1. Data

```bash
uv run python -m src.dataset    # downloads the images listed in the metadata (skips the ones already there)
uv run python -m src.features   # validates the metadata and writes data/processed/{train,val,test}.csv
```

The split is stratified 70/15/15 with seed 42, the same as the original notebook: 510 train, 108 validation, 112 test images, each set with about 21% clean images.

### 2. Training

> TODO (Elena): how to train the model and where the model artifact ends up.

### 3. Pipeline

> TODO (Diego): how to run the Dagger pipeline locally and in GitHub Actions.

### 4. Serving

> TODO (Andrea): how to run the API (`/health`, `/version`, `/infer`, `/drift`) and how it is deployed.

## Monitoring (drift detection)

`src/monitoring.py` checks whether the requests the API receives still look like the data the model was trained on.

- After training, `compute_baseline_statistics` runs the model on the validation set and saves a baseline (image statistics, prediction counts, confidence scores) to `artifacts/drift_baseline_stats.json`.
- `detect_drift` compares the requests logged in the last 7 days with that baseline using Kolmogorov-Smirnov tests on the confidence scores and on the predicted classes. A p-value below 0.05 is reported as drift.

## Tests

```bash
uv run pytest
```

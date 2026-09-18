# CAD CT phenotype analysis

Code for unsupervised phenotyping of coronary artery disease from multidomain cardiac CT
(plaque quantification, CT-FFR and CT myocardial perfusion), projection of the phenotypes to
an external cohort with a fixed 13-feature classifier, and assessment of their incremental
prognostic value for composite and hard MACE.

Patient data are not distributed. The repository contains the analysis code, the frozen
classifier parameters and the configuration used for the study.

## Layout

| Path | Content |
|---|---|
| `run.py` | Pipeline driver; runs the stages below and logs every script call |
| `predict.py` | Command-line phenotype assignment with the frozen classifier |
| `config.yaml` | All analysis settings (preprocessing, clustering, classifier, survival models, bootstrap counts) |
| `artifacts/final_projection_model.json` | Scaler and coefficients of the 13-feature multinomial classifier |
| `scripts/` | One script per analysis step (see below) |
| `tests/` | Unit tests of the statistical kernels and input handling |
| `examples/calculator_input.csv` | Example input row for `predict.py` |
| `examples/synthetic/` | Two artificial 240-patient cohorts (and their generator) for running the pipeline without patient data |
| `docs/` | Data format and statistical methods |

## Installation

Python 3.9–3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

R with the `cmprsk` package is needed only for the `competing` stage (Gray's tests).

## Usage

Assign phenotypes to new patients from a CSV containing the 13 classifier features
(column order is free; `IMV` and CT-FFR are fractions in [0, 1]):

```bash
python predict.py --input examples/calculator_input.csv --output results/phenotypes.csv
```

Run the cohort analysis on the derivation and validation tables described in
[docs/data_format.md](docs/data_format.md):

```bash
# quick execution check with very small bootstrap counts
python run.py --mode smoke --derivation data/derivation.csv --validation data/validation.csv --output results/smoke

# study settings (1,000 / 500x50 hard-MACE bootstrap, 4,999 null-bootstrap samples, 5,000 calibration samples)
python run.py --mode full --derivation data/derivation.csv --validation data/validation.csv --output results/full --jobs 8
```

`--stages` selects a comma-separated subset of stages; `--resume` continues an interrupted
run in the same output directory.

### Demo without patient data

`examples/synthetic/` contains two artificial cohorts (IDs prefixed `SYNTHETIC_`) whose values are
drawn from hand-specified distributions and whose outcomes are independent of all covariates. They
exercise the code path only; their results have no scientific meaning.

```bash
python run.py --mode smoke --derivation examples/synthetic/derivation.csv --validation examples/synthetic/validation.csv --output results/demo --jobs 2
```

The seven default smoke stages complete in about one minute and write the incremental-performance
tables to `results/demo/tables/`. The cohorts can be regenerated with
`python examples/generate_synthetic_cohorts.py --output-dir results/synthetic` (seeds are fixed).

## Stages

| Stage | Script | Purpose |
|---|---|---|
| `prepare` | `apply_projection.py` | Validate the cohorts and assign external phenotypes with the frozen classifier |
| `preprocess` | `preprocess.py` | Export the clustering and feature-selection matrices |
| `cluster` | `clustering.py`, `attach_labels.py`, `describe_k2.py` | Metric-wise PCA, K-prototypes for K = 2–5, seed and bootstrap stability, K = 2 description |
| `selection` | `feature_selection.py` | LASSO + Boruta consensus ranking of source variables |
| `feature_count` | `feature_count.py` | Classifier performance versus number of ranked features |
| `stability` | `feature_stability.py`, `plot_feature_stability.py` | Ranking stability across 5x5 cross-validation folds |
| `projection_refit` | `refit_projection.py` | Refit of the 13-feature classifier; `--oof` adds 25-fold out-of-fold evaluation |
| `oof_summary` | `oof_summary.py` | Mean-of-repeats OOF metrics and confusion matrices |
| `survival` | `survival.py` | Ridge Cox Models A–C, composite-MACE C-index/Brier/IBS, phenotype HRs, PH tests, KM curves |
| `hard` | `hard_mace_bootstrap.py`, `competing_cox.py` | Optimism-corrected hard-MACE performance with competing-risk CIF and nested bootstrap intervals |
| `comparisons` | `model_comparison.py` | Penalised LR statistics, null-bootstrap P values, effective-df AIC |
| `hr_sensitivity` | `hr_sensitivity.py` | Hard-MACE phenotype HRs across ridge penalties |
| `interactions` | `medication_interactions.py` | Phenotype-by-medication interaction tests |
| `competing` | `competing_risk_gray.py` (+ R) | Cumulative incidence and Gray's tests |
| `figures` | `figures.py` | KM/forest panels, radar chart, heatmaps, CIF curves, confidence-restricted analyses |
| `calibration` | `calibration.py` | Grouped 36-month calibration (Kaplan-Meier / Aalen-Johansen) |
| `tables` | `tables.py` | Baseline, phenotype-characteristic and incremental-performance tables |



## License

MIT, see `LICENSE`.

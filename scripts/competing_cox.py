"""Cause-specific Cox models for hard MACE with soft MACE as a competing event.

Provides the fitting/evaluation engine used by the hard-MACE bootstrap, model comparison,
HR sensitivity and calibration scripts, and a command-line entry point that runs one
cohort's bootstrap replicates in parallel and stores one JSON record per replicate.
"""

from __future__ import annotations

import os

for name in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import concurrent.futures
import json
import multiprocessing
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from formulaic import model_matrix
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index

from common import validate_study_categories
from graf_ipcw import censor_km, integrate_curve
from survival import FORMULAS, load_analysis_data

MODEL_SIZES = [8, 13, 15]
HORIZONS = [36, 60]
GRID_START = 3.0
GRID_POINTS = 1000
PENALTY = 0.01
STEP_SIZES = [0.5, 0.1, 0.02]
COHORT_SEEDS = {"derivation": 42, "validation": 10042}
CATEGORICAL = ["gender", "Antihypertensive medication", "smoking", "DM", "CADRADS", "Cluster"]

D: pd.DataFrame | None = None
COLS: list[str] = []


def init(cohort: str, input_dir: str | Path) -> None:
    """Load a prepared cohort and expand the Model C design matrix into columns x0..x14."""
    global D, COLS
    folder = Path(input_dir)
    path = folder / ("derivation.csv" if cohort == "derivation" else "validation_projected.csv")
    D = load_analysis_data(path, path)
    D["soft_event"] = D.MACET.eq(2)
    validate_study_categories(D, require_all_levels=True)
    for column in CATEGORICAL:
        D[column] = pd.Categorical(D[column], categories=sorted(D[column].unique()))
    X = model_matrix(FORMULAS["C"], D).drop(columns="Intercept")
    if X.shape[1] != 15:
        raise ValueError(f"Model C design must have 15 coefficients, got {X.shape[1]}")
    for formula, size in zip([FORMULAS["A"], FORMULAS["B"]], [8, 13]):
        names = list(model_matrix(formula, D).drop(columns="Intercept").columns)
        if names != list(X.columns[:size]):
            raise ValueError("Model A/B columns must equal the leading columns of Model C")
    COLS = []
    for k in range(15):
        column = f"x{k}"
        D[column] = X.iloc[:, k].to_numpy()
        COLS.append(column)


def weights(train: pd.DataFrame, evaluation: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """IPCW weights and hard-event indicators on the IBS grid for each horizon."""
    tt, st = train.TIME.to_numpy(), train.MACET.to_numpy()
    t, s = evaluation.TIME.to_numpy(), evaluation.MACET.to_numpy()
    unique, g = censor_km(tt, st != 0)
    g_event = np.r_[1.0, g][np.searchsorted(unique, t, side="left")]
    out = []
    for horizon in HORIZONS:
        grid = np.linspace(GRID_START, horizon, GRID_POINTS)
        g_grid = np.r_[1.0, g][np.searchsorted(unique, grid, side="right")]
        any_event = (s[:, None] != 0) & (t[:, None] <= grid)
        alive = t[:, None] > grid
        if np.any((g_event <= 0) & any_event.any(axis=1)) or np.any((g_grid <= 0) & alive.any(axis=0)):
            raise ValueError("Zero censoring survival within the horizon")
        w = np.divide(
            any_event, g_event[:, None], out=np.zeros(any_event.shape), where=g_event[:, None] > 0
        ) + np.divide(alive, g_grid, out=np.zeros(alive.shape), where=g_grid > 0)
        y = (s[:, None] == 1) & (t[:, None] <= grid)
        out.append((grid, w, y))
    return out


def fit_cox(
    frame: pd.DataFrame, event: str, size: int, precision: float = 1e-8, r_precision: float = 1e-10
) -> CoxPHFitter:
    """Fit a ridge Cox model on the first `size` design columns, retrying with smaller step sizes."""
    last = None
    for step in STEP_SIZES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = CoxPHFitter(penalizer=PENALTY).fit(
                    frame[["TIME", event] + COLS[:size]],
                    "TIME",
                    event,
                    fit_options={
                        "step_size": step,
                        "precision": precision,
                        "r_precision": r_precision,
                        "max_steps": 500,
                    },
                )
            except Exception as error:
                last = repr(error)
                continue
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            last = "; ".join(str(w.message) for w in caught)
            continue
        if not np.isfinite(model.params_).all():
            last = "Nonfinite coefficients"
            continue
        return model
    raise RuntimeError(f"Cox fit did not converge: {last}")


def fit(frame: pd.DataFrame) -> list[list[CoxPHFitter]]:
    """Hard- and soft-event cause-specific models for A, B and C."""
    return [[fit_cox(frame, event, size) for event in ("hard_event", "soft_event")] for size in MODEL_SIZES]


def cumulative_incidence(
    hard: CoxPHFitter, soft: CoxPHFitter, frame: pd.DataFrame, horizon: float = 60.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hard-event CIF on the baseline-hazard time grid using exponential survival increments."""
    times = hard.baseline_hazard_.index.to_numpy()
    if not np.array_equal(times, soft.baseline_hazard_.index.to_numpy()):
        raise ValueError("Cause-specific time grids differ")
    keep = times <= horizon
    times = times[keep]
    d1 = hard.predict_partial_hazard(frame).to_numpy()[:, None] * hard.baseline_hazard_.iloc[keep, 0].to_numpy()
    d2 = soft.predict_partial_hazard(frame).to_numpy()[:, None] * soft.baseline_hazard_.iloc[keep, 0].to_numpy()
    total = d1 + d2
    previous = np.exp(-np.c_[np.zeros(len(frame)), np.cumsum(total, axis=1)[:, :-1]])
    fraction = np.divide(d1, total, out=np.zeros_like(d1), where=total > 0)
    cif = np.cumsum(previous * (-np.expm1(-total)) * fraction, axis=1)
    if np.any(cif < -1e-10) or np.any(cif > 1 + 1e-10):
        raise ValueError("Cumulative incidence outside [0, 1]")
    return times, cif, float(total.max()) if total.size else 0.0


def evaluate(
    models: list[list[CoxPHFitter]], train: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Per model: [cause-specific C-index, Brier36, IBS3-36, Brier60, IBS3-60]."""
    weight_sets = weights(train, evaluation)
    rows = []
    max_jump = 0.0
    for hard, soft in models:
        times, cif, jump = cumulative_incidence(hard, soft, evaluation)
        max_jump = max(max_jump, jump)
        risk = hard.predict_partial_hazard(evaluation).to_numpy()
        values = [float(concordance_index(evaluation.TIME, -risk, evaluation.hard_event))]
        for grid, w, y in weight_sets:
            p = np.c_[np.zeros(len(evaluation)), cif][:, np.searchsorted(times, grid, side="right")]
            brier = (w * (y - p) ** 2).mean(axis=0)
            values.extend([float(brier[-1]), integrate_curve(brier, grid)])
        rows.append(values)
    return np.array(rows), max_jump


def bootstrap_iteration(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    index = rng.integers(0, len(frame), len(frame))
    sample = frame.iloc[index].reset_index(drop=True)
    models = fit(sample)
    train_scores, jump_train = evaluate(models, sample, sample)
    test_scores, jump_test = evaluate(models, sample, frame)
    return train_scores, test_scores, max(jump_train, jump_test)


def job(args: tuple) -> dict:
    cohort, index, phase, inner = args
    started = time.time()
    seed = COHORT_SEEDS[cohort]
    try:
        if index == 0:
            apparent, jump = evaluate(fit(D), D, D)
            return {"index": 0, "apparent": apparent.tolist(), "jump": jump, "seconds": time.time() - started}
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        sample = D.iloc[rng.integers(0, len(D), len(D))].reset_index(drop=True)
        models = fit(sample)
        train_scores, jump_train = evaluate(models, sample, sample)
        test_scores, jump_test = evaluate(models, sample, D)
        result = {
            "index": index,
            "train": train_scores.tolist(),
            "test": test_scores.tolist(),
            "jump": max(jump_train, jump_test),
            "hard_events": int(sample.hard_event.sum()),
        }
        if phase == "nested":
            inner_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 99173]))
            deltas, jumps, errors = [], [], []
            for k in range(inner):
                try:
                    a, t, jump = bootstrap_iteration(sample, inner_rng)
                    deltas.append((a - t).tolist())
                    jumps.append(jump)
                except Exception as error:
                    errors.append({"inner": k + 1, "error": str(error)})
            result.update(inner_delta=deltas, inner_errors=errors, inner_max_jump=max(jumps) if jumps else None)
        result["seconds"] = time.time() - started
        return result
    except Exception as error:
        return {"index": index, "error": str(error), "trace": traceback.format_exc(), "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cohort", required=True, choices=["derivation", "validation"])
    parser.add_argument("--phase", default="point", choices=["point", "nested"])
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs = [
        (args.cohort, i, args.phase, args.inner)
        for i in range(args.count + 1)
        if not (args.out / f"{i:04d}.json").exists()
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init,
        initargs=(args.cohort, args.input_dir),
    ) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            (args.out / f"{record['index']:04d}.json").write_text(json.dumps(record))
            if n % 10 == 0 or "error" in record or n == len(jobs):
                print(
                    f"{args.cohort} {args.phase} {n}/{len(jobs)} elapsed {time.time() - started:.1f}s "
                    f"last_error={record.get('error')} last_runtime={record['seconds']:.2f}s",
                    flush=True,
                )
    if any("error" in json.loads(f.read_text()) for f in args.out.glob("*.json")):
        sys.exit(2)


if __name__ == "__main__":
    main()

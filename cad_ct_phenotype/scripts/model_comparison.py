"""Nested-model comparison of ridge Cox models A < B < C.

Reports the penalised partial-likelihood ratio statistic, a P value calibrated by model-based
null bootstrap (event times simulated under the reduced model with reverse-KM censoring and, for
hard MACE, competing soft events), and AIC using unpenalised partial likelihood at the penalised
estimates with effective degrees of freedom.
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
import csv
import json
import multiprocessing
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import beta, chi2

import competing_cox as engine

PENALTY = engine.PENALTY
SIZES = {"A": 8, "B": 13, "C": 15}
ENDPOINTS = [("composite", "composite_event"), ("hard", "hard_event")]
COMPARISONS = [("A", "B"), ("B", "C")]
CONTEXT: dict[str, dict] = {}


def fit(frame: pd.DataFrame, event: str, size: int) -> CoxPHFitter:
    return engine.fit_cox(frame, event, size, precision=1e-9, r_precision=1e-11)


def model_summary(frame: pd.DataFrame, event: str, model: CoxPHFitter, cohort: str, endpoint: str, name: str) -> dict:
    columns = list(model.params_.index)
    std = frame[columns].std(ddof=1).to_numpy()
    variance = model.variance_matrix_.to_numpy()
    penalty_matrix = len(frame) * PENALTY * np.diag(std**2)
    information = np.linalg.inv(variance) - penalty_matrix
    edf = float(np.trace(information @ variance))
    if not 0 < edf < len(columns):
        raise ValueError("Effective degrees of freedom outside (0, p)")
    log_likelihood = float(model.score(frame, scoring_method="log_likelihood") * len(frame))
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "model": name,
        "n": len(frame),
        "events": int(frame[event].sum()),
        "penalty": PENALTY,
        "nominal_df": len(columns),
        "effective_df": edf,
        "unpenalized_log_partial_likelihood": log_likelihood,
        "penalized_log_partial_likelihood": float(model.log_likelihood_),
        "effective_df_AIC": -2 * log_likelihood + 2 * edf,
    }


def null_specification(frame: pd.DataFrame, null_model: CoxPHFitter, soft_model: CoxPHFitter | None = None) -> dict:
    grid = null_model.baseline_hazard_.index.to_numpy(dtype=float)
    d1 = (
        null_model.predict_partial_hazard(frame).to_numpy()[:, None]
        * null_model.baseline_hazard_.iloc[:, 0].to_numpy()[None, :]
    )
    if soft_model is None:
        total, fraction = d1, None
    else:
        if not np.array_equal(grid, soft_model.baseline_hazard_.index.to_numpy()):
            raise ValueError("Cause-specific time grids differ")
        d2 = (
            soft_model.predict_partial_hazard(frame).to_numpy()[:, None]
            * soft_model.baseline_hazard_.iloc[:, 0].to_numpy()[None, :]
        )
        total = d1 + d2
        fraction = np.divide(d1, total, out=np.zeros_like(d1), where=total > 0)
    unique, g = engine.censor_km(frame.TIME.to_numpy(), frame.MACST.to_numpy() != 0)
    return {
        "grid": grid,
        "cum": np.cumsum(total, axis=1),
        "fraction": fraction,
        "censor_grid": unique,
        "censor_cdf": 1 - g,
        "tau": float(frame.TIME.max()),
    }


def init(input_dir: str | Path) -> None:
    for cohort in ["derivation", "validation"]:
        engine.init(cohort, input_dir)
        frame = engine.D.copy()
        models, summaries = {}, []
        for endpoint, event in ENDPOINTS:
            models[endpoint] = {name: fit(frame, event, size) for name, size in SIZES.items()}
            for name, model in models[endpoint].items():
                summaries.append(model_summary(frame, event, model, cohort, endpoint, name))
        soft = fit(frame, "soft_event", 15)
        nulls = {}
        for endpoint, _ in ENDPOINTS:
            for reduced, full in COMPARISONS:
                nulls[(endpoint, reduced, full)] = null_specification(
                    frame, models[endpoint][reduced], soft if endpoint == "hard" else None
                )
        CONTEXT[cohort] = {"frame": frame, "models": models, "nulls": nulls, "summaries": summaries}


def simulate(
    frame: pd.DataFrame, spec: dict, rng: np.random.Generator, endpoint: str
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(frame)
    threshold = rng.exponential(size=n)
    index = np.sum(spec["cum"] < threshold[:, None], axis=1)
    has_event = index < len(spec["grid"])
    event_time = np.r_[spec["grid"], np.inf][index]
    if endpoint == "hard":
        fraction = np.zeros(n)
        fraction[has_event] = spec["fraction"][np.arange(n)[has_event], index[has_event]]
        hard = has_event & (rng.random(n) < fraction)
        cause = np.where(hard, 1, np.where(has_event, 2, 0))
    else:
        cause = has_event.astype(int)
    censor_index = np.searchsorted(spec["censor_cdf"], rng.random(n), side="left")
    censor_time = np.minimum(np.r_[spec["censor_grid"], np.inf][censor_index], spec["tau"])
    observed = has_event & (event_time <= censor_time)
    status = np.where(observed, cause, 0)
    out = frame.copy()
    out["TIME"] = np.minimum(event_time, censor_time)
    out["sim_event"] = status != 0 if endpoint == "composite" else status == 1
    if not np.isfinite(out.TIME).all() or (out.TIME < 0).any():
        raise ValueError("Invalid simulated times")
    return out, status


def replicate(cohort: str, endpoint: str, reduced: str, full: str, index: int, seed: int) -> dict:
    context = CONTEXT[cohort]
    keys = [0 if cohort == "derivation" else 1, 0 if endpoint == "composite" else 1, 0 if reduced == "A" else 1]
    rng = np.random.default_rng(np.random.SeedSequence([seed] + keys + [index]))
    frame, status = simulate(context["frame"], context["nulls"][(endpoint, reduced, full)], rng, endpoint)
    lower = fit(frame, "sim_event", SIZES[reduced])
    upper = fit(frame, "sim_event", SIZES[full])
    value = 2 * (upper.log_likelihood_ - lower.log_likelihood_)
    if not np.isfinite(value) or value < -1e-6:
        raise ValueError(f"Invalid nested likelihood difference {value}")
    return {
        "index": index,
        "statistic": float(value),
        "events": int(frame.sim_event.sum()),
        "competing_events": int(np.sum(status == 2)),
        "censored": int(np.sum(status == 0)),
    }


def batch(args: tuple) -> dict:
    cohort, endpoint, reduced, full, start, count, seed = args
    started = time.time()
    rows = []
    for index in range(start, start + count):
        try:
            rows.append(replicate(cohort, endpoint, reduced, full, index, seed))
        except Exception as error:
            rows.append({"index": index, "error": repr(error), "trace": traceback.format_exc()})
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "reduced": reduced,
        "full": full,
        "start": start,
        "count": count,
        "rows": rows,
        "seconds": time.time() - started,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out: Path, repeats: int, input_dir: Path) -> None:
    init(input_dir)
    models, tests = [], []
    for cohort, context in CONTEXT.items():
        models += context["summaries"]
        for endpoint, _ in ENDPOINTS:
            aics = {r["model"]: r["effective_df_AIC"] for r in context["summaries"] if r["endpoint"] == endpoint}
            for reduced, full in COMPARISONS:
                folder = out / f"{cohort}_{endpoint}_{reduced}{full}"
                records = []
                for path in sorted(folder.glob("*.json")):
                    records += json.loads(path.read_text())["rows"]
                records.sort(key=lambda r: r["index"])
                if [r["index"] for r in records] != list(range(1, repeats + 1)):
                    raise RuntimeError(f"Missing or duplicate records in {folder}")
                if any("error" in r for r in records):
                    raise RuntimeError(f"Failed null-bootstrap replicates in {folder}")
                lower, upper = context["models"][endpoint][reduced], context["models"][endpoint][full]
                observed = 2 * (upper.log_likelihood_ - lower.log_likelihood_)
                values = np.array([r["statistic"] for r in records])
                exceed = int(np.sum(values >= observed))
                tests.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "comparison": f"{reduced}_vs_{full}",
                        "penalized_LR": float(observed),
                        "null_bootstrap_P": (exceed + 1) / (repeats + 1),
                        "exceedances": exceed,
                        "null_replicates": repeats,
                        "P_MC_lower": 0.0 if exceed == 0 else float(beta.ppf(0.025, exceed, repeats - exceed + 1)),
                        "P_MC_upper": (
                            1.0 if exceed == repeats else float(beta.ppf(0.975, exceed + 1, repeats - exceed))
                        ),
                        "nominal_chi_square_P": float(chi2.sf(observed, SIZES[full] - SIZES[reduced])),
                        "effective_AIC_difference": aics[full] - aics[reduced],
                        "mean_null_events": float(np.mean([r["events"] for r in records])),
                        "min_null_events": min(r["events"] for r in records),
                        "mean_null_competing_events": float(np.mean([r["competing_events"] for r in records])),
                    }
                )
    write_csv(out / "model_AIC_effective_df.csv", models)
    write_csv(out / "penalized_LR_bootstrap_P.csv", tests)
    print(json.dumps(tests, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4999)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.workers, args.batch) < 1:
        parser.error("Repeats, workers and batch must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.out, args.repeats, args.input_dir)
        return
    jobs = []
    for cohort in ["derivation", "validation"]:
        for endpoint, _ in ENDPOINTS:
            for reduced, full in COMPARISONS:
                folder = args.out / f"{cohort}_{endpoint}_{reduced}{full}"
                folder.mkdir(exist_ok=True)
                for start in range(1, args.repeats + 1, args.batch):
                    if not (folder / f"{start:05d}.json").exists():
                        jobs.append(
                            (
                                cohort,
                                endpoint,
                                reduced,
                                full,
                                start,
                                min(args.batch, args.repeats - start + 1),
                                args.seed,
                            )
                        )
    jobs.sort(key=lambda x: (x[4], x[0], x[1], x[2]))
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init,
        initargs=(args.input_dir,),
    ) as pool:
        futures = [pool.submit(batch, j) for j in jobs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            folder = args.out / f"{result['cohort']}_{result['endpoint']}_{result['reduced']}{result['full']}"
            (folder / f"{result['start']:05d}.json").write_text(json.dumps(result))
            errors = sum("error" in r for r in result["rows"])
            if n % 8 == 0 or errors or n == len(futures):
                print(f"{n}/{len(futures)} batches; elapsed {time.time() - started:.1f}s; errors {errors}", flush=True)
    summarize(args.out, args.repeats, args.input_dir)


if __name__ == "__main__":
    main()

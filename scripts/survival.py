"""Penalised Cox models A-C, apparent composite-MACE performance, phenotype hazard ratios and KM curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
from scipy.stats import norm

from common import (
    benjamini_hochberg,
    load_config,
    load_table,
    require_columns,
    semantic_numeric,
    validate_patient_table,
    validate_study_categories,
)
from graf_ipcw import graf_curve, integrate_curve

FORMULAS = {
    "A": "age + C(gender) + SBP + C(`Antihypertensive medication`) + TC + HDL + C(smoking) + C(DM)",
    "B": "age + C(gender) + SBP + C(`Antihypertensive medication`) + TC + HDL + C(smoking) + C(DM) + C(CADRADS)",
    "C": "age + C(gender) + SBP + C(`Antihypertensive medication`) + TC + HDL + C(smoking) + C(DM) + C(CADRADS) + C(Cluster)",
}
REQUIRED_COLUMNS = ["ID", "TIME", "MACET", "age", "gender", "SBP", "Antihypertensive medication", "TC", "HDL", "smoking", "DM", "CAD-RADS"]
NUMERIC_COLUMNS = ["TIME", "MACET", "age", "SBP", "TC", "HDL", "CAD-RADS", "Cluster"]


def load_analysis_data(cohort_path: Path, phenotype_path: Path) -> pd.DataFrame:
    cohort = load_table(cohort_path)
    phenotype = load_table(phenotype_path)
    validate_patient_table(cohort)
    validate_patient_table(phenotype)
    require_columns(cohort, REQUIRED_COLUMNS, "Cohort table")
    require_columns(phenotype, ["ID", "Cluster"], "Phenotype table")
    data = cohort[REQUIRED_COLUMNS].merge(phenotype[["ID", "Cluster"]], on="ID", how="left", validate="one_to_one")
    if data["Cluster"].isna().any():
        raise ValueError("Phenotype labels do not cover every cohort patient")
    for column in NUMERIC_COLUMNS:
        data[column] = semantic_numeric(data[column]).astype(float)
    if data[REQUIRED_COLUMNS[1:] + ["Cluster"]].isna().any().any():
        raise ValueError("Survival-analysis variables contain missing values")
    validate_study_categories(data)
    data["CADRADS"] = data["CAD-RADS"].astype(int)
    data["Cluster"] = data["Cluster"].astype(int)
    data["composite_event"] = data["MACET"].ne(0)
    data["hard_event"] = data["MACET"].eq(1)
    return data


def fit_models(data: pd.DataFrame, event_column: str, penalizer: float) -> dict[str, CoxPHFitter]:
    fitted = {}
    for name, formula in FORMULAS.items():
        fitted[name] = CoxPHFitter(penalizer=penalizer).fit(
            data, duration_col="TIME", event_col=event_column, formula=formula
        )
    return fitted


def proportional_hazards_diagnostics(
    data: pd.DataFrame, endpoint: str, fitted: dict[str, CoxPHFitter], alpha: float
) -> pd.DataFrame:
    tables = []
    for name, model in fitted.items():
        table = proportional_hazard_test(model, data, time_transform="rank").summary.reset_index()
        table = table.rename(columns={table.columns[0]: "term", "p": "p_value"})
        table.insert(0, "model", name)
        table.insert(0, "endpoint", endpoint)
        table["p_value_bh_within_model"] = benjamini_hochberg(table["p_value"])
        table["alpha"] = alpha
        table["flag_after_bh"] = table["p_value_bh_within_model"].lt(alpha)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def bootstrap_c_index(
    data: pd.DataFrame, event_column: str, risk: np.ndarray, repeats: int, seed: int
) -> tuple[float, float, int]:
    """Percentile interval from resampling patients together with their fitted risk scores."""
    generator = np.random.RandomState(seed)
    durations = data["TIME"].to_numpy(dtype=float)
    events = data[event_column].to_numpy(dtype=bool)
    risk = np.asarray(risk, dtype=float).ravel()
    if len(risk) != len(data):
        raise ValueError("Risk-score vector length does not match the data")
    values = []
    for _ in range(repeats):
        sample = generator.choice(len(data), len(data), replace=True)
        sampled_events = events[sample]
        if sampled_events.sum() < 2 or sampled_events.sum() == len(sampled_events):
            continue
        try:
            values.append(concordance_index(durations[sample], -risk[sample], sampled_events))
        except (ArithmeticError, ValueError):
            continue
    if not values:
        return np.nan, np.nan, 0
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def model_metrics(
    data: pd.DataFrame, endpoint: str, event_column: str, fitted: dict[str, CoxPHFitter], survival_config: dict
) -> pd.DataFrame:
    durations = data["TIME"].to_numpy(dtype=float)
    events = data[event_column].to_numpy(dtype=bool)
    horizons = [float(v) for v in survival_config["evaluation_months"]]
    grid_start = float(survival_config["ibs_grid_start_month"])
    grid_points = int(survival_config["ibs_grid_points"])
    repeats = int(survival_config["bootstrap_repeats"])
    seed = int(survival_config["random_state"])
    rows = []
    for name, model in fitted.items():
        risk = model.predict_partial_hazard(data).to_numpy().ravel()
        c_index = concordance_index(durations, -risk, events)
        ci_low, ci_high, successes = bootstrap_c_index(data, event_column, risk, repeats, seed)
        if repeats and successes < max(1, int(0.90 * repeats)):
            raise RuntimeError(
                f"Only {successes}/{repeats} C-index bootstrap replicates succeeded for {endpoint} Model {name}"
            )
        for horizon in horizons:
            if not grid_start < horizon < durations.max():
                raise ValueError(f"Evaluation horizon {horizon:g} is outside follow-up support")
            times = np.linspace(grid_start, horizon, grid_points)
            survival_probability = model.predict_survival_function(data, times=times).T.to_numpy()
            scores = graf_curve(durations, events, durations, events, survival_probability, times)
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": name,
                    "n": len(data),
                    "events": int(events.sum()),
                    "horizon_months": int(horizon),
                    "penalizer": float(survival_config["cox_penalizer"]),
                    "c_index": c_index,
                    "c_index_ci_low": ci_low,
                    "c_index_ci_high": ci_high,
                    "c_index_bootstrap_repeats": repeats,
                    "c_index_bootstrap_successful": successes,
                    "log_likelihood": model.log_likelihood_,
                    "n_parameters": len(model.params_),
                    "brier_ipcw": float(scores[-1]),
                    "ibs_ipcw": integrate_curve(scores, times),
                    "ibs_start_month": grid_start,
                    "ibs_end_month": horizon,
                    "ibs_grid_points": grid_points,
                }
            )
    return pd.DataFrame(rows)


def cluster_contrast(model: CoxPHFitter, numerator: int, denominator: int) -> dict[str, float]:
    """Wald contrast between two phenotype levels from the fitted coefficient covariance matrix."""
    names = list(model.params_.index)
    vector = np.zeros(len(names))
    for cluster, sign in [(numerator, 1.0), (denominator, -1.0)]:
        if cluster == 1:
            continue
        name = f"C(Cluster)[T.{cluster}]"
        if name not in names:
            raise ValueError(f"Missing fitted coefficient: {name}")
        vector[names.index(name)] += sign
    log_hr = float(vector @ model.params_.to_numpy())
    variance = float(vector @ model.variance_matrix_.to_numpy() @ vector)
    se = math.sqrt(max(variance, 0.0))
    z = log_hr / se if se > 0 else np.nan
    return {
        "hazard_ratio": math.exp(log_hr),
        "ci_low": math.exp(log_hr - 1.96 * se),
        "ci_high": math.exp(log_hr + 1.96 * se),
        "p_value": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


def cluster_contrasts(endpoint: str, model: CoxPHFitter) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"endpoint": endpoint, "contrast": f"Cluster_{a}_vs_{b}", **cluster_contrast(model, a, b)}
            for a, b in ((2, 1), (3, 1), (3, 2))
        ]
    )


def plot_kaplan_meier(data: pd.DataFrame, event_column: str, endpoint: str, destination: Path) -> dict:
    test = multivariate_logrank_test(data["TIME"], data["Cluster"], data[event_column])
    colors = {1: "#2C7BB6", 2: "#FDAE61", 3: "#D7191C"}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for cluster in sorted(data["Cluster"].unique()):
        selected = data["Cluster"].eq(cluster)
        KaplanMeierFitter(label=f"Cluster {cluster}").fit(
            data.loc[selected, "TIME"], data.loc[selected, event_column]
        ).plot_survival_function(ax=ax, ci_show=True, color=colors.get(cluster))
    ax.set(xlabel="Follow-up (months)", ylabel="Event-free survival")
    ax.text(0.98, 0.04, f"Log-rank P={test.p_value:.3g}", transform=ax.transAxes, ha="right", va="bottom")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    return {
        "endpoint": endpoint,
        "chisq": float(test.test_statistic),
        "df": int(test.degrees_of_freedom),
        "p_value": float(test.p_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--phenotypes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()

    survival_config = dict(load_config(args.config)["survival"])
    if args.bootstrap is not None:
        survival_config["bootstrap_repeats"] = args.bootstrap
    penalizer = float(survival_config["cox_penalizer"])
    data = load_analysis_data(args.cohort, args.phenotypes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics, contrasts, diagnostics, logrank = [], [], [], []
    for endpoint, event_column in (("composite_mace", "composite_event"), ("hard_mace", "hard_event")):
        fitted = fit_models(data, event_column, penalizer)
        diagnostics.append(
            proportional_hazards_diagnostics(data, endpoint, fitted, float(survival_config.get("ph_test_alpha", 0.05)))
        )
        if endpoint == "composite_mace":
            metrics.append(model_metrics(data, endpoint, event_column, fitted, survival_config))
        contrasts.append(cluster_contrasts(endpoint, fitted["C"]))
        fitted["C"].summary.to_csv(args.output_dir / f"model_c_{endpoint}_coefficients.csv")
        logrank.append(
            plot_kaplan_meier(data, event_column, endpoint, args.output_dir / f"kaplan_meier_{endpoint}.png")
        )

    pd.concat(metrics, ignore_index=True).to_csv(args.output_dir / "model_performance.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(args.output_dir / "cluster_pairwise_hazard_ratios.csv", index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(args.output_dir / "proportional_hazards_tests.csv", index=False)
    pd.DataFrame(logrank).to_csv(args.output_dir / "logrank_by_cluster.csv", index=False)
    settings = {
        "composite_mace": "MACET != 0",
        "hard_mace": "MACET == 1",
        "time_unit": "months",
        "cox_penalizer": penalizer,
        "model_formulas": FORMULAS,
        "evaluation_months": survival_config["evaluation_months"],
        "bootstrap_repeats": int(survival_config["bootstrap_repeats"]),
        "ibs_grid": [float(survival_config["ibs_grid_start_month"]), int(survival_config["ibs_grid_points"])],
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

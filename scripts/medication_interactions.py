"""Likelihood-ratio tests of phenotype-by-medication interactions in ridge Cox models."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2

from common import (
    benjamini_hochberg,
    load_config,
    load_table,
    require_columns,
    semantic_numeric,
    validate_patient_table,
)

INTERACTION_BASE = "age + C(gender) + SBP + TC + HDL + C(smoking) + C(DM) + C(CADRADS) + C(Cluster)"
ADJUSTMENT_SET = ("age", "gender", "SBP", "TC", "HDL", "smoking", "DM", "CAD-RADS", "Cluster")
ENDPOINTS = (("Composite_MACE", "composite_event"), ("Hard_MACE", "hard_event"))
BASE_COLUMNS = ["TIME", "MACST", "Cluster", "age", "gender", "SBP", "Anti2", "TC", "HDL", "smoking", "DM", "CAD-RADS"]


def interaction_formulas(medication: str) -> tuple[str, str]:
    reduced = f"{INTERACTION_BASE} + C({medication})"
    return reduced, f"{reduced} + C(Cluster):C({medication})"


def attach_phenotypes(cohort: pd.DataFrame, phenotypes_path: Path | None, name: str) -> pd.DataFrame:
    data = cohort.copy()
    if phenotypes_path is not None:
        phenotypes = load_table(phenotypes_path)
        validate_patient_table(phenotypes)
        require_columns(phenotypes, ["ID", "Cluster"], f"{name} phenotypes")
        data = data.drop(columns=["Cluster"], errors="ignore").merge(
            phenotypes[["ID", "Cluster"]], on="ID", how="left", validate="one_to_one"
        )
    require_columns(data, ["Cluster"], f"{name} cohort")
    if data["Cluster"].isna().any():
        raise ValueError(f"{name} phenotype labels do not cover all patients")
    return data


def prepare_analysis_data(frame: pd.DataFrame, medications: list[str], name: str) -> pd.DataFrame:
    required = ["ID", *BASE_COLUMNS, *medications]
    require_columns(frame, required, f"{name} cohort")
    data = frame.copy()
    for column in BASE_COLUMNS + medications:
        data[column] = semantic_numeric(data[column]).astype(float)
    if data[required].isna().any().any():
        missing = data[required].isna().sum()
        raise ValueError(f"{name} interaction inputs contain missing values: {missing[missing.gt(0)].to_dict()}")
    for medication in medications:
        levels = set(data[medication].astype(int).unique())
        if not levels.issubset({0, 1}) or len(levels) < 2:
            raise ValueError(f"{name} {medication} must be a nonconstant binary 0/1 variable")
    data["Cluster"] = data["Cluster"].astype(int)
    if not set(data["Cluster"].unique()).issubset({1, 2, 3}):
        raise ValueError(f"{name} Cluster must contain labels 1, 2 and 3")
    data["CADRADS"] = data["CAD-RADS"].astype(int)
    data["composite_event"] = data["MACST"].ne(0).astype(int)
    data["hard_event"] = data["MACST"].eq(1).astype(int)
    return data


def run_interaction_tests(
    data: pd.DataFrame, name: str, medications: list[str], penalizer: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests, coefficients = [], []
    for medication in medications:
        reduced_formula, full_formula = interaction_formulas(medication)
        for endpoint, event_column in ENDPOINTS:
            row = {
                "cohort": name,
                "medication": medication,
                "endpoint": endpoint,
                "n": len(data),
                "events": int(data[event_column].sum()),
                "medication_exposed_n": int(data[medication].eq(1).sum()),
                "penalizer": penalizer,
                "reduced_formula": reduced_formula,
                "full_formula": full_formula,
            }
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    reduced = CoxPHFitter(penalizer=penalizer).fit(
                        data, duration_col="TIME", event_col=event_column, formula=reduced_formula
                    )
                    full = CoxPHFitter(penalizer=penalizer).fit(
                        data, duration_col="TIME", event_col=event_column, formula=full_formula
                    )
                statistic = max(0.0, float(2 * (full.log_likelihood_ - reduced.log_likelihood_)))
                degrees = int(len(full.params_) - len(reduced.params_))
                row.update(
                    {
                        "reduced_log_likelihood": float(reduced.log_likelihood_),
                        "full_log_likelihood": float(full.log_likelihood_),
                        "likelihood_ratio_chisq": statistic,
                        "degrees_of_freedom": degrees,
                        "p_interaction": float(chi2.sf(statistic, degrees)),
                        "fit_warnings": " || ".join(str(item.message) for item in caught),
                        "status": "success",
                        "error": "",
                    }
                )
                terms = [
                    t for t in full.summary.index if ":" in str(t) and "Cluster" in str(t) and medication in str(t)
                ]
                if len(terms) != degrees:
                    raise RuntimeError(f"Expected {degrees} interaction coefficients but found {len(terms)}")
                for term in terms:
                    summary = full.summary.loc[term]
                    coefficients.append(
                        {
                            "cohort": name,
                            "medication": medication,
                            "endpoint": endpoint,
                            "term": term,
                            "coefficient": float(summary["coef"]),
                            "hazard_ratio": float(summary["exp(coef)"]),
                            "ci95_low": float(summary["exp(coef) lower 95%"]),
                            "ci95_high": float(summary["exp(coef) upper 95%"]),
                            "coefficient_p": float(summary["p"]),
                        }
                    )
            except Exception as error:
                row.update(
                    {
                        "reduced_log_likelihood": np.nan,
                        "full_log_likelihood": np.nan,
                        "likelihood_ratio_chisq": np.nan,
                        "degrees_of_freedom": np.nan,
                        "p_interaction": np.nan,
                        "fit_warnings": "",
                        "status": "failed",
                        "error": str(error),
                    }
                )
            tests.append(row)
    return pd.DataFrame(tests), pd.DataFrame(coefficients)


def interaction_table(tests: pd.DataFrame, medications: list[str]) -> pd.DataFrame:
    failures = tests.loc[tests["status"].ne("success")]
    if not failures.empty:
        raise RuntimeError(
            "Interaction models failed: "
            + failures[["cohort", "medication", "endpoint", "error"]].to_json(orient="records")
        )
    table = tests.pivot(index="medication", columns=["cohort", "endpoint"], values="p_interaction")
    columns = [(cohort, endpoint) for cohort in ("Derivation", "Validation") for endpoint, _ in ENDPOINTS]
    table = table.reindex(index=medications, columns=pd.MultiIndex.from_tuples(columns))
    table.columns = [f"{cohort}_{endpoint}_P_interaction" for cohort, endpoint in table.columns]
    return table.reset_index(names="Medication")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--derivation", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--validation-phenotypes", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    medications = list(config["survival"]["interaction_medications"])
    penalizer = float(config["survival"]["cox_penalizer"])
    derivation = load_table(args.derivation)
    validation = load_table(args.validation)
    validate_patient_table(derivation)
    validate_patient_table(validation)
    derivation = prepare_analysis_data(attach_phenotypes(derivation, None, "Derivation"), medications, "Derivation")
    validation = prepare_analysis_data(
        attach_phenotypes(validation, args.validation_phenotypes, "Validation"), medications, "Validation"
    )

    test_blocks, coefficient_blocks = [], []
    for name, data in (("Derivation", derivation), ("Validation", validation)):
        tests, coefficients = run_interaction_tests(data, name, medications, penalizer)
        test_blocks.append(tests)
        coefficient_blocks.append(coefficients)
    tests = pd.concat(test_blocks, ignore_index=True)
    coefficients = pd.concat(coefficient_blocks, ignore_index=True)
    tests["p_interaction_bh_within_cohort_endpoint"] = tests.groupby(["cohort", "endpoint"], group_keys=False)[
        "p_interaction"
    ].apply(benjamini_hochberg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.output_dir / "medication_interaction_tests_long.csv", index=False)
    coefficients.to_csv(args.output_dir / "medication_interaction_coefficients.csv", index=False)
    interaction_table(tests, medications).to_csv(args.output_dir / "medication_interaction_p_values.csv", index=False)
    settings = {
        "endpoints": {"Composite_MACE": "MACST != 0", "Hard_MACE": "MACST == 1"},
        "base_model": INTERACTION_BASE,
        "adjustment_set": list(ADJUSTMENT_SET),
        "medications": medications,
        "cox_penalizer": penalizer,
        "multiplicity": "Benjamini-Hochberg across medications within each cohort and endpoint",
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(tests[["cohort", "medication", "endpoint", "p_interaction", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

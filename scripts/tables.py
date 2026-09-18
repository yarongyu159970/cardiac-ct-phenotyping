"""Baseline, phenotype-characteristic and incremental-performance tables."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

from common import load_config, load_table, semantic_numeric

CONTINUOUS = [
    ("Age, years", "age", 1.0),
    ("BMI, kg/m2", "BMI", 1.0),
    ("SBP, mmHg", "SBP", 1.0),
    ("Fasting glucose, mmol/L", "FG", 1.0),
    ("HbA1c, %", "HbA1c", 1.0),
    ("LDL, mmol/L", "LDL", 1.0),
    ("HDL, mmol/L", "HDL", 1.0),
    ("TG, mmol/L", "TG", 1.0),
    ("TC, mmol/L", "TC", 1.0),
    ("Total plaque volume, mm3", "Total plaque volume", 1.0),
    ("Calcified plaque volume, mm3", "calcified volume", 1.0),
    ("Low-attenuation plaque volume, mm3", "low attenuation volume", 1.0),
    ("Fibrous plaque volume, mm3", "fibrotic volume", 1.0),
    ("Fibro-fatty plaque volume, mm3", "fibrous fatty volume", 1.0),
    ("CT-FFR_LAD", "CT-FFR-LAD", 1.0),
    ("CT-FFR_LCx", "CT-FFR-LcX", 1.0),
    ("CT-FFR_RCA", "CT-FFR-RCA", 1.0),
    ("Global MBF, mL/100 mL/min", "Global", 1.0),
    ("IMV, %", "IMV", 100.0),
    ("LAD_MBF, mL/100 mL/min", "LAD_MBF", 1.0),
    ("LCx_MBF, mL/100 mL/min", "LCx_MBF", 1.0),
    ("RCA_MBF, mL/100 mL/min", "RCA_MBF", 1.0),
    ("LAD_MBV, mL/100mL", "LAD_MBV", 1.0),
    ("LCx_MBV, mL/100mL", "LCx_MBV", 1.0),
    ("RCA_MBV, mL/100mL", "RCA_MBV", 1.0),
    ("LAD_TTP, s", "LAD_TTP", 1.0),
    ("LCx_TTP, s", "LCx_TTP", 1.0),
    ("RCA_TTP, s", "RCA_TTP", 1.0),
    ("LAD_PCBV, mL/100mL", "LAD_PCBV", 1.0),
    ("LCx_PCBV, mL/100mL", "LCx_PCBV", 1.0),
    ("RCA_PCBV, mL/100mL", "RCA_PCBV", 1.0),
    ("LAD_FE, mL/100mL/min", "LAD_FE", 1.0),
    ("LCx_FE, mL/100mL/min", "LCx_FE", 1.0),
    ("RCA_FE, mL/100mL/min", "RCA_FE", 1.0),
]
BINARY = [
    ("Male, n (%)", "gender", 1),
    ("Diabetes mellitus, n (%)", "DM", 1),
    ("Hypertension, n (%)", "HTN", 1),
    ("Dyslipidemia, n (%)", "dislipidemia", 1),
    ("Current smoking, n (%)", "smoking", 1),
    ("HRP, n (%)", "HRP", 1),
    ("LAP, n (%)", "LAP", 1),
    ("PR, n (%)", "PR", 1),
    ("NRS, n (%)", "NRS", 1),
    ("SC, n (%)", "SC", 1),
    ("Antiplatelet therapy, n (%)", "Antiplatelet therapy", 1),
    ("Antihypertensive medication, n (%)", "Antihypertensive medication", 1),
    ("Antidiabetic medication, n (%)", "Antidiabetic medication", 1),
    ("Anti-ischemic therapy, n (%)", "Antiischemic medication", 1),
    ("Nitrates, n (%)", "Nitrates", 1),
    ("Statin therapy, n (%)", "Statin", 1),
]
MULTICATEGORY = [
    ("CCS category", "CCS", [(1, "I"), (2, "II"), (3, "III")]),
    ("CACS", "CACS", [(1, "Score 0"), (2, "Score 1-100"), (3, "Score 101-400"), (4, "Score >400")]),
    ("CAD-RADS category", "CAD-RADS", [(value, str(value)) for value in range(6)]),
]
TABLE_ORDER = [
    "age",
    "gender",
    "BMI",
    "DM",
    "HTN",
    "SBP",
    "dislipidemia",
    "smoking",
    "CCS",
    "FG",
    "HbA1c",
    "LDL",
    "HDL",
    "TG",
    "TC",
    "CACS",
    "CAD-RADS",
    "HRP",
    "LAP",
    "PR",
    "NRS",
    "SC",
    "Total plaque volume",
    "calcified volume",
    "low attenuation volume",
    "fibrotic volume",
    "fibrous fatty volume",
    "CT-FFR-LAD",
    "CT-FFR-LcX",
    "CT-FFR-RCA",
    "Global",
    "IMV",
    "LAD_MBF",
    "LCx_MBF",
    "RCA_MBF",
    "LAD_MBV",
    "LCx_MBV",
    "RCA_MBV",
    "LAD_TTP",
    "LCx_TTP",
    "RCA_TTP",
    "LAD_PCBV",
    "LCx_PCBV",
    "RCA_PCBV",
    "LAD_FE",
    "LCx_FE",
    "RCA_FE",
    "Antiplatelet therapy",
    "Antihypertensive medication",
    "Antidiabetic medication",
    "Antiischemic medication",
    "Nitrates",
    "Statin",
]


def p_display(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def add_endpoint_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if "MACET" not in frame.columns:
        raise ValueError("Cohort table is missing MACET")
    output = frame.copy()
    code = semantic_numeric(output["MACET"])
    if code.isna().any() or not set(code.astype(int).unique()).issubset({0, 1, 2}):
        raise ValueError("MACET must be complete and coded 0, 1 or 2")
    output["Composite_MACE"] = code.ne(0).astype(int)
    output["Hard_MACE"] = code.eq(1).astype(int)
    return output


def validate_categories(frame: pd.DataFrame, label_column: str, clusters: tuple[int, ...]) -> None:
    if label_column not in frame.columns:
        raise ValueError(f"Label column is missing: {label_column}")
    observed = set(semantic_numeric(frame[label_column]).dropna().astype(int))
    if observed != set(clusters):
        raise ValueError(f"{label_column} must contain exactly labels {sorted(clusters)}; observed {sorted(observed)}")
    for _, column, levels in MULTICATEGORY:
        values = semantic_numeric(frame[column])
        if values.isna().any():
            raise ValueError(f"{column} contains missing or nonnumeric values")
        unknown = set(values.astype(int)) - {int(level) for level, _ in levels}
        if unknown:
            raise ValueError(f"{column} contains levels absent from the table specification: {sorted(unknown)}")


def median_iqr(values: pd.Series, digits: int = 2) -> str:
    values = semantic_numeric(values)
    return f"{values.median():.{digits}f} ({values.quantile(.25):.{digits}f}, {values.quantile(.75):.{digits}f})"


def count_percent(values: pd.Series, level: int) -> str:
    count = int(semantic_numeric(values).eq(level).sum())
    return f"{count} ({100 * count / len(values):.1f})"


def adjust_pairwise_p_values(values: list[float], method: str) -> list[float]:
    if method == "bonferroni":
        return [min(1.0, len(values) * value) for value in values]
    if method != "holm":
        raise ValueError(f"Unsupported pairwise correction: {method}")
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    adjusted_sorted = np.maximum.accumulate([(len(values) - rank) * value for rank, value in enumerate(sorted_values)])
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return adjusted.tolist()


def continuous_tests(
    frame: pd.DataFrame, column: str, correction: str, label_column: str = "Cluster", clusters=(1, 2, 3)
):
    values = semantic_numeric(frame[column])
    groups = [values.loc[frame[label_column].eq(cluster)].to_numpy() for cluster in clusters]
    overall = float(kruskal(*groups).pvalue)
    raw = [
        float(
            mannwhitneyu(
                values.loc[frame[label_column].eq(left)],
                values.loc[frame[label_column].eq(right)],
                alternative="two-sided",
                method="asymptotic",
                use_continuity=True,
            ).pvalue
        )
        for left, right in combinations(clusters, 2)
    ]
    return overall, adjust_pairwise_p_values(raw, correction)


def categorical_tests(
    frame: pd.DataFrame, column: str, binary: bool, correction: str, label_column: str = "Cluster", clusters=(1, 2, 3)
):
    table = pd.crosstab(frame[column], frame[label_column])
    overall = float(chi2_contingency(table, correction=False).pvalue)
    raw = []
    for left, right in combinations(clusters, 2):
        selected = frame[label_column].isin([left, right])
        pair = pd.crosstab(frame.loc[selected, column], frame.loc[selected, label_column])
        if binary and pair.shape == (2, 2):
            raw.append(float(fisher_exact(pair.to_numpy()).pvalue))
        else:
            raw.append(float(chi2_contingency(pair, correction=False).pvalue))
    return overall, adjust_pairwise_p_values(raw, correction)


def add_p_values(row: dict, overall: float, pairwise: list[float], clusters=(1, 2, 3)) -> None:
    row["P_overall"] = p_display(overall)
    for (left, right), value in zip(combinations(clusters, 2), pairwise):
        row[f"P_{left}_vs_{right}"] = p_display(value)


def cluster_table(
    frame: pd.DataFrame,
    pairwise_correction: str,
    label_column: str = "Cluster",
    clusters: tuple[int, ...] = (1, 2, 3),
    include_outcomes: bool = True,
) -> pd.DataFrame:
    """Characteristics by phenotype with overall and pairwise tests."""
    validate_categories(frame, label_column, clusters)
    rows = []
    if include_outcomes:
        for label, event in (("Patients with MACE, n", "Composite_MACE"), ("Patients with hard MACE, n", "Hard_MACE")):
            if event not in frame.columns:
                raise ValueError(f"Outcome row requested but {event} is missing")
            row = {"Variable": label}
            for cluster in clusters:
                row[f"Cluster_{cluster}"] = int(frame.loc[frame[label_column].eq(cluster), event].sum())
            rows.append(row)
    continuous = {column: (label, scale) for label, column, scale in CONTINUOUS}
    binary = {column: (label, level) for label, column, level in BINARY}
    multi = {column: (label, levels) for label, column, levels in MULTICATEGORY}
    for column in TABLE_ORDER:
        if column in continuous:
            label, scale = continuous[column]
            row = {"Variable": label}
            for cluster in clusters:
                row[f"Cluster_{cluster}"] = median_iqr(
                    semantic_numeric(frame.loc[frame[label_column].eq(cluster), column]) * scale
                )
            add_p_values(row, *continuous_tests(frame, column, pairwise_correction, label_column, clusters), clusters)
            rows.append(row)
        elif column in binary:
            label, level = binary[column]
            row = {"Variable": label}
            for cluster in clusters:
                row[f"Cluster_{cluster}"] = count_percent(frame.loc[frame[label_column].eq(cluster), column], level)
            add_p_values(
                row, *categorical_tests(frame, column, True, pairwise_correction, label_column, clusters), clusters
            )
            rows.append(row)
        else:
            label, levels = multi[column]
            parent = {"Variable": label}
            add_p_values(
                parent, *categorical_tests(frame, column, False, pairwise_correction, label_column, clusters), clusters
            )
            rows.append(parent)
            for level, level_label in levels:
                child = {"Variable": f"  {level_label}, n (%)"}
                for cluster in clusters:
                    child[f"Cluster_{cluster}"] = count_percent(
                        frame.loc[frame[label_column].eq(cluster), column], level
                    )
                rows.append(child)
    return pd.DataFrame(rows)


def baseline_table(derivation: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    total = pd.concat([derivation, validation], ignore_index=True, sort=False)
    cohorts = [("Derivation", derivation), ("Validation", validation), ("Total", total)]
    rows = []
    specification = [
        ("continuous", "Age, years", "age", 1.0, 2),
        ("binary", "Male, n (%)", "gender", 1, None),
        ("continuous", "BMI, kg/m2", "BMI", 1.0, 2),
        ("binary", "Diabetes mellitus, n (%)", "DM", 1, None),
        ("binary", "Hypertension, n (%)", "HTN", 1, None),
        ("continuous", "SBP, mmHg", "SBP", 1.0, 2),
        ("binary", "Dyslipidemia, n (%)", "dislipidemia", 1, None),
        ("binary", "Current smoking, n (%)", "smoking", 1, None),
    ]
    for kind, label, column, parameter, digits in specification:
        row = {"Variable": label}
        for name, data in cohorts:
            if kind == "continuous":
                row[name] = median_iqr(semantic_numeric(data[column]) * float(parameter), int(digits))
            else:
                row[name] = count_percent(data[column], int(parameter))
        rows.append(row)
    rows.append({"Variable": "CCS category"})
    for level, label in ((1, "I"), (2, "II"), (3, "III")):
        row = {"Variable": f"  {label}, n (%)"}
        for name, data in cohorts:
            row[name] = count_percent(data["CCS"], level)
        rows.append(row)
    for label, column, scale, digits in [
        ("Fasting glucose, mmol/L", "FG", 1.0, 2),
        ("HbA1c, %", "HbA1c", 1.0, 2),
        ("LDL, mmol/L", "LDL", 1.0, 2),
        ("HDL, mmol/L", "HDL", 1.0, 2),
        ("TC, mmol/L", "TC", 1.0, 2),
        ("TG, mmol/L", "TG", 1.0, 2),
        ("Follow-up, months", "TIME", 1.0, 1),
    ]:
        row = {"Variable": label}
        for name, data in cohorts:
            row[name] = median_iqr(semantic_numeric(data[column]) * scale, digits)
        rows.append(row)
    return pd.DataFrame(rows)


def merge_k2_labels(derivation: pd.DataFrame, k2_labels_path: Path) -> pd.DataFrame:
    labels = load_table(k2_labels_path)
    missing = sorted({"ID", "K2_Cluster"}.difference(labels.columns))
    if missing:
        raise ValueError(f"K=2 label table is missing columns: {missing}")
    labels = labels[["ID", "K2_Cluster"]]
    if labels["ID"].isna().any() or labels["ID"].duplicated().any():
        raise ValueError("K=2 label IDs must be complete and unique")
    if set(derivation["ID"]) != set(labels["ID"]):
        raise ValueError("K=2 labels must cover the derivation cohort exactly")
    output = derivation.merge(labels, on="ID", how="left", validate="one_to_one")
    values = semantic_numeric(output["K2_Cluster"])
    if values.isna().any():
        raise ValueError("K2_Cluster contains missing or nonnumeric values")
    output["K2_Cluster"] = values.astype(int)
    return output


def pairwise_corrections(config: dict) -> dict[str, str]:
    settings = config.get("table_statistics", {})
    values = {cohort: settings.get(f"{cohort}_pairwise_correction", "holm") for cohort in ["derivation", "validation"]}
    if any(value not in ["holm", "bonferroni"] for value in values.values()):
        raise ValueError("Table pairwise correction must be holm or bonferroni")
    return values


def performance_table(run: Path, cohort: str, endpoint: str) -> pd.DataFrame:
    """Compact incremental-value table: C-index (95% CI), LR, P, AIC, Brier and IBS for Models A-C."""
    hard = endpoint == "hard_mace"
    key = "hard" if hard else "composite"
    models = pd.read_csv(run / "model_comparison/model_AIC_effective_df.csv")
    models = models[models.cohort.eq(cohort) & models.endpoint.eq(key)].set_index("model")
    tests = pd.read_csv(run / "model_comparison/penalized_LR_bootstrap_P.csv")
    tests = tests[tests.cohort.eq(cohort) & tests.endpoint.eq(key)].set_index("comparison")
    if hard:
        values = pd.read_csv(run / "hard_competing/intervals/corrected_table_long.csv")
        values = values[values.cohort.eq(cohort)].set_index(["comparison", "metric"])
    else:
        values = pd.read_csv(run / f"{cohort}_survival/model_performance.csv")
        values = values[values.endpoint.eq(endpoint)].set_index(["model", "horizon_months"])
    comparisons = [("A vs B", "A_vs_B", "A", "B"), ("B vs C", "B_vs_C", "B", "C")]
    rows = []

    def add_row(name: str, estimates: dict | None = None, contrasts: dict | None = None) -> None:
        row = {"Metric": name, **{f"Model {m}": "—" for m in "ABC"}, "A vs B": "—", "B vs C": "—"}
        if estimates:
            row.update({f"Model {m}": v for m, v in estimates.items()})
        if contrasts:
            row.update(contrasts)
        rows.append(row)

    ci = {}
    for m in "ABC":
        if hard:
            x = values.loc[(m, "C-index")]
            estimate, low, high = x["estimate"], x["lower"], x["upper"]
        else:
            x = values.loc[(m, 36)]
            estimate, low, high = x["c_index"], x["c_index_ci_low"], x["c_index_ci_high"]
        ci[m] = f"{estimate:.3f} ({low:.3f}–{high:.3f})"
    add_row("C-index (95% CI)", ci)
    add_row(
        "Likelihood-ratio test value",
        contrasts={name: f"{tests.loc[key_, 'penalized_LR']:.2f}" for name, key_, _, _ in comparisons},
    )
    add_row(
        "P value", contrasts={name: p_display(tests.loc[key_, "null_bootstrap_P"]) for name, key_, _, _ in comparisons}
    )
    add_row(
        "AIC",
        {m: f"{models.loc[m, 'effective_df_AIC']:.2f}" for m in "ABC"},
        {
            name: f"{models.loc[b, 'effective_df_AIC'] - models.loc[a, 'effective_df_AIC']:.2f}"
            for name, _, a, b in comparisons
        },
    )
    for horizon in [36, 60]:
        for metric, label in [("brier", "Brier score"), ("ibs", "IBS")]:
            estimates = {}
            for m in "ABC":
                if hard:
                    value = values.loc[(m, ("Brier" if metric == "brier" else "IBS") + str(horizon)), "estimate"]
                else:
                    value = values.loc[(m, horizon), metric + "_ipcw"]
                estimates[m] = f"{value:.3f}"
            add_row(f"{label} ({horizon}-month horizon)", estimates)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    args = parser.parse_args()
    corrections = pairwise_corrections(load_config(args.config))
    root = args.run_dir
    out = root / "tables"
    out.mkdir(exist_ok=True)
    derivation = add_endpoint_indicators(pd.read_csv(root / "analysis_inputs/derivation.csv"))
    validation = add_endpoint_indicators(pd.read_csv(root / "analysis_inputs/validation_projected.csv"))
    for path in [root / "oof_summary/oof_repeat_mean.csv", root / "projection_refit/oof_summary/oof_repeat_mean.csv"]:
        if path.exists():
            pd.read_csv(path).to_csv(out / "OOF_repeat_mean.csv", index=False)
            break
    baseline_table(derivation, validation).to_csv(out / "Baseline_derivation_vs_validation.csv", index=False)
    cluster_table(derivation, corrections["derivation"]).to_csv(out / "Derivation_K3_characteristics.csv", index=False)
    cluster_table(validation, corrections["validation"]).to_csv(
        out / "Validation_projected_characteristics.csv", index=False
    )
    for cohort in ["derivation", "validation"]:
        if (root / f"{cohort}_survival/model_performance.csv").exists():
            for endpoint in ["composite_mace", "hard_mace"]:
                performance_table(root, cohort, endpoint).to_csv(
                    out / f"{cohort}_{endpoint}_incremental.csv", index=False
                )
    k2 = root / "clustering/labels_k2.csv"
    if k2.exists():
        with_k2 = merge_k2_labels(derivation, k2)
        cluster_table(with_k2, "holm", label_column="K2_Cluster", clusters=(1, 2), include_outcomes=False).to_csv(
            out / "Derivation_K2_characteristics.csv", index=False
        )
        pd.crosstab(with_k2.Cluster, with_k2.K2_Cluster).to_csv(out / "K3_to_K2_cross_tabulation.csv")


if __name__ == "__main__":
    main()

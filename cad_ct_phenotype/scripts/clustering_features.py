from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    OUTCOME_OR_LABEL_COLUMNS,
    SEGMENT_COLUMNS,
    SEGMENTS,
    TERRITORY_COLUMNS,
    require_columns,
    semantic_numeric,
)


def classify_features(frame: pd.DataFrame, categorical_max_unique: int) -> tuple[list[str], list[str], pd.DataFrame]:
    """Split candidate columns into categorical and continuous by observed cardinality."""
    excluded = OUTCOME_OR_LABEL_COLUMNS | {"ID"} | SEGMENT_COLUMNS | TERRITORY_COLUMNS
    candidates = [column for column in frame.columns if column not in excluded]
    categorical: list[str] = []
    continuous: list[str] = []
    rows = []
    for column in candidates:
        numeric = semantic_numeric(frame[column])
        if int(numeric.notna().sum()) != int(frame[column].notna().sum()):
            raise ValueError(f"{column} cannot be interpreted as numeric")
        n_unique = int(numeric.nunique(dropna=True))
        kind = "categorical" if n_unique <= categorical_max_unique else "continuous"
        (categorical if kind == "categorical" else continuous).append(column)
        rows.append(
            {
                "feature": column,
                "n_unique": n_unique,
                "inferred_type": kind,
                "levels": (
                    "|".join(map(str, sorted(numeric.dropna().unique().tolist()))) if kind == "categorical" else ""
                ),
            }
        )
    return categorical, continuous, pd.DataFrame(rows)


def pca_scores(
    frame: pd.DataFrame, metrics: Iterable[str], variance_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Metric-wise PCA over the 17 segments, retaining components up to the variance threshold."""
    blocks = []
    report = []
    component_variance: dict[str, float] = {}
    for metric in metrics:
        columns = [f"{segment}_{metric}" for segment in SEGMENTS]
        require_columns(frame, columns, f"{metric} PCA input")
        numeric = pd.DataFrame({c: semantic_numeric(frame[c]) for c in columns}, index=frame.index)
        if numeric.isna().any().any():
            raise ValueError(f"{metric} PCA inputs contain missing or nonnumeric values")
        scaled = StandardScaler().fit_transform(numeric)
        cumulative = np.cumsum(PCA().fit(scaled).explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        fitted = PCA(n_components=n_components).fit(scaled)
        names = [f"{metric}_PC{i + 1}" for i in range(n_components)]
        blocks.append(pd.DataFrame(fitted.transform(scaled), columns=names, index=frame.index))
        for name, ratio in zip(names, fitted.explained_variance_ratio_):
            component_variance[name] = float(ratio)
        report.append(
            {
                "metric": metric,
                "n_components": n_components,
                "cumulative_variance": float(fitted.explained_variance_ratio_.sum()),
            }
        )
    return pd.concat(blocks, axis=1), pd.DataFrame(report), component_variance


def correlation_filter(
    numeric: pd.DataFrame, component_variance: dict[str, float], threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop one member of each pair with |r| above the threshold, preferring PCA scores over raw fields."""
    correlation = numeric.corr(method="pearson").abs()
    mean_abs = ((correlation.sum(axis=1) - 1.0) / max(len(correlation) - 1, 1)).to_dict()
    columns = numeric.columns.tolist()
    pairs = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(correlation.loc[left, right])
            if value > threshold:
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "abs_pearson_r": value,
                        "feature_1_is_pca": left in component_variance,
                        "feature_2_is_pca": right in component_variance,
                    }
                )
    pair_table = pd.DataFrame(pairs)
    if not pair_table.empty:
        pair_table = pair_table.sort_values(
            ["abs_pearson_r", "feature_1", "feature_2"], ascending=[False, True, True]
        ).reset_index(drop=True)
    dropped: set[str] = set()
    decisions = []
    for row in pair_table.to_dict("records") if not pair_table.empty else []:
        left, right = str(row["feature_1"]), str(row["feature_2"])
        if left in dropped or right in dropped:
            continue
        left_pca, right_pca = left in component_variance, right in component_variance
        if left_pca != right_pca:
            keep, drop = (left, right) if left_pca else (right, left)
            reason = "prefer_pca_over_raw"
        elif left_pca:
            lv, rv = component_variance[left], component_variance[right]
            keep, drop = sorted([left, right]) if lv == rv else ((left, right) if lv > rv else (right, left))
            reason = "both_pca_keep_higher_component_variance"
        else:
            lm, rm = mean_abs[left], mean_abs[right]
            keep, drop = sorted([left, right]) if lm == rm else ((left, right) if lm < rm else (right, left))
            reason = "both_raw_drop_higher_mean_abs_correlation"
        dropped.add(drop)
        decisions.append(
            {
                "kept_feature": keep,
                "dropped_feature": drop,
                "trigger_abs_pearson_r": row["abs_pearson_r"],
                "reason": reason,
            }
        )
    return pair_table, pd.DataFrame(decisions), sorted(dropped)


def eta_squared_filter(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    numeric: pd.DataFrame,
    threshold: float,
    whitelist: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Remove categorical fields that are nearly determined by a numeric feature (eta squared above threshold)."""
    rows = []
    for category in categorical_columns:
        labels = semantic_numeric(frame[category])
        for continuous in numeric.columns:
            values = numeric[continuous]
            valid = labels.notna() & values.notna()
            pair = pd.DataFrame({"category": labels.loc[valid], "value": values.loc[valid]})
            total_ss = float(np.square(pair["value"] - pair["value"].mean()).sum())
            if pair["category"].nunique() < 2 or total_ss <= 0:
                eta_squared = np.nan
            else:
                grand_mean = float(pair["value"].mean())
                between_ss = float(
                    sum(
                        len(group) * (float(group.mean()) - grand_mean) ** 2
                        for _, group in pair.groupby("category", observed=True)["value"]
                    )
                )
                eta_squared = between_ss / total_ss
            rows.append(
                {
                    "categorical_feature": category,
                    "continuous_feature": continuous,
                    "eta_squared": eta_squared,
                    "threshold": threshold,
                    "above_threshold": bool(np.isfinite(eta_squared) and eta_squared > threshold),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("categorical_feature", as_index=False)
        .agg(
            n_continuous_tested=("continuous_feature", "size"),
            n_eta_squared_above_threshold=("above_threshold", "sum"),
            maximum_eta_squared=("eta_squared", "max"),
        )
        .sort_values(["n_eta_squared_above_threshold", "maximum_eta_squared"], ascending=[False, False])
        .reset_index(drop=True)
    )
    flagged = summary.loc[summary["n_eta_squared_above_threshold"].ge(1), "categorical_feature"].tolist()
    protected = set(whitelist)
    unknown = sorted(protected.difference(categorical_columns))
    if unknown:
        raise ValueError(f"Eta-squared whitelist variables are not categorical: {unknown}")
    removed = [column for column in flagged if column not in protected]
    retained = [column for column in categorical_columns if column not in removed]
    summary["flagged_for_redundancy"] = summary["categorical_feature"].isin(flagged)
    summary["whitelisted"] = summary["categorical_feature"].isin(protected)
    summary["retained"] = summary["categorical_feature"].isin(retained)
    return detail, summary, retained, removed


def build_clustering_matrix(
    frame: pd.DataFrame, preprocessing: dict
) -> tuple[np.ndarray, list[int], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the mixed numeric/categorical matrix used by K-prototypes."""
    categorical, continuous, type_table = classify_features(frame, int(preprocessing["categorical_max_unique"]))
    pca, pca_report, component_variance = pca_scores(
        frame, preprocessing["pca_metrics"], float(preprocessing["pca_variance_threshold"])
    )
    raw_numeric = pd.DataFrame({c: semantic_numeric(frame[c]) for c in continuous}, index=frame.index)
    if raw_numeric.isna().any().any():
        raise ValueError("Continuous clustering variables contain missing values")
    numeric_before = pd.concat([raw_numeric, pca], axis=1)
    correlation_pairs, correlation_decisions, dropped_numeric = correlation_filter(
        numeric_before, component_variance, float(preprocessing["correlation_threshold"])
    )
    numeric = numeric_before.drop(columns=dropped_numeric)
    eta_detail, eta_summary, retained_categorical, removed_categorical = eta_squared_filter(
        frame,
        categorical,
        numeric,
        float(preprocessing["eta_squared_threshold"]),
        preprocessing.get("eta_squared_whitelist", []),
    )
    numeric_scaled = pd.DataFrame(StandardScaler().fit_transform(numeric), columns=numeric.columns, index=frame.index)
    categorical_frame = frame[retained_categorical].copy()
    if categorical_frame.isna().any().any():
        raise ValueError("Categorical clustering variables contain missing values")
    categorical_frame = categorical_frame.astype("string").astype(str)
    mixed = np.column_stack([numeric_scaled.to_numpy(dtype=float), categorical_frame.to_numpy(dtype=object)])
    categorical_indices = list(range(numeric_scaled.shape[1], mixed.shape[1]))
    diagnostics = {
        "feature_types": type_table,
        "pca_report": pca_report,
        "pca_scores": pca,
        "correlation_pairs": correlation_pairs,
        "correlation_decisions": correlation_decisions,
        "eta_squared_detail": eta_detail,
        "eta_squared_summary": eta_summary,
        "dropped_numeric": dropped_numeric,
        "removed_categorical": removed_categorical,
        "all_categorical": categorical,
        "retained_categorical": retained_categorical,
    }
    return mixed, categorical_indices, numeric_scaled, categorical_frame, diagnostics

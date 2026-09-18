from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    OUTCOME_OR_LABEL_COLUMNS,
    PERFUSION_METRICS,
    SEGMENT_COLUMNS,
    TERRITORIES,
    TERRITORY_COLUMNS,
    assert_outcome_independent,
    require_columns,
    semantic_numeric,
    territory_mean,
)


def candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric candidate matrix: direct clinical/plaque fields plus territory-level perfusion means."""
    excluded = OUTCOME_OR_LABEL_COLUMNS | {"ID", "Cohort"} | SEGMENT_COLUMNS | TERRITORY_COLUMNS
    direct_columns = [column for column in frame.columns if column not in excluded]
    assert_outcome_independent(direct_columns)
    output = pd.DataFrame(index=frame.index)
    for column in direct_columns:
        output[column] = semantic_numeric(frame[column])
    for territory in TERRITORIES:
        for metric in PERFUSION_METRICS:
            output[f"{territory}_{metric}"] = territory_mean(frame, metric, territory)
    output = output.loc[:, output.nunique(dropna=False).gt(1)]
    missing = output.isna().sum()
    if missing.any():
        raise ValueError(f"Candidate features contain missing values: {missing[missing.gt(0)].to_dict()}")
    return output


def term_mapping(original: pd.DataFrame, categorical: list[str], drop_first: bool = False) -> dict[str, str]:
    """Map every dummy-encoded term back to its source variable."""
    if not original.columns.is_unique or len(categorical) != len(set(categorical)):
        raise ValueError("Source and categorical feature names must be unique")
    if not set(categorical).issubset(original.columns):
        raise ValueError("Categorical feature is absent from the source matrix")
    mapping: dict[str, str] = {}
    for source in [c for c in original.columns if c not in categorical] + categorical:
        if source in categorical:
            terms = pd.get_dummies(
                original[[source]], columns=[source], drop_first=drop_first, dtype=float, prefix_sep="="
            ).columns.tolist()
        else:
            terms = [source]
        if not terms:
            raise ValueError(f"Source variable has no encoded terms: {source}")
        for term in terms:
            if term in mapping:
                raise ValueError(f"Ambiguous encoded term: {term}")
            mapping[term] = source
    return mapping


def dummy_encode_candidates(
    frame: pd.DataFrame, categorical_max_unique: int, drop_first: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    original = candidate_features(frame)
    categorical = [c for c in original.columns if int(original[c].nunique(dropna=True)) <= categorical_max_unique]
    encoded = pd.get_dummies(original, columns=categorical, drop_first=drop_first, dtype=float, prefix_sep="=")
    if encoded.isna().any().any():
        raise ValueError("Dummy-encoded candidate matrix contains missing values")
    mapping = term_mapping(original, categorical, drop_first)
    if not encoded.columns.is_unique or set(encoded.columns) != set(mapping):
        raise ValueError("Encoded candidate columns do not have a unique source mapping")
    return encoded, original, categorical


def source_consensus_ranking(term_table: pd.DataFrame, term_to_source: dict[str, str]) -> pd.DataFrame:
    """Collapse term-level ranks to one row per source variable (lowest rank sum, term-name tie break)."""
    require_columns(term_table, ["term", "consensus_rank_sum"], "Term ranking")
    if not term_table["term"].is_unique:
        raise ValueError("Encoded terms must occur exactly once in a ranking")
    if set(term_table["term"]) != set(term_to_source):
        raise ValueError("Term ranking and source mapping must cover identical candidates")
    table = term_table.copy()
    sources = table["term"].map(term_to_source)
    if "feature" in table and not table["feature"].equals(sources):
        raise ValueError("Recorded source feature disagrees with the term mapping")
    table["feature"] = sources
    if not np.isfinite(table["consensus_rank_sum"].to_numpy(dtype=float)).all():
        raise ValueError("Every candidate requires a finite consensus score")
    result = (
        table.sort_values(["consensus_rank_sum", "term"], kind="stable")
        .drop_duplicates("feature", keep="first")
        .sort_values(["consensus_rank_sum", "feature"], kind="stable")
        .reset_index(drop=True)
    )
    result["consensus_rank"] = np.arange(1, len(result) + 1)
    return result


def source_feature_order(ranking: pd.DataFrame, source_names: Iterable[str]) -> list[str]:
    require_columns(ranking, ["feature", "consensus_rank_sum"], "Source ranking")
    expected, actual = set(source_names), set(ranking["feature"])
    if not ranking["feature"].is_unique or actual != expected:
        raise ValueError(
            "Ranking must contain exactly one row per candidate source variable; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    if not np.isfinite(ranking["consensus_rank_sum"].to_numpy(dtype=float)).all():
        raise ValueError("Source ranking contains nonfinite consensus scores")
    return ranking.sort_values(["consensus_rank_sum", "feature"], kind="stable")["feature"].tolist()


def encoded_terms_for_sources(source_names: list[str], term_to_source: dict[str, str]) -> list[str]:
    if len(source_names) != len(set(source_names)):
        raise ValueError("Selected source names must be unique")
    missing = set(source_names) - set(term_to_source.values())
    if missing:
        raise ValueError(f"Selected sources have no encoded terms: {sorted(missing)}")
    return [term for source in source_names for term, mapped in term_to_source.items() if mapped == source]


def lasso_boruta_ranking(
    x: pd.DataFrame,
    y: np.ndarray,
    projection: dict,
    seed: int,
    term_to_source: dict[str, str],
    boruta_max_iter: int = 100,
    boruta_n_estimators: str | int = "auto",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Rank encoded terms by multinomial LASSO importance plus Boruta rank and collapse to sources."""
    lasso = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=10,
                    cv=int(projection["cv_folds"]),
                    penalty="l1",
                    solver="saga",
                    scoring="neg_log_loss",
                    class_weight=projection["class_weight"],
                    max_iter=int(projection["max_iter"]),
                    tol=float(projection["tolerance"]),
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lasso.fit(x, y)
    convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    lasso_importance = np.abs(lasso.named_steps["model"].coef_).max(axis=0)
    lasso_rank = pd.Series(-lasso_importance).rank(method="min").astype(int).to_numpy()
    forest = RandomForestClassifier(
        n_estimators=1000, class_weight="balanced", random_state=seed, n_jobs=n_jobs, max_depth=7
    )
    boruta = BorutaPy(forest, n_estimators=boruta_n_estimators, random_state=seed, max_iter=boruta_max_iter, verbose=0)
    boruta.fit(x.to_numpy(dtype=float), y)
    boruta_rank = boruta.ranking_.astype(int)
    rank_sum = lasso_rank + boruta_rank
    order = np.lexsort((np.asarray(x.columns, dtype=str), rank_sum))
    consensus_rank = np.empty(len(order), dtype=int)
    consensus_rank[order] = np.arange(1, len(order) + 1)
    terms = pd.DataFrame(
        {
            "term": x.columns,
            "feature": [term_to_source[column] for column in x.columns],
            "lasso_max_absolute_coefficient": lasso_importance,
            "lasso_rank": lasso_rank,
            "boruta_rank": boruta_rank,
            "boruta_confirmed": boruta.support_.astype(bool),
            "boruta_tentative": boruta.support_weak_.astype(bool),
            "consensus_rank_sum": rank_sum,
            "consensus_rank": consensus_rank,
        }
    )
    sources = source_consensus_ranking(terms, term_to_source)
    return sources, terms, int(convergence_warnings)

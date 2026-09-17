from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

SEGMENTS = tuple(range(1, 18))
PERFUSION_METRICS = ("MBF", "MBV", "TTP", "PCBV", "FE")
TERRITORIES = {
    "LAD": (1, 2, 7, 8, 13, 14, 17),
    "LCx": (5, 6, 11, 12, 16),
    "RCA": (3, 4, 9, 10, 15),
}
SEGMENT_COLUMNS = {f"{s}_{m}" for s in SEGMENTS for m in PERFUSION_METRICS}
TERRITORY_COLUMNS = {f"{t}_{m}" for t in TERRITORIES for m in PERFUSION_METRICS}

OUTCOME_OR_LABEL_COLUMNS = {
    "MACST",
    "TIME",
    "MACS",
    "Hard_MACS",
    "Composite_MACE",
    "Hard_MACE",
    "Cluster",
    "Cluster_Raw",
    "Cluster_Result",
    "Cohort",
    "P_Cluster1",
    "P_Cluster2",
    "P_Cluster3",
    "Max_probability",
    "CCS_PreRecode",
    "MICE",
}

STUDY_CATEGORIES = {
    "gender": (0, 1),
    "Anti2": (0, 1),
    "smoking": (0, 1),
    "DM": (0, 1),
    "CAD-RADS": (0, 1, 2, 3, 4, 5),
    "Cluster": (1, 2, 3),
}

BOUNDED_UNIT_FEATURES = ("IMV", "CT-FFR-LAD", "CT-FFR-LcX", "CT-FFR-RCA")


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_csv_header(path: str | Path) -> None:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        names = next(csv.reader(stream), [])
    if not names:
        raise ValueError("Input CSV has no header")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate input columns in CSV header")


def load_table(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(source, sheet_name=sheet_name or 0)
    validate_csv_header(source)
    return pd.read_csv(source, encoding="utf-8-sig", low_memory=False)


def semantic_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace("　", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def validate_patient_table(frame: pd.DataFrame, id_column: str = "ID") -> None:
    require_columns(frame, [id_column], "Input table")
    if frame[id_column].isna().any():
        raise ValueError(f"{id_column} contains missing values")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique")


def assert_outcome_independent(feature_names: Iterable[str]) -> None:
    overlap = sorted(set(feature_names).intersection(OUTCOME_OR_LABEL_COLUMNS))
    if overlap:
        raise RuntimeError(f"Outcome or label columns entered the feature set: {overlap}")


def validate_projection_inputs(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input columns")
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    if len(frame) == 0:
        raise ValueError("At least one input row is required")
    try:
        x = frame[features].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Projection inputs must be finite numeric values") from exc
    if not np.isfinite(x).all():
        raise ValueError("Projection inputs must be finite numeric values")
    if np.any(x < 0):
        raise ValueError("Projection inputs must not be negative")
    for name in BOUNDED_UNIT_FEATURES:
        if name in features and np.any(x[:, features.index(name)] > 1):
            raise ValueError(f"{name} must lie in [0, 1]")
    return x


def validate_study_categories(frame: pd.DataFrame, require_all_levels: bool = False) -> None:
    for column, levels in STUDY_CATEGORIES.items():
        if column not in frame:
            raise ValueError(f"Missing categorical column {column}")
        try:
            values = frame[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must use numeric codes {list(levels)}") from exc
        if not np.isfinite(values).all() or not np.isin(values, levels).all():
            raise ValueError(f"{column} must use only the integer codes {list(levels)}")
        missing = sorted(set(levels) - set(values))
        if require_all_levels and missing:
            raise ValueError(f"{column} must contain all levels {list(levels)}; missing levels: {missing}")


def territory_mean(frame: pd.DataFrame, metric: str, territory: str) -> pd.Series:
    derived = f"{territory}_{metric}"
    if derived in frame.columns:
        values = semantic_numeric(frame[derived])
        if values.notna().all():
            if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
                raise ValueError(f"{derived} must contain finite nonnegative values")
            return values
    columns = [f"{segment}_{metric}" for segment in TERRITORIES[territory]]
    require_columns(frame, columns, f"{derived} calculation")
    values = pd.DataFrame({column: semantic_numeric(frame[column]) for column in columns})
    array = values.to_numpy(dtype=float)
    invalid = ~np.isfinite(array) | (array < 0)
    if invalid.any():
        bad = [column for column, flag in zip(columns, invalid.any(axis=0)) if flag]
        raise ValueError(
            f"{derived} requires finite nonnegative values in every segment; "
            f"invalid segment columns: {bad}. Partial-segment averaging is not supported."
        )
    return values.mean(axis=1, skipna=False)


def projection_features(frame: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate input columns")
    direct = {
        "whole lesion volume",
        "CT-FFR-RCA",
        "TC",
        "CT-FFR-LcX",
        "IMV",
        "low attenuation volume",
        "HbA1c",
        "FG",
        "CT-FFR-LAD",
    }
    output = pd.DataFrame(index=frame.index)
    for name in direct:
        require_columns(frame, [name], "Projection input")
        output[name] = semantic_numeric(frame[name])
    output["LAD_MBF"] = territory_mean(frame, "MBF", "LAD")
    output["RCA_PCBV"] = territory_mean(frame, "PCBV", "RCA")
    output["LAD_PCBV"] = territory_mean(frame, "PCBV", "LAD")
    output["LAD_TTP"] = territory_mean(frame, "TTP", "LAD")
    result = output[feature_order]
    missing = result.isna().sum()
    if missing.any():
        raise ValueError(f"Projection variables contain missing values: {missing[missing.gt(0)].to_dict()}")
    validate_projection_inputs(result, feature_order)
    return result


def load_json_model(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    validate_json_model(model)
    return model


def validate_json_model(model: dict) -> None:
    required = {"classes", "features", "scaler_mean", "scaler_scale", "coefficients", "intercepts"}
    missing = sorted(required.difference(model))
    if missing:
        raise ValueError(f"Projection model is missing fields: {missing}")
    classes = np.asarray(model["classes"], dtype=int)
    features = list(model["features"])
    means = np.asarray(model["scaler_mean"], dtype=float)
    scales = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    intercepts = np.asarray(model["intercepts"], dtype=float)
    if not np.array_equal(classes, np.array([1, 2, 3])):
        raise ValueError(f"Expected ordered classes [1, 2, 3], found {classes.tolist()}")
    if len(features) != len(set(features)):
        raise ValueError("Projection feature order contains duplicate names")
    if means.shape != (len(features),) or scales.shape != (len(features),):
        raise ValueError("Scaler vectors do not match the feature count")
    if coefficients.shape != (len(classes), len(features)) or intercepts.shape != (len(classes),):
        raise ValueError("Coefficient and intercept dimensions do not match classes and features")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Scaler parameters must be finite and strictly positive")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercepts).all():
        raise ValueError("Model coefficients and intercepts must be finite")


def predict_json_model(features: pd.DataFrame, model: dict) -> tuple[np.ndarray, np.ndarray]:
    values = validate_projection_inputs(features, model["features"])
    standardized = (values - np.asarray(model["scaler_mean"], dtype=float)) / np.asarray(
        model["scaler_scale"], dtype=float
    )
    logits = standardized @ np.asarray(model["coefficients"], dtype=float).T + np.asarray(
        model["intercepts"], dtype=float
    )
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = np.asarray(model["classes"], dtype=int)
    labels = classes[np.argmax(probabilities, axis=1)]
    return labels, probabilities


def attach_phenotypes(frame: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    for table in (frame, prediction):
        validate_patient_table(table)
    if set(frame.ID) != set(prediction.ID):
        raise ValueError("Cohort and phenotype ID sets must match exactly")
    columns = ["ID", "Cluster"] + [c for c in prediction if c.startswith("P_Cluster") or c == "Max_probability"]
    return frame.drop(columns=[c for c in columns if c != "ID"], errors="ignore").merge(
        prediction[columns], on="ID", how="left", validate="one_to_one"
    )


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy() * len(ordered) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result

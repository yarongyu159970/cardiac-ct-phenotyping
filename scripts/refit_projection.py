"""Refit the 13-feature multinomial classifier and, optionally, run 5x5 repeated out-of-fold evaluation."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_config, load_json_model, load_table, predict_json_model, projection_features
from oof_summary import summarize


def apply_row_order(data: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows by an explicit ID sequence without altering any values."""
    if "ID" not in data or "ID" not in order:
        raise ValueError("Input and row-order table must contain ID")
    if data.ID.isna().any() or order.ID.isna().any() or not data.ID.is_unique or not order.ID.is_unique:
        raise ValueError("Row-order IDs must be complete and unique")
    if set(data.ID) != set(order.ID):
        raise ValueError("Row-order table must contain exactly the input patient IDs")
    return data.set_index("ID").loc[order.ID].reset_index()


def fit(x: pd.DataFrame, y: np.ndarray, config: dict):
    p = config["projection"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=p["c_grid"],
            cv=int(p["cv_folds"]),
            penalty=p["penalty"],
            solver=p["solver"],
            multi_class="multinomial",
            class_weight=p["class_weight"],
            scoring=p["scoring"],
            random_state=int(p["random_state"]),
            max_iter=int(p["max_iter"]),
            tol=float(p["tolerance"]),
            n_jobs=1,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    return model, [str(w.message) for w in caught]


def fold_metrics(y: np.ndarray, probabilities: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y == predicted)),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro"),
        "log_loss": log_loss(y, probabilities, labels=[1, 2, 3]),
        "multiclass_brier": float(np.mean(np.sum((probabilities - np.eye(3)[y - 1]) ** 2, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "input", "frozen-model", "output-dir"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--oof", action="store_true", help="Run 5x5 repeated out-of-fold evaluation")
    parser.add_argument("--row-order", type=Path, help="Optional CSV with an ID column giving the training row order")
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_table(args.input)
    if args.row_order is not None:
        data = apply_row_order(data, load_table(args.row_order))
    frozen = load_json_model(args.frozen_model)
    x = projection_features(data, frozen["features"])
    y = data.Cluster.astype(int).to_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, notes = fit(x, y, config)
    scaler, estimator = model.steps[0][1], model.steps[1][1]
    artifact = {
        "features": list(x),
        "classes": estimator.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": estimator.coef_.tolist(),
        "intercepts": estimator.intercept_.tolist(),
        "selected_C": estimator.C_.tolist(),
    }
    (args.output_dir / "refitted_projection_model.json").write_text(json.dumps(artifact, indent=2))
    frozen_labels, frozen_p = predict_json_model(x, frozen)
    new_p = model.predict_proba(x)
    report = {
        "n": len(data),
        "selected_C": estimator.C_.tolist(),
        "max_absolute_probability_difference": float(np.max(np.abs(new_p - frozen_p))),
        "label_agreement": float(np.mean(model.predict(x) == frozen_labels)),
        "matches_frozen_model": bool(
            np.array_equal(model.predict(x), frozen_labels) and np.allclose(new_p, frozen_p, rtol=0, atol=1e-10)
        ),
        "warnings": notes,
        "row_order": args.row_order.name if args.row_order is not None else "input_file_order",
    }

    if args.oof:
        splitter = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=5, random_state=int(config["projection"]["random_state"])
        )
        sums = np.zeros((len(x), 3))
        counts = np.zeros(len(x), int)
        rows = []
        fold_predictions = []
        for fold, (train, test) in enumerate(splitter.split(x, y), 1):
            fitted, fold_warnings = fit(x.iloc[train], y[train], config)
            probabilities = fitted.predict_proba(x.iloc[test])
            sums[test] += probabilities
            counts[test] += 1
            predicted = fitted.classes_[probabilities.argmax(axis=1)]
            rows.append(
                {
                    "fold": fold,
                    **fold_metrics(y[test], probabilities, predicted),
                    "selected_C": float(fitted.steps[1][1].C_[0]),
                    "warnings": "; ".join(fold_warnings),
                }
            )
            fold_predictions.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "repeat": (fold - 1) // 5 + 1,
                        "ID": data.ID.iloc[test].to_numpy(),
                        "Cluster": y[test],
                        "Predicted_Cluster": predicted,
                        **{f"P_Cluster{k + 1}": probabilities[:, k] for k in range(3)},
                    }
                )
            )
            pd.concat(fold_predictions, ignore_index=True).to_csv(
                args.output_dir / "oof_fold_predictions.csv", index=False
            )
            pd.DataFrame(rows).to_csv(args.output_dir / "oof_fold_metrics.csv", index=False)
            print(f"OOF fold {fold}/25 complete", flush=True)
        mean_probabilities = sums / counts[:, None]
        pd.DataFrame(
            {
                "ID": data.ID,
                "Cluster": y,
                "Predicted_Cluster": mean_probabilities.argmax(axis=1) + 1,
                **{f"P_Cluster{k + 1}": mean_probabilities[:, k] for k in range(3)},
            }
        ).to_csv(args.output_dir / "oof_patient_mean_predictions.csv", index=False)
        summary = summarize(args.output_dir / "oof_fold_predictions.csv", args.output_dir / "oof_summary")
        report["oof_mean_5_repeats"] = dict(zip(summary.metric, summary.mean_5_repeats))

    (args.output_dir / "refit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

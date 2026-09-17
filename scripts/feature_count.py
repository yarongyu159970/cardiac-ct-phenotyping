"""Evaluate classifier performance against the number of top-ranked source variables."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from candidate_features import dummy_encode_candidates, encoded_terms_for_sources, source_feature_order, term_mapping
from common import load_config, load_table, require_columns, semantic_numeric, validate_patient_table


def select_feature_count(performance: pd.DataFrame, primary_metrics: list[str], tolerance: float) -> tuple[int, dict]:
    """Smallest feature count whose primary metrics are all within `tolerance` of their best value."""
    if not primary_metrics:
        raise ValueError("At least one primary metric is required")
    missing = sorted(set(primary_metrics).difference(performance.columns))
    if missing:
        raise ValueError(f"Unknown primary metrics: {missing}")
    if tolerance < 0:
        raise ValueError("near_best_tolerance must be non-negative")
    best = {metric: float(performance[metric].max()) for metric in primary_metrics}
    eligible = np.ones(len(performance), dtype=bool)
    for metric in primary_metrics:
        performance[f"{metric}_gap_to_best"] = best[metric] - performance[metric]
        eligible &= performance[f"{metric}_gap_to_best"].le(tolerance).to_numpy()
    performance["eligible"] = eligible
    if not eligible.any():
        raise RuntimeError("No feature count satisfies the near-best rule")
    selected = int(performance.loc[eligible, "feature_count"].min())
    performance["selected"] = performance["feature_count"].eq(selected)
    return selected, best


def save_plot(performance: pd.DataFrame, selected: int, destination: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    count = performance["feature_count"]
    for metric, sd, label, color in (
        ("balanced_accuracy_mean", "balanced_accuracy_sd", "Balanced accuracy", "#2F6B9A"),
        ("macro_f1_mean", "macro_f1_sd", "Macro-F1", "#D97925"),
    ):
        axes[0].errorbar(
            count,
            performance[metric],
            yerr=performance[sd],
            marker="o",
            linewidth=1.6,
            capsize=2.5,
            label=label,
            color=color,
        )
    for metric, label, color in (
        ("log_loss_mean", "Log loss", "#6A51A3"),
        ("multiclass_brier_mean", "Multiclass Brier score", "#238B45"),
    ):
        axes[1].plot(count, performance[metric], marker="o", linewidth=1.6, label=label, color=color)
    for axis in axes:
        axis.axvline(selected, color="#B2182B", linestyle="--", linewidth=1.2, label=f"Selected n={selected}")
        axis.set_xlabel("Number of ranked features")
        axis.set_xticks(count)
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
    axes[0].set(ylabel="Higher is better", title="Discrimination")
    axes[1].set(ylabel="Lower is better", title="Probability error")
    fig.suptitle("Feature count by repeated stratified cross-validation", fontsize=11)
    fig.tight_layout()
    outputs = []
    for suffix in (".png", ".pdf", ".svg"):
        path = destination.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-features", type=int)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    projection = config["projection"]
    count_config = config.get("feature_count", {})
    minimum = int(args.min_features or count_config.get("min_features", 5))
    maximum_requested = int(args.max_features or count_config.get("max_features", 15))
    tolerance = float(args.tolerance if args.tolerance is not None else count_config.get("near_best_tolerance", 0.010))
    primary_metrics = list(count_config.get("primary_metrics", ["balanced_accuracy_mean", "macro_f1_mean"]))
    solver = str(count_config.get("solver", projection["solver"]))
    max_iter = int(count_config.get("max_iter", projection["max_iter"]))

    frame = load_table(args.input)
    validate_patient_table(frame, config["endpoint"]["id"])
    label_column = projection["label_column"]
    require_columns(frame, [label_column], "Derivation table")
    y = semantic_numeric(frame[label_column]).astype(int).to_numpy()
    selection = config["feature_selection"]
    drop_first = bool(selection["dummy_drop_first"])
    encoded, candidates, categorical = dummy_encode_candidates(
        frame, int(selection["categorical_max_unique"]), drop_first
    )
    mapping = term_mapping(candidates, categorical, drop_first)
    ranking = source_feature_order(load_table(args.ranking), candidates.columns)
    if len(ranking) < minimum:
        raise ValueError("Feature ranking does not contain enough candidates")

    splits = list(
        RepeatedStratifiedKFold(
            n_splits=int(projection["cv_folds"]),
            n_repeats=int(projection["cv_repeats"]),
            random_state=int(projection["random_state"]),
        ).split(candidates, y)
    )
    rows = []
    maximum = min(maximum_requested, len(ranking))
    for count in range(minimum, maximum + 1):
        feature_names = ranking[:count]
        term_names = encoded_terms_for_sources(feature_names, mapping)
        x = encoded[term_names].to_numpy(dtype=float)
        fold_metrics = []
        nonconverged = 0
        for train, test in splits:
            estimator = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=float(projection["selected_c"]),
                            penalty=projection["penalty"],
                            solver=solver,
                            class_weight=projection["class_weight"],
                            max_iter=max_iter,
                            tol=float(projection["tolerance"]),
                            random_state=int(projection["random_state"]),
                        ),
                    ),
                ]
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x[train], y[train])
            nonconverged += sum(issubclass(item.category, ConvergenceWarning) for item in caught)
            probability = estimator.predict_proba(x[test])
            classes = estimator.named_steps["model"].classes_
            prediction = classes[np.argmax(probability, axis=1)]
            one_hot = (y[test, None] == classes[None, :]).astype(float)
            fold_metrics.append(
                [
                    balanced_accuracy_score(y[test], prediction),
                    f1_score(y[test], prediction, average="macro"),
                    log_loss(y[test], probability, labels=classes),
                    np.mean(np.sum((one_hot - probability) ** 2, axis=1)),
                ]
            )
        values = np.asarray(fold_metrics)
        rows.append(
            {
                "feature_count": count,
                "encoded_term_count": len(term_names),
                "balanced_accuracy_mean": values[:, 0].mean(),
                "balanced_accuracy_sd": values[:, 0].std(ddof=1),
                "macro_f1_mean": values[:, 1].mean(),
                "macro_f1_sd": values[:, 1].std(ddof=1),
                "log_loss_mean": values[:, 2].mean(),
                "log_loss_sd": values[:, 2].std(ddof=1),
                "multiclass_brier_mean": values[:, 3].mean(),
                "multiclass_brier_sd": values[:, 3].std(ddof=1),
                "nonconverged_folds": nonconverged,
                "total_folds": len(splits),
                "ranked_features": " | ".join(feature_names),
                "encoded_terms": " | ".join(term_names),
            }
        )

    result = pd.DataFrame(rows)
    result["balanced_accuracy_change_from_previous"] = result["balanced_accuracy_mean"].diff()
    result["macro_f1_change_from_previous"] = result["macro_f1_mean"].diff()
    selected, best = select_feature_count(result, primary_metrics, tolerance)
    final_features = list(projection["features"])
    if len(final_features) != selected:
        raise ValueError(
            f"The near-best rule selected {selected} features, but projection.features contains {len(final_features)}"
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    plot_files = save_plot(result, selected, destination.with_name("feature_count_performance"))
    selected_features = ranking[:selected]
    same_membership = set(selected_features) == set(final_features)
    if bool(selection.get("require_top_n_membership_match", True)) and not same_membership:
        raise ValueError(
            f"Selected ranked features do not match projection.features: ranked={selected_features}, configured={final_features}"
        )
    summary = {
        "selected_feature_count": selected,
        "candidate_range": [minimum, maximum],
        "primary_metrics": primary_metrics,
        "near_best_tolerance": tolerance,
        "best_primary_metric_values": best,
        "cross_validation": {
            "folds": int(projection["cv_folds"]),
            "repeats": int(projection["cv_repeats"]),
            "random_state": int(projection["random_state"]),
        },
        "classifier": {
            "penalty": projection["penalty"],
            "C": float(projection["selected_c"]),
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": projection["class_weight"],
        },
        "selected_features": selected_features,
        "selected_encoded_terms": encoded_terms_for_sources(selected_features, mapping),
        "final_features": final_features,
        "same_membership_as_final": same_membership,
        "same_order_as_final": selected_features == final_features,
        "plot_files": plot_files,
    }
    destination.with_name("feature_count_selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.drop(columns=["ranked_features", "encoded_terms"]).to_string(index=False))
    print(f"Selected feature count: {selected}")


if __name__ == "__main__":
    main()

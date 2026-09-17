"""Summarise repeated out-of-fold predictions of the 13-feature classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss

METRICS = ["balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier"]
PROBABILITY_COLUMNS = ["P_Cluster1", "P_Cluster2", "P_Cluster3"]


def scores(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.Cluster.to_numpy(int)
    predicted = frame.Predicted_Cluster.to_numpy(int)
    p = frame[PROBABILITY_COLUMNS].to_numpy(float)
    if not set(y).issubset({1, 2, 3}) or not set(predicted).issubset({1, 2, 3}):
        raise ValueError("Invalid class labels")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-10):
        raise ValueError("Invalid class probabilities")
    if not np.array_equal(predicted, p.argmax(1) + 1):
        raise ValueError("Predicted labels disagree with probabilities")
    return {
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "macro_f1": f1_score(y, predicted, average="macro"),
        "log_loss": log_loss(y, p, labels=[1, 2, 3]),
        "multiclass_brier": float(np.mean(np.sum((p - np.eye(3)[y - 1]) ** 2, axis=1))),
    }


def confusion_table(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = confusion_matrix(frame.Cluster, frame.Predicted_Cluster, labels=[1, 2, 3])
    return pd.DataFrame(matrix, index=["True1", "True2", "True3"], columns=["Pred1", "Pred2", "Pred3"])


def summarize(predictions: Path, output: Path) -> pd.DataFrame:
    fp = pd.read_csv(predictions)
    if fp.ID.isna().any() or fp[["repeat", "ID"]].duplicated().any():
        raise ValueError("Duplicate or missing patient ID within a repeat")
    if set(fp.fold.unique()) != set(range(1, 26)) or set(fp["repeat"].unique()) != set(range(1, 6)):
        raise ValueError("Exactly 25 folds and 5 repeats are required")
    if not ((fp.fold - 1) // 5 + 1 == fp["repeat"]).all():
        raise ValueError("Fold/repeat mapping is inconsistent")
    if not fp.groupby("ID").Cluster.nunique().eq(1).all():
        raise ValueError("Patient targets changed between repeats")
    if not fp.groupby("ID").size().eq(5).all():
        raise ValueError("Each patient must have one prediction per repeat")

    folds = pd.DataFrame([{"fold": int(k), **scores(g)} for k, g in fp.groupby("fold")])
    repeats = pd.DataFrame([{"repeat": int(k), **scores(g)} for k, g in fp.groupby("repeat")])
    result = pd.DataFrame(
        [
            {"metric": m, "mean_5_repeats": repeats[m].mean(), "sd_across_25_folds": folds[m].std(ddof=1)}
            for m in METRICS
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "oof_metrics_by_fold.csv", index=False)
    repeats.to_csv(output / "oof_metrics_by_repeat.csv", index=False)
    result.to_csv(output / "oof_repeat_mean.csv", index=False)
    confusion_table(fp).to_csv(output / "oof_pooled_confusion.csv")
    confusion_table(fp[fp["repeat"].eq(5)]).to_csv(output / "oof_repeat_5_confusion.csv")
    from figures import plot_oof_figure

    plot_oof_figure(fp, output / "Supplementary_Figure_6")
    print(result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()

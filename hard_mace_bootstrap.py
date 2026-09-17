"""Optimism-corrected hard-MACE performance with two-stage (nested) bootstrap percentile intervals.

Stage 1 ("point"): B bootstrap samples, each refitting all cause-specific models; optimism is the
mean of training-minus-test performance and the corrected estimate is apparent minus optimism.
Stage 2 ("nested"): for each of B_outer outer samples, B_inner inner refits estimate that sample's
optimism; the 2.5/97.5 percentiles of the outer corrected estimates form the interval.
Paired differences B-A and C-B are formed within each replicate before any quantile.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

METRICS = ["cause_specific_Harrell_C", "CIF_Brier_36", "CIF_IBS_3_36", "CIF_Brier_60", "CIF_IBS_3_60"]
METRIC_LABELS = dict(zip(METRICS, ["C-index", "Brier36", "IBS36", "Brier60", "IBS60"]))
TERMS = ["A", "B", "C", "B_minus_A", "C_minus_B"]
TERM_LABELS = dict(zip(TERMS, ["A", "B", "C", "B-A", "C-B"]))


def expand(x: np.ndarray) -> np.ndarray:
    """Append paired differences B-A and C-B along the model axis."""
    return np.concatenate(
        [x, (x[..., 1, :] - x[..., 0, :])[..., None, :], (x[..., 2, :] - x[..., 1, :])[..., None, :]], axis=-2
    )


def load_record(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Incomplete run: missing {path}")
    record = json.loads(path.read_text())
    if "error" in record:
        raise ValueError(f"Failed replicate {path}: {record['error']}")
    return record


def array(record: dict, key: str, shape: tuple, path: Path) -> np.ndarray:
    values = np.asarray(record[key], dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid {key} in {path}: expected finite array of shape {shape}, got {values.shape}")
    return values


def load_cohort(root: Path, cohort: str, point_n: int, outer_n: int, inner_n: int):
    point_dir, nested_dir = root / f"{cohort}_point", root / f"{cohort}_nested"
    apparent = array(load_record(point_dir / "0000.json"), "apparent", (3, 5), point_dir / "0000.json")
    point_train, point_test, outer_train, inner = [], [], [], []
    for i in range(1, point_n + 1):
        path = point_dir / f"{i:04d}.json"
        record = load_record(path)
        point_train.append(array(record, "train", (3, 5), path))
        point_test.append(array(record, "test", (3, 5), path))
    for i in range(1, outer_n + 1):
        path = nested_dir / f"{i:04d}.json"
        record = load_record(path)
        if record.get("inner_errors"):
            raise ValueError(f"Incomplete inner sampling in {path}: {record['inner_errors']}")
        train = array(record, "train", (3, 5), path)
        if i <= point_n and not np.allclose(train, point_train[i - 1], atol=1e-10, rtol=1e-9):
            raise ValueError(f"Point and nested outer replicates disagree at {path}")
        outer_train.append(train)
        inner.append(array(record, "inner_delta", (inner_n, 3, 5), path))
    return (
        expand(apparent),
        expand(np.asarray(point_train)),
        expand(np.asarray(point_test)),
        expand(np.asarray(outer_train)),
        expand(np.asarray(inner)),
    )


def summarize_cohort(
    root: Path, cohort: str, point_n: int, outer_n: int, inner_n: int
) -> tuple[list[dict], list[dict]]:
    apparent, train, test, outer_apparent, inner_delta = load_cohort(root, cohort, point_n, outer_n, inner_n)
    optimism = (train - test).mean(axis=0)
    corrected = apparent - optimism
    optimism_mcse = (train - test).std(axis=0, ddof=1) / np.sqrt(point_n)
    nested = outer_apparent - inner_delta.mean(axis=1)
    lower, upper = np.quantile(nested, [0.025, 0.975], axis=0)
    rows, replicates = [], []
    for j, term in enumerate(TERMS):
        for m, metric in enumerate(METRICS):
            rows.append(
                {
                    "cohort": cohort,
                    "term": term,
                    "metric": metric,
                    "apparent": float(apparent[j, m]),
                    "optimism": float(optimism[j, m]),
                    "corrected": float(corrected[j, m]),
                    "ci_lower": float(lower[j, m]),
                    "ci_upper": float(upper[j, m]),
                    "optimism_mcse": float(optimism_mcse[j, m]),
                    "point_n": point_n,
                    "outer_n": outer_n,
                    "inner_n": inner_n,
                }
            )
            for b in range(outer_n):
                replicates.append(
                    {
                        "cohort": cohort,
                        "outer_index": b + 1,
                        "term": term,
                        "metric": metric,
                        "corrected": float(nested[b, j, m]),
                    }
                )
    return rows, replicates


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, out: Path, point_n: int, outer_n: int, inner_n: int) -> None:
    rows, replicates = [], []
    for cohort in ["derivation", "validation"]:
        cohort_rows, cohort_replicates = summarize_cohort(root, cohort, point_n, outer_n, inner_n)
        rows.extend(cohort_rows)
        replicates.extend(cohort_replicates)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "corrected_estimates.csv", rows)
    write_csv(out / "nested_corrected_replicates.csv", replicates)
    write_csv(
        out / "corrected_table_long.csv",
        [
            {
                "cohort": r["cohort"],
                "metric": METRIC_LABELS[r["metric"]],
                "comparison": TERM_LABELS[r["term"]],
                "estimate": r["corrected"],
                "lower": r["ci_lower"],
                "upper": r["ci_upper"],
            }
            for r in rows
        ],
    )
    (out / "settings.json").write_text(
        json.dumps({"point_n": point_n, "outer_n": outer_n, "inner_n": inner_n}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--outer", type=int, default=500)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.bootstrap_repeats, args.outer) < 2 or args.inner < 2 or args.jobs < 1:
        parser.error("Require bootstrap-repeats and outer >= 2, inner >= 2 and jobs >= 1")

    engine = Path(__file__).with_name("competing_cox.py")
    if not args.summarize_only:
        for cohort in ["derivation", "validation"]:
            for phase, count in [("point", args.bootstrap_repeats), ("nested", args.outer)]:
                subprocess.run(
                    [
                        sys.executable,
                        str(engine),
                        "--input-dir",
                        str(args.input_dir),
                        "--cohort",
                        cohort,
                        "--phase",
                        phase,
                        "--count",
                        str(count),
                        "--inner",
                        str(args.inner),
                        "--workers",
                        str(args.jobs),
                        "--out",
                        str(args.output_dir / f"{cohort}_{phase}"),
                    ],
                    check=True,
                )
    summarize(args.output_dir, args.output_dir / "intervals", args.bootstrap_repeats, args.outer, args.inner)


if __name__ == "__main__":
    main()

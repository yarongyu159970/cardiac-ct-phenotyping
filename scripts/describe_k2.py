"""Descriptive heatmap and label export for the K=2 sensitivity solution (derivation cohort only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_table, require_columns, semantic_numeric, validate_patient_table
from figures import plot_k2_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cohort = load_table(args.cohort)
    labels = load_table(args.labels)
    validate_patient_table(cohort)
    validate_patient_table(labels)
    require_columns(labels, ["ID", "K2_Cluster"], "K=2 label table")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_k2_heatmap(cohort, args.labels, args.output_dir / "K2_phenotypic_heatmap")
    labels[["ID", "K2_Cluster"]].to_csv(args.output_dir / "K2_patient_labels.csv", index=False)
    sizes = semantic_numeric(labels["K2_Cluster"]).astype(int).value_counts().sort_index()
    summary = {"n": int(len(labels)), "cluster_sizes": {str(k): int(v) for k, v in sizes.items()}}
    (args.output_dir / "K2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

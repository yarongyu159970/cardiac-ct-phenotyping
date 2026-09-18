"""Attach K=3 cluster labels produced by clustering.py to the derivation cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import load_table, require_columns, semantic_numeric, validate_patient_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cohort = load_table(args.cohort)
    labels = load_table(args.labels)
    validate_patient_table(cohort)
    validate_patient_table(labels)
    require_columns(labels, ["ID", "K3_Cluster"], "K=3 label table")
    if set(cohort["ID"]) != set(labels["ID"]):
        raise ValueError("K=3 labels and derivation cohort must contain the same IDs")
    unexpected = sorted(set(labels.columns) - {"ID", "K3_Cluster", "K3_Cluster_Raw"})
    if unexpected:
        raise ValueError(f"K=3 label file has unexpected columns: {unexpected}")

    previous = semantic_numeric(cohort["Cluster"]).astype(int) if "Cluster" in cohort.columns else None
    output = cohort.drop(columns=["Cluster", "Cluster_Raw"], errors="ignore").merge(
        labels, on="ID", how="left", validate="one_to_one"
    )
    output["Cluster"] = semantic_numeric(output.pop("K3_Cluster")).astype(int)
    if "K3_Cluster_Raw" in output.columns:
        output["Cluster_Raw"] = semantic_numeric(output.pop("K3_Cluster_Raw")).astype(int)
    if set(output["Cluster"].unique()) != {1, 2, 3}:
        raise ValueError("K=3 labels must contain clusters 1, 2 and 3")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "n": int(len(output)),
        "cluster_sizes": {str(k): int(v) for k, v in output["Cluster"].value_counts().sort_index().items()},
    }
    if previous is not None:
        current = output.set_index("ID").loc[cohort["ID"], "Cluster"].to_numpy()
        summary["agreement_with_input_cluster"] = {
            "adjusted_rand_index": float(adjusted_rand_score(previous, current)),
            "normalized_mutual_information": float(normalized_mutual_info_score(previous, current)),
        }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

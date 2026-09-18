"""Export the derivation preprocessing matrices used for clustering and feature selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from candidate_features import dummy_encode_candidates, term_mapping
from clustering_features import build_clustering_matrix
from common import assert_outcome_independent, load_config, load_table, validate_patient_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = load_table(args.input)
    validate_patient_table(frame)
    config = load_config(args.config)
    selection = config["feature_selection"]
    mixed, indices, numeric, categorical, diagnostics = build_clustering_matrix(frame, config["preprocessing"])
    encoded, candidates, categorical_features = dummy_encode_candidates(
        frame, selection["categorical_max_unique"], selection["dummy_drop_first"]
    )
    assert_outcome_independent([*numeric.columns, *categorical.columns, *candidates.columns])
    mapping = term_mapping(candidates, categorical_features, selection["dummy_drop_first"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"term": term, "feature": source, "categorical": source in categorical_features}
            for term, source in mapping.items()
        ]
    ).to_csv(args.output_dir / "term_to_source.csv", index=False)
    for name, table in [
        ("clustering_numeric_standardized", numeric),
        ("clustering_categorical", categorical),
        ("candidate_source_features", candidates),
        ("candidate_dummy_encoded", encoded),
    ]:
        result = table.copy()
        result.insert(0, "ID", frame.ID.to_numpy())
        result.to_csv(args.output_dir / f"{name}.csv", index=False)
    summary = {}
    for key, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(args.output_dir / f"{key}.csv", index=False)
        else:
            summary[key] = value
    summary.update(
        n=len(frame),
        clustering_shape=list(mixed.shape),
        categorical_indices=indices,
        candidate_source_count=len(candidates.columns),
        encoded_term_count=len(encoded.columns),
        dummy_categorical_features=categorical_features,
    )
    (args.output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

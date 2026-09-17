"""Rank candidate source variables for the phenotype classifier by LASSO and Boruta consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidate_features import dummy_encode_candidates, lasso_boruta_ranking, term_mapping
from common import load_config, load_table, require_columns, semantic_numeric, validate_patient_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    projection = config["projection"]
    selection = config["feature_selection"]
    frame = load_table(args.input)
    validate_patient_table(frame, config["endpoint"]["id"])
    label_column = projection["label_column"]
    require_columns(frame, [label_column], "Derivation table")
    labels = semantic_numeric(frame[label_column])
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({1, 2, 3}):
        raise ValueError(f"{label_column} must contain labels 1, 2 and 3")
    y = labels.astype(int).to_numpy()
    drop_first = bool(selection["dummy_drop_first"])
    x, original, categorical = dummy_encode_candidates(frame, int(selection["categorical_max_unique"]), drop_first)
    mapping = term_mapping(original, categorical, drop_first)

    seed = int(projection["random_state"])
    result, terms, convergence_warnings = lasso_boruta_ranking(x, y, projection, seed, mapping)
    final_features = list(projection["features"])
    ranked_top = result["feature"].tolist()[: len(final_features)]
    same_membership = set(ranked_top) == set(final_features)
    result["in_final_feature_set"] = result["feature"].isin(final_features)
    result["selected_top_n"] = result["consensus_rank"] <= len(final_features)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    terms.sort_values(["consensus_rank_sum", "term"], kind="stable").to_csv(
        destination.with_name(f"{destination.stem}_terms.csv"), index=False
    )
    summary = {
        "random_state": seed,
        "n_patients": int(len(frame)),
        "n_source_candidates": int(original.shape[1]),
        "n_encoded_terms": int(x.shape[1]),
        "categorical_features": categorical,
        "dummy_drop_first": drop_first,
        "lasso_convergence_warnings": convergence_warnings,
        "ranked_top_n": ranked_top,
        "final_features": final_features,
        "same_membership_as_final": same_membership,
        "same_order_as_final": ranked_top == final_features,
    }
    destination.with_name("feature_ranking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    if bool(selection.get("require_top_n_membership_match", True)) and not same_membership:
        raise RuntimeError(
            f"Consensus ranking does not reproduce the configured feature set: ranked={ranked_top}, configured={final_features}"
        )


if __name__ == "__main__":
    main()

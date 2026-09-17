"""Assign phenotypes in an external cohort with the frozen 13-feature classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_json_model, load_table, predict_json_model, projection_features, validate_patient_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    frame = load_table(args.input)
    validate_patient_table(frame)
    model = load_json_model(args.model)
    features = projection_features(frame, model["features"])
    labels, probabilities = predict_json_model(features, model)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("Projection probabilities do not sum to one")
    output = pd.DataFrame(
        {
            "ID": frame["ID"].to_numpy(),
            "Cluster": labels,
            "P_Cluster1": probabilities[:, 0],
            "P_Cluster2": probabilities[:, 1],
            "P_Cluster3": probabilities[:, 2],
            "Max_probability": probabilities.max(axis=1),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output["Cluster"].value_counts().sort_index().rename("n").to_string())


if __name__ == "__main__":
    main()

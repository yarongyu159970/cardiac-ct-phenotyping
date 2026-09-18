"""Assign CAD phenotypes with the frozen 13-feature classifier from a CSV of input features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_json_model, load_table, predict_json_model, projection_features  # noqa: E402

MODEL = ROOT / "artifacts" / "final_projection_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV containing the 13 classifier features (any column order)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        parser.error("Output must differ from the input file")
    model = load_json_model(MODEL)
    frame = load_table(args.input)
    labels, probabilities = predict_json_model(projection_features(frame, model["features"]), model)
    result = pd.DataFrame(
        {
            "row_number": range(1, len(frame) + 1),
            "Cluster": labels,
            **{f"P_Cluster{k + 1}": probabilities[:, k] for k in range(3)},
            "Max_probability": probabilities.max(1),
        }
    )
    if "ID" in frame.columns:
        result.insert(0, "ID", frame["ID"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} phenotype assignments to {args.output}")


if __name__ == "__main__":
    main()

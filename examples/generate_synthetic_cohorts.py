"""Generate two artificial 240-patient cohorts for exercising the pipeline without patient data.

Distributions are hand specified; derivation labels are assigned with the frozen classifier and
outcomes are drawn independently of all covariates. The data carry no scientific information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import TERRITORIES, load_json_model, predict_json_model, projection_features  # noqa: E402

SEEDS = {"derivation": 120926, "validation": 120927}
BINARY = [
    "gender",
    "DM",
    "HTN",
    "dislipidemia",
    "smoking",
    "HRP",
    "LAP",
    "PR",
    "NRS",
    "SC",
    "Anti1",
    "Anti2",
    "Antidiabetic",
    "Antiischemic",
    "Nitrates",
    "Statin",
]
PERFUSION = [
    ("MBF", [175.0, 140.0, 90.0], 12.0),
    ("MBV", [12.0, 10.0, 7.0], 1.0),
    ("TTP", [10.0, 13.0, 17.0], 1.0),
    ("PCBV", [11.0, 9.0, 6.0], 0.8),
    ("FE", [90.0, 75.0, 55.0], 4.0),
]


def make_cohort(name: str, seed: int, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profile = rng.permutation(np.tile(np.arange(3), n // 3))

    def measurement(centers, sd):
        return np.maximum(0.001, np.asarray(centers)[profile] + rng.normal(0, sd, n))

    d = pd.DataFrame({"ID": [f"SYNTHETIC_{name.upper()}_{i:04d}" for i in range(1, n + 1)]})
    d["age"] = rng.uniform(40, 80, n)
    d["BMI"] = rng.uniform(19, 32, n)
    d["SBP"] = rng.uniform(105, 165, n)
    for field in BINARY:
        d[field] = rng.permutation(np.tile([0, 1], n // 2))
    for field, levels in [("CAD-RADS", 6), ("CCS", 3), ("CACS", 4)]:
        d[field] = rng.permutation(np.arange(n) % levels) + (0 if field == "CAD-RADS" else 1)
    d["TC"] = measurement([4.2, 7.5, 4.8], 0.4)
    d["FG"] = measurement([5.0, 11.0, 6.0], 0.6)
    d["HbA1c"] = measurement([5.5, 9.5, 6.5], 0.4)
    d["HDL"] = rng.uniform(0.8, 1.8, n)
    d["LDL"] = rng.uniform(1.5, 4.5, n)
    d["TG"] = rng.uniform(0.7, 3.0, n)
    d["whole lesion volume"] = measurement([80.0, 230.0, 1000.0], 25.0)
    d["low attenuation volume"] = d["whole lesion volume"] * rng.uniform(0.01, 0.12, n)
    d["calcified volume"] = d["whole lesion volume"] * 0.30
    d["fibrotic volume"] = d["whole lesion volume"] * 0.40
    d["fibrous fatty volume"] = (
        d["whole lesion volume"] - d["low attenuation volume"] - d["calcified volume"] - d["fibrotic volume"]
    )
    for artery in ["LAD", "LcX", "RCA"]:
        d[f"CT-FFR-{artery}"] = np.clip(measurement([0.96, 0.92, 0.65], 0.025), 0, 1)
    d["IMV"] = np.clip(measurement([0.01, 0.08, 0.25], 0.012), 0, 1)
    segments = {}
    for metric, centers, sd in PERFUSION:
        for s in range(1, 18):
            segments[f"{s}_{metric}"] = measurement(centers, sd)
        for territory, indices in TERRITORIES.items():
            segments[f"{territory}_{metric}"] = np.mean([segments[f"{s}_{metric}"] for s in indices], axis=0)
    d = pd.concat([d, pd.DataFrame(segments)], axis=1)
    d["Global"] = d[[f"{s}_MBF" for s in range(1, 18)]].mean(axis=1)
    model = load_json_model(ROOT / "artifacts" / "final_projection_model.json")
    labels, _ = predict_json_model(projection_features(d, model["features"]), model)
    if set(labels) != {1, 2, 3}:
        raise ValueError("Synthetic profiles must produce all three classifier labels")
    if name == "derivation":
        d["Cluster"] = labels
    d["MACST"] = rng.permutation(np.tile([0, 0, 1, 2], n // 4))
    d["TIME"] = np.where(d.MACST.eq(0), rng.uniform(18, 96, n), rng.uniform(3, 72, n))
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "synthetic")
    args = parser.parse_args()
    paths = {name: args.output_dir / f"{name}.csv" for name in SEEDS}
    if any(p.exists() for p in paths.values()):
        parser.error("Output files already exist; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, seed in SEEDS.items():
        frame = make_cohort(name, seed)
        frame.to_csv(paths[name], index=False, float_format="%.10g")
        print(f"{paths[name].name}: {len(frame)} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    main()

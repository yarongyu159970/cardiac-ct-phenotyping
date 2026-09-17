"""Sensitivity of hard-MACE phenotype hazard ratios (Model C) to the ridge penalty."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

import competing_cox as engine
from survival import FORMULAS, cluster_contrast

PENALTIES = [0.001, 0.01, 0.1, 1.0]
STEP_SIZES = [0.1, 0.5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, checks = [], []
    for cohort in ["derivation", "validation"]:
        engine.init(cohort, args.input_dir)
        frame = engine.D
        for penalty in PENALTIES:
            fits = []
            for step in STEP_SIZES:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = CoxPHFitter(penalizer=penalty).fit(
                        frame,
                        "TIME",
                        "hard_event",
                        formula=FORMULAS["C"],
                        fit_options={"step_size": step, "precision": 1e-10, "r_precision": 1e-12, "max_steps": 1000},
                    )
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError(f"Cox fit did not converge ({cohort}, penalty {penalty}, step {step})")
                fits.append(model)
            difference = float(np.max(np.abs(fits[0].params_ - fits[1].params_)))
            if difference >= 1e-5:
                raise RuntimeError(f"Estimates differ across step sizes ({cohort}, penalty {penalty})")
            checks.append({"cohort": cohort, "penalizer": penalty, "max_coefficient_difference": difference})
            for numerator, denominator in [(2, 1), (3, 1), (3, 2)]:
                rows.append(
                    {
                        "cohort": cohort,
                        "penalizer": penalty,
                        "contrast": f"Cluster_{numerator}_vs_{denominator}",
                        **cluster_contrast(fits[0], numerator, denominator),
                    }
                )
    pd.DataFrame(rows).to_csv(args.output_dir / "penalty_sensitivity_cluster_hr.csv", index=False)
    pd.DataFrame(checks).to_csv(args.output_dir / "convergence_checks.csv", index=False)


if __name__ == "__main__":
    main()

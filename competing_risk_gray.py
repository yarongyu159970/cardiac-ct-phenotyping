"""Run cmprsk cumulative-incidence estimation and Gray's tests by phenotype (requires R with cmprsk)."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def r_command(executable: Path, script: Path, arguments: list[str]) -> list[str]:
    if executable.name == "R":
        return [str(executable), "--vanilla", "--slave", f"--file={script}", "--args", *arguments]
    return [str(executable), str(script), *arguments]


def validate_inputs(cohort_path: Path, phenotype_path: Path) -> None:
    cohort = pd.read_csv(cohort_path, encoding="utf-8-sig", low_memory=False)
    phenotypes = pd.read_csv(phenotype_path, encoding="utf-8-sig", low_memory=False)
    for name, frame, required in (
        ("cohort", cohort, {"ID", "TIME", "MACST"}),
        ("phenotypes", phenotypes, {"ID", "Cluster"}),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} input is missing required columns: {missing}")
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise ValueError(f"{name} ID values must be complete and unique")
    if set(cohort["ID"].astype(str)) != set(phenotypes["ID"].astype(str)):
        raise ValueError("Cohort and phenotype files must contain the same patient IDs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--phenotypes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--r-executable", type=Path, default=Path("Rscript"))
    parser.add_argument("--r-home", type=Path)
    args = parser.parse_args()

    script = Path(__file__).with_name("competing_risk_gray.R").resolve()
    validate_inputs(args.cohort.resolve(), args.phenotypes.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = r_command(
        args.r_executable, script, [str(args.cohort.resolve()), str(args.phenotypes.resolve()), str(output)]
    )
    environment = os.environ.copy()
    if args.r_home:
        environment["R_HOME"] = str(args.r_home.resolve())
    subprocess.run(command, check=True, env=environment)

    tests = pd.read_csv(output / "gray_tests.csv")
    expected = {"Hard_MACE", "Soft_MACE"}
    if (
        set(tests["endpoint"]) != expected
        or tests["endpoint"].duplicated().any()
        or tests["p_value"].isna().any()
        or not tests["p_value"].between(0, 1).all()
        or not tests["df"].eq(2).all()
    ):
        raise RuntimeError(f"Gray-test output is incomplete: expected {expected}, observed {set(tests['endpoint'])}")
    curves = pd.read_csv(output / "cumulative_incidence_source_data.csv")
    required = {"Cluster", "event_code", "time", "cumulative_incidence", "variance", "endpoint"}
    if not required.issubset(curves.columns) or not set(curves["endpoint"]).issubset(expected):
        raise RuntimeError("Cumulative-incidence source data are incomplete")
    if (curves["variance"] < -1e-12).any():
        raise RuntimeError("Negative cumulative-incidence variance")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()

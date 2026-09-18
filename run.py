"""Run the analysis pipeline stage by stage, logging each script call into an output directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from common import attach_phenotypes, load_config, load_table, validate_patient_table  # noqa: E402

ALL_STAGES = [
    "prepare",
    "preprocess",
    "cluster",
    "selection",
    "feature_count",
    "stability",
    "projection_refit",
    "oof_summary",
    "survival",
    "hard",
    "comparisons",
    "hr_sensitivity",
    "interactions",
    "competing",
    "figures",
    "calibration",
    "tables",
]
FULL_STAGES = [
    "prepare",
    "preprocess",
    "cluster",
    "survival",
    "hard",
    "comparisons",
    "hr_sensitivity",
    "interactions",
    "competing",
    "figures",
    "calibration",
    "tables",
]
SMOKE_STAGES = ["prepare", "survival", "hard", "comparisons", "hr_sensitivity", "calibration", "tables"]
MODEL = ROOT / "artifacts" / "final_projection_model.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="smoke", help="smoke uses very small bootstrap counts"
    )
    parser.add_argument("--stages", default=None, help=f"Comma-separated subset of: {','.join(ALL_STAGES)}")
    parser.add_argument("--derivation", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip stages already completed in the output directory")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--oof", action="store_true", help="Run the 25-fold out-of-fold evaluation in projection_refit")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.derivation, args.validation, args.config = (
        p.resolve() for p in (args.derivation, args.validation, args.config)
    )
    for path in [args.derivation, args.validation, args.config]:
        if not path.is_file():
            parser.error(f"Missing input file: {path}")

    output = (args.output or ROOT / "results" / f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; use a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(manifest_path.read_text())
        if args.resume and manifest_path.exists()
        else {
            "mode": args.mode,
            "python": sys.version,
            "commands": [],
            "stages_completed": [],
            "status": "running",
        }
    )
    (output / "logs").mkdir(exist_ok=True)
    cache = output / ".cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(cache),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    def call(script: str, *options) -> None:
        command = [sys.executable, str(SCRIPTS / script), *[str(v) for v in options]]
        log = output / "logs" / f"{len(manifest['commands']) + 1:02d}_{Path(script).stem}.log"
        entry = {"command": command, "log": str(log.relative_to(output)), "start": time.time()}
        print(f"Running {script}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, elapsed_seconds=time.time() - entry["start"])
        manifest["commands"].append(entry)
        save()
        if result.returncode:
            print(log.read_text()[-5000:])
            raise RuntimeError(f"{script} failed; see {log}")

    stages = args.stages.split(",") if args.stages else (FULL_STAGES if args.mode == "full" else SMOKE_STAGES)
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}")
    cfg = load_config(args.config)
    sv = cfg["survival"]
    smoke = args.mode == "smoke"
    ready = output / "analysis_inputs"
    ready.mkdir(exist_ok=True)
    derivation = ready / "derivation.csv"
    validation = ready / "validation_projected.csv"
    predictions = output / "external_predictions.csv"

    try:
        for stage in stages:
            if stage in manifest["stages_completed"]:
                print(f"Already completed: {stage}")
                continue
            if stage != "prepare" and not (derivation.exists() and validation.exists()):
                raise ValueError("Run the prepare stage first")
            if stage == "prepare":
                for path in [args.derivation, args.validation]:
                    frame = load_table(path)
                    validate_patient_table(frame)
                    if (
                        frame.MACET.isna().any()
                        or frame.TIME.isna().any()
                        or not set(frame.MACET.unique()).issubset({0, 1, 2})
                        or (frame.TIME < 0).any()
                    ):
                        raise ValueError(f"{path.name}: MACET must be 0/1/2 and TIME nonnegative")
                frame = load_table(args.derivation)
                if set(frame.Cluster) != {1, 2, 3}:
                    raise ValueError("Derivation table must contain Cluster labels 1, 2 and 3")
                frame.to_csv(derivation, index=False)
                call("apply_projection.py", "--input", args.validation, "--model", MODEL, "--output", predictions)
                attach_phenotypes(load_table(args.validation), load_table(predictions)).to_csv(validation, index=False)
            elif stage == "preprocess":
                call(
                    "preprocess.py",
                    "--config",
                    args.config,
                    "--input",
                    derivation,
                    "--output-dir",
                    output / "preprocessing",
                )
            elif stage == "cluster":
                extra = ["--bootstrap-repeats", "2", "--bootstrap-n-init", "1"] if smoke else []
                call(
                    "clustering.py",
                    "--config",
                    args.config,
                    "--input",
                    args.derivation,
                    "--output-dir",
                    output / "clustering",
                    "--reference-column",
                    "Cluster",
                    "--jobs",
                    args.jobs,
                    *extra,
                )
                call(
                    "attach_labels.py",
                    "--cohort",
                    args.derivation,
                    "--labels",
                    output / "clustering" / "labels_k3.csv",
                    "--output",
                    derivation,
                )
                call(
                    "describe_k2.py",
                    "--cohort",
                    derivation,
                    "--labels",
                    output / "clustering" / "labels_k2.csv",
                    "--output-dir",
                    output / "k2",
                )
            elif stage == "selection":
                call(
                    "feature_selection.py",
                    "--config",
                    args.config,
                    "--input",
                    derivation,
                    "--output",
                    output / "selection" / "feature_ranking.csv",
                )
            elif stage == "feature_count":
                call(
                    "feature_count.py",
                    "--config",
                    args.config,
                    "--input",
                    derivation,
                    "--ranking",
                    output / "selection" / "feature_ranking.csv",
                    "--output",
                    output / "selection" / "feature_count_performance.csv",
                )
            elif stage == "stability":
                extra = ["--max-splits", "1", "--boruta-max-iter", "5"] if smoke else []
                call(
                    "feature_stability.py",
                    "--config",
                    args.config,
                    "--input",
                    derivation,
                    "--output-dir",
                    output / "feature_stability",
                    *extra,
                )
                if not smoke:
                    call(
                        "plot_feature_stability.py",
                        "--source-dir",
                        output / "feature_stability",
                        "--output-dir",
                        output / "feature_stability" / "focused",
                    )
            elif stage == "projection_refit":
                call(
                    "refit_projection.py",
                    "--config",
                    args.config,
                    "--input",
                    derivation,
                    "--frozen-model",
                    MODEL,
                    "--output-dir",
                    output / "projection_refit",
                    *(["--oof"] if args.oof else []),
                )
            elif stage == "oof_summary":
                source = output / "projection_refit" / "oof_fold_predictions.csv"
                if not source.exists():
                    raise ValueError("Run projection_refit with --oof first")
                call("oof_summary.py", "--predictions", source, "--output-dir", output / "oof_summary")
            elif stage == "survival":
                for name, source in [("derivation", derivation), ("validation", validation)]:
                    call(
                        "survival.py",
                        "--config",
                        args.config,
                        "--cohort",
                        source,
                        "--phenotypes",
                        source,
                        "--output-dir",
                        output / f"{name}_survival",
                        "--bootstrap",
                        20 if smoke else sv["bootstrap_repeats"],
                    )
            elif stage == "hard":
                call(
                    "hard_mace_bootstrap.py",
                    "--input-dir",
                    ready,
                    "--output-dir",
                    output / "hard_competing",
                    "--bootstrap-repeats",
                    3 if smoke else sv["hard_bootstrap_repeats"],
                    "--outer",
                    3 if smoke else sv["hard_ci_outer_repeats"],
                    "--inner",
                    4 if smoke else sv["hard_ci_inner_repeats"],
                    "--jobs",
                    args.jobs,
                )
            elif stage == "comparisons":
                call(
                    "model_comparison.py",
                    "--input-dir",
                    ready,
                    "--out",
                    output / "model_comparison",
                    "--repeats",
                    9 if smoke else sv["comparison_null_repeats"],
                    "--workers",
                    args.jobs,
                )
            elif stage == "hr_sensitivity":
                call("hr_sensitivity.py", "--input-dir", ready, "--output-dir", output / "hr_sensitivity")
            elif stage == "interactions":
                call(
                    "medication_interactions.py",
                    "--config",
                    args.config,
                    "--derivation",
                    derivation,
                    "--validation",
                    args.validation,
                    "--validation-phenotypes",
                    predictions,
                    "--output-dir",
                    output / "interactions",
                )
            elif stage == "competing":
                for name, source in [("derivation", derivation), ("validation", validation)]:
                    call(
                        "competing_risk_gray.py",
                        "--cohort",
                        source,
                        "--phenotypes",
                        source,
                        "--output-dir",
                        output / "competing" / name,
                        "--r-executable",
                        args.rscript,
                    )
            elif stage == "figures":
                extra = []
                if (output / "clustering" / "labels_k2.csv").exists():
                    extra += ["--k2-labels", output / "clustering" / "labels_k2.csv"]
                if (output / "competing" / "derivation" / "gray_tests.csv").exists():
                    extra += [
                        "--derivation-gray-tests",
                        output / "competing" / "derivation" / "gray_tests.csv",
                        "--validation-gray-tests",
                        output / "competing" / "validation" / "gray_tests.csv",
                    ]
                call(
                    "figures.py",
                    "--config",
                    args.config,
                    "--derivation",
                    derivation,
                    "--validation",
                    args.validation,
                    "--phenotypes",
                    predictions,
                    "--output-dir",
                    output / "figures",
                    *extra,
                )
            elif stage == "calibration":
                call(
                    "calibration.py",
                    "--input-dir",
                    ready,
                    "--output-dir",
                    output / "calibration",
                    "--bootstrap",
                    50 if smoke else sv["calibration_bootstrap_repeats"],
                )
            elif stage == "tables":
                call("tables.py", "--run-dir", output, "--config", args.config)
            manifest["stages_completed"].append(stage)
            save()
        manifest["status"] = "completed"
        save()
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save()
        raise
    print(f"Finished ({args.mode}): {output}")


if __name__ == "__main__":
    main()

"""Grouped 36-month calibration of Model C: Kaplan-Meier (composite) and Aalen-Johansen (hard MACE) observed risk."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import competing_cox as engine
from survival import fit_models

HORIZON = 36.0
BINS = {
    ("composite", "derivation"): 10,
    ("composite", "validation"): 8,
    ("hard", "derivation"): 5,
    ("hard", "validation"): 5,
}


def hard_event_risk(models: list[engine.CoxPHFitter], frame: pd.DataFrame, horizon: float = HORIZON) -> np.ndarray:
    hard, soft = models
    times, cif, _ = engine.cumulative_incidence(hard, soft, frame, horizon)
    return cif[:, -1] if len(times) else np.zeros(len(frame))


def weighted_aalen_johansen(time, status, weights, horizon: float = HORIZON) -> float:
    time, status, weights = np.asarray(time), np.asarray(status), np.asarray(weights)
    order = np.argsort(time, kind="stable")
    time, status, weights = time[order], status[order], weights[order]
    unique, start = np.unique(time, return_index=True)
    total = np.add.reduceat(weights, start)
    hard = np.add.reduceat(weights * (status == 1), start)
    events = np.add.reduceat(weights * (status != 0), start)
    at_risk = np.cumsum(total[::-1])[::-1]
    all_fraction = np.divide(events, at_risk, out=np.zeros_like(events, dtype=float), where=at_risk > 0)
    hard_fraction = np.divide(hard, at_risk, out=np.zeros_like(hard, dtype=float), where=at_risk > 0)
    previous = np.r_[1.0, np.cumprod(1 - all_fraction)[:-1]]
    return float(np.sum((previous * hard_fraction)[unique <= horizon]))


def grouped_points(
    frame: pd.DataFrame, risk: np.ndarray, endpoint: str, cohort: str, repeats: int = 5000
) -> pd.DataFrame:
    groups = pd.qcut(risk, BINS[(endpoint, cohort)], labels=False, duplicates="drop")
    if pd.isna(groups).any():
        raise ValueError("Predicted risk cannot form calibration groups")
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        t = frame.loc[mask, "TIME"].to_numpy()
        s = frame.loc[mask, "MACST"].to_numpy()
        n = int(mask.sum())
        events = int(np.sum((s == 1 if endpoint == "hard" else s != 0) & (t <= HORIZON)))
        low = high = float("nan")
        if endpoint == "hard":
            observed = weighted_aalen_johansen(t, s, np.ones(n))
            if events:
                rng = np.random.default_rng(2026091100 + (0 if cohort == "derivation" else 100) + int(group))
                draws = [weighted_aalen_johansen(t, s, rng.multinomial(n, np.full(n, 1 / n))) for _ in range(repeats)]
                low, high = np.quantile(draws, [0.025, 0.975])
        else:
            km = KaplanMeierFitter().fit(t, s != 0)
            observed = 1 - float(km.predict(HORIZON))
            ci = km.confidence_interval_survival_function_
            index = np.searchsorted(ci.index, HORIZON, side="right") - 1
            if events and index >= 0:
                low, high = 1 - ci.iloc[index, 1], 1 - ci.iloc[index, 0]
        rows.append(
            {
                "cohort": cohort,
                "endpoint": endpoint,
                "horizon_months": int(HORIZON),
                "group": int(group) + 1,
                "n": n,
                "mean_predicted_risk": float(np.mean(risk[mask])),
                "observed_risk": observed,
                "observed_risk_ci_low": low,
                "observed_risk_ci_high": high,
                "zero_event": events == 0,
            }
        )
    return pd.DataFrame(rows)


def draw(points: pd.DataFrame, cohort: str, out: Path) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.05))
    orange = "#cf6929"
    for ax, endpoint, title in zip(axes, ["composite", "hard"], ["Composite MACE", "Hard MACE"]):
        q = points[points.endpoint.eq(endpoint)]
        floor = 0.65 if endpoint == "composite" else (0.22 if cohort == "derivation" else 0.25)
        upper = max(floor, float(np.nanmax(q[["mean_predicted_risk", "observed_risk_ci_high"]].to_numpy())) * 1.06)
        ax.plot([0, upper], [0, upper], ls="--", color=".6", lw=1, label="Ideal calibration")
        ax.plot(q.mean_predicted_risk, q.observed_risk, color=orange, lw=1.3, label="Grouped observed risk")
        for row in q.itertuples():
            if np.isfinite(row.observed_risk_ci_low):
                ax.vlines(
                    row.mean_predicted_risk, row.observed_risk_ci_low, row.observed_risk_ci_high, color=".55", lw=1
                )
                ax.hlines(
                    [row.observed_risk_ci_low, row.observed_risk_ci_high],
                    row.mean_predicted_risk - upper * 0.012,
                    row.mean_predicted_risk + upper * 0.012,
                    color=".55",
                    lw=1,
                )
            ax.plot(
                row.mean_predicted_risk,
                row.observed_risk,
                "o",
                ms=5,
                mec=orange,
                mfc="white" if row.zero_event else orange,
                clip_on=False,
                zorder=4,
            )
        ax.set(
            xlim=(0, upper),
            ylim=(0, upper),
            xlabel="Mean predicted 36-month risk",
            ylabel="Observed 36-month risk",
            title=("a  " if endpoint == "composite" else "b  ") + f"{title} (N = {q.n.sum()})",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.5, -0.24, "Group n: " + ", ".join(map(str, q.n)), transform=ax.transAxes, ha="center", fontsize=7)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.25, wspace=0.36)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{cohort}_calibration_36m.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("At least two bootstrap samples are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cohort in ["derivation", "validation"]:
        engine.init(cohort, args.input_dir)
        frame = engine.D
        composite_model = fit_models(frame, "composite_event", engine.PENALTY)["C"]
        composite_risk = 1 - composite_model.predict_survival_function(frame, times=[HORIZON]).iloc[0].to_numpy()
        hard_risk = hard_event_risk(engine.fit(frame)[2], frame)
        points = pd.concat(
            [
                grouped_points(frame, composite_risk, "composite", cohort, args.bootstrap),
                grouped_points(frame, hard_risk, "hard", cohort, args.bootstrap),
            ],
            ignore_index=True,
        )
        points.to_csv(args.output_dir / f"{cohort}_calibration_points.csv", index=False)
        draw(points, cohort, args.output_dir)


if __name__ == "__main__":
    main()

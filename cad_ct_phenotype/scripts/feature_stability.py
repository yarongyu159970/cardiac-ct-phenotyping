"""Repeat the LASSO-Boruta consensus ranking inside each cross-validation training fold and summarise rank stability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import RepeatedStratifiedKFold

from candidate_features import dummy_encode_candidates, lasso_boruta_ranking, term_mapping
from common import load_config, load_table, require_columns, semantic_numeric, validate_patient_table

FINAL_COLOR = "#B64342"
OTHER_COLOR = "#B8B8B8"
POINT_COLOR = "#4D4D4D"
ACCENT_COLOR = "#3775BA"
TOP_N = 13


def summarize(long_table: pd.DataFrame, final_features: set[str]) -> pd.DataFrame:
    summary = (
        long_table.groupby("feature", as_index=False)
        .agg(
            median_consensus_rank=("consensus_rank", "median"),
            q1_consensus_rank=("consensus_rank", lambda x: x.quantile(0.25)),
            q3_consensus_rank=("consensus_rank", lambda x: x.quantile(0.75)),
            mean_consensus_rank=("consensus_rank", "mean"),
            top13_count=("top13", "sum"),
            top13_frequency=("top13", "mean"),
            boruta_confirmed_frequency=("boruta_confirmed", "mean"),
            mean_lasso_max_importance=("lasso_max_absolute_coefficient", "mean"),
        )
        .sort_values(["median_consensus_rank", "mean_consensus_rank", "feature"])
        .reset_index(drop=True)
    )
    summary.insert(1, "in_final_feature_set", summary["feature"].isin(final_features))
    return summary


def style_axis(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", colors="#333333", width=0.7, length=2.5)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, zorder=0)


def rank_panel(
    ax: plt.Axes,
    long_table: pd.DataFrame,
    summary: pd.DataFrame,
    features: list[str],
    final: set[str],
    title: str,
    label: str,
) -> None:
    ordered = summary.set_index("feature").loc[features].reset_index()
    values = [long_table.loc[long_table["feature"].eq(f), "consensus_rank"].to_numpy() for f in features]
    positions = np.arange(len(features))
    box = ax.boxplot(
        values,
        vert=True,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.0},
        whiskerprops={"color": "#777777", "linewidth": 0.75},
        capprops={"color": "#777777", "linewidth": 0.75},
        boxprops={"edgecolor": "#777777", "linewidth": 0.75},
    )
    for patch, feature in zip(box["boxes"], features):
        patch.set_facecolor(FINAL_COLOR if feature in final else OTHER_COLOR)
        patch.set_alpha(0.85 if feature in final else 0.65)
    rng = np.random.default_rng(20260905)
    for pos, vals in enumerate(values):
        ax.scatter(
            pos + rng.uniform(-0.15, 0.15, size=len(vals)),
            vals,
            s=4.5,
            color=POINT_COLOR,
            alpha=0.32,
            linewidths=0,
            zorder=3,
        )
    ax.axhline(TOP_N + 0.5, color=ACCENT_COLOR, linestyle=(0, (3, 2)), linewidth=0.9)
    total = int(long_table["split"].nunique())
    small = len(features) > 25
    for pos, row in ordered.iterrows():
        ax.text(
            pos,
            -1.2,
            f"{int(row['top13_count'])}/{total}",
            fontsize=4.7 if small else 5.4,
            va="bottom",
            ha="center",
            rotation=90,
            color=FINAL_COLOR if row["in_final_feature_set"] else "#555555",
            fontweight="bold" if row["in_final_feature_set"] else "normal",
        )
    ax.text(
        1.0,
        1.02,
        f"Top-{TOP_N} frequency (count/{total})",
        transform=ax.transAxes,
        fontsize=5.8,
        ha="right",
        va="bottom",
        color="#333333",
        fontweight="bold",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(features, fontsize=4.7 if small else 5.5, rotation=68, ha="right", rotation_mode="anchor")
    for tick, feature in zip(ax.get_xticklabels(), features):
        if feature in final:
            tick.set_color(FINAL_COLOR)
            tick.set_fontweight("bold")
    ax.set_ylim(int(long_table["consensus_rank"].max()) + 2, -8)
    ax.set_ylabel("Consensus rank across training folds\n(lower is more important)")
    ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=5)
    ax.text(-0.14, 1.015, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    style_axis(ax)


def make_figure(long_table: pd.DataFrame, summary: pd.DataFrame, final: set[str], output_dir: Path) -> list[str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6,
            "axes.linewidth": 0.7,
        }
    )
    all_features = summary["feature"].tolist()
    vessel = ["CT-FFR-LAD", "CT-FFR-LcX", "CT-FFR-RCA"] + [
        f"{t}_{m}" for m in ["MBF", "MBV", "PCBV", "TTP", "FE"] for t in ["LAD", "LCx", "RCA"]
    ]
    vessel = [f for f in vessel if f in all_features]
    medians = summary.set_index("feature")["median_consensus_rank"]
    vessel = sorted(vessel, key=lambda f: (float(medians.loc[f]), f))
    fig = plt.figure(figsize=(14.5, 10.5))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1.45, 1.0], left=0.055, right=0.99, bottom=0.125, top=0.925, hspace=0.62
    )
    rank_panel(
        fig.add_subplot(grid[0, 0]),
        long_table,
        summary,
        all_features,
        final,
        f"All {len(all_features)} candidate features",
        "a",
    )
    rank_panel(
        fig.add_subplot(grid[1, 0]),
        long_table,
        summary,
        vessel,
        final,
        "Vessel-territory CT-FFR and perfusion features",
        "b",
    )
    legend = [
        Line2D([0], [0], color=FINAL_COLOR, lw=6, label=f"Final {TOP_N} features"),
        Line2D([0], [0], color=OTHER_COLOR, lw=6, label="Other candidates"),
        Line2D([0], [0], color=ACCENT_COLOR, lw=1, linestyle=(0, (3, 2)), label=f"Top-{TOP_N} rank threshold"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=3, frameon=False, fontsize=6)
    fig.suptitle(
        "Stability of consensus feature ranking across repeated cross-validation",
        x=0.5,
        y=0.992,
        fontsize=9,
        fontweight="bold",
    )
    base = output_dir / "feature_selection_stability"
    outputs = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--boruta-max-iter", type=int, default=100)
    parser.add_argument("--boruta-n-estimators", default="auto")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()
    started = time.time()

    config = load_config(args.config)
    projection = config["projection"]
    selection = config["feature_selection"]
    frame = load_table(args.input)
    validate_patient_table(frame, config["endpoint"]["id"])
    label_column = projection["label_column"]
    require_columns(frame, [label_column], "Derivation table")
    labels = semantic_numeric(frame[label_column])
    if labels.isna().any():
        raise ValueError(f"{label_column} contains missing values")
    y = labels.astype(int).to_numpy()
    drop_first = bool(selection["dummy_drop_first"])
    x, original, categorical = dummy_encode_candidates(frame, int(selection["categorical_max_unique"]), drop_first)
    mapping = term_mapping(original, categorical, drop_first)
    final = set(projection["features"])
    if not final.issubset(original.columns):
        raise KeyError(f"Final features absent from the candidate matrix: {sorted(final - set(original.columns))}")

    folds = args.folds or int(projection["cv_folds"])
    repeats = args.repeats or int(projection["cv_repeats"])
    seed = int(projection["random_state"])
    splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(x, y))
    if args.max_splits is not None:
        splits = splits[: args.max_splits]
    n_estimators = "auto" if args.boruta_n_estimators == "auto" else int(args.boruta_n_estimators)

    rows, split_rows = [], []
    for index, (train, test) in enumerate(splits, start=1):
        split_started = time.time()
        result, _, convergence_warnings = lasso_boruta_ranking(
            x.iloc[train], y[train], projection, seed + index, mapping, args.boruta_max_iter, n_estimators, args.jobs
        )
        result["top13"] = result["consensus_rank"] <= TOP_N
        result.insert(0, "split", index)
        rows.append(result)
        split_rows.append(
            {
                "split": index,
                "train_n": len(train),
                "test_n": len(test),
                **{f"train_cluster_{c}": int(np.sum(y[train] == c)) for c in (1, 2, 3)},
                "lasso_convergence_warnings": convergence_warnings,
                "elapsed_seconds": time.time() - split_started,
            }
        )
        print(f"Completed split {index}/{len(splits)} in {split_rows[-1]['elapsed_seconds']:.1f}s", flush=True)

    long_table = pd.concat(rows, ignore_index=True)
    full_ranking, _, _ = lasso_boruta_ranking(
        x, y, projection, seed, mapping, args.boruta_max_iter, n_estimators, args.jobs
    )
    full_ranking = full_ranking.sort_values(["consensus_rank_sum", "feature"], kind="stable")
    full_ranking["selected_top13"] = full_ranking["consensus_rank"] <= TOP_N
    full_ranking["in_final_feature_set"] = full_ranking["feature"].isin(final)
    ranked_top = set(full_ranking.loc[full_ranking["selected_top13"], "feature"])
    if ranked_top != final and bool(selection.get("require_top_n_membership_match", False)):
        raise RuntimeError(
            f"Consensus top-{TOP_N} differs from the configured feature set: extra={sorted(ranked_top - final)}, missing={sorted(final - ranked_top)}"
        )

    summary = summarize(long_table, final)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(args.output_dir / "feature_ranks_by_split.csv", index=False)
    summary.to_csv(args.output_dir / "feature_rank_stability_summary.csv", index=False)
    full_ranking.to_csv(args.output_dir / "full_derivation_feature_ranking.csv", index=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "split_run_metadata.csv", index=False)
    figure_files = make_figure(long_table, summary, final, args.output_dir)
    settings = {
        "n_patients": len(frame),
        "cluster_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "n_source_candidates": int(original.shape[1]),
        "n_encoded_terms": int(x.shape[1]),
        "categorical_features": categorical,
        "dummy_drop_first": drop_first,
        "folds": folds,
        "repeats": repeats,
        "completed_splits": len(splits),
        "random_state": seed,
        "boruta_max_iter": args.boruta_max_iter,
        "boruta_n_estimators": n_estimators,
        "consensus_top13": full_ranking.head(TOP_N)["feature"].tolist(),
        "final_features": list(projection["features"]),
        "same_membership_as_final": ranked_top == final,
        "elapsed_seconds": time.time() - started,
        "figure_files": figure_files,
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

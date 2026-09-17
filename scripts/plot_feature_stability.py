"""Focused rank-stability figure for the final 13 features and the vessel-territory candidates."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

TERRITORY_FEATURES = [
    "CT-FFR-LAD",
    "CT-FFR-LcX",
    "CT-FFR-RCA",
    "LAD_MBF",
    "LCx_MBF",
    "RCA_MBF",
    "LAD_PCBV",
    "LCx_PCBV",
    "RCA_PCBV",
    "LAD_TTP",
    "LCx_TTP",
    "RCA_TTP",
]
DISPLAY_LABELS = {
    "whole lesion volume": "TPV",
    "low attenuation volume": "LAPV",
    "CT-FFR-LAD": "CT-FFR_LAD",
    "CT-FFR-LcX": "CT-FFR_LCx",
    "CT-FFR-RCA": "CT-FFR_RCA",
    "FG": "Fasting glucose",
    "TC": "Total cholesterol",
}
RED = "#B84A48"
GREY = "#C7C7C7"
TOP_N = 13


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_table = pd.read_csv(args.source_dir / "feature_ranks_by_split.csv")
    full = pd.read_csv(args.source_dir / "full_derivation_feature_ranking.csv").sort_values("consensus_rank")
    selected = full.loc[full["in_final_feature_set"], "feature"].tolist()
    n_splits = int(long_table["split"].nunique())
    ranks = {}
    for feature in set(selected + TERRITORY_FEATURES):
        block = long_table[long_table["feature"].eq(feature)].sort_values("split")
        if len(block) != n_splits:
            raise ValueError(f"Incomplete fold coverage for {feature}")
        ranks[feature] = block["consensus_rank"].to_numpy(int)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.87, hspace=0.66)
    fig.suptitle("Stability of consensus feature ranking", fontsize=11, fontweight="bold", y=0.985)
    fig.legend(
        handles=[
            Patch(facecolor=RED, edgecolor="#444444", label=f"Final {TOP_N} features"),
            Patch(facecolor=GREY, edgecolor="#666666", label="Other territory candidates"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.954),
        ncol=2,
        fontsize=8,
    )
    max_rank = int(long_table["consensus_rank"].max())
    for ax, features, title, panel in zip(
        axes, [selected, TERRITORY_FEATURES], [f"Final {TOP_N} features", "Vessel-territory comparison"], ["a", "b"]
    ):
        ax.text(-0.07, 1.105, panel, transform=ax.transAxes, fontweight="bold", fontsize=11)
        ax.set_title(title, loc="left", pad=16, fontweight="bold")
        ax.text(
            1,
            1.08,
            f"Top-{TOP_N} frequency: count / {n_splits} (%)",
            ha="right",
            transform=ax.transAxes,
            fontsize=7,
            color="#444444",
        )
        ax.axhline(TOP_N, color="#6089A5", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        for i, feature in enumerate(features, 1):
            values = ranks[feature]
            color = RED if feature in selected else GREY
            ax.boxplot(
                values,
                positions=[i],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                medianprops={"color": "#333333", "linewidth": 1.1},
                whiskerprops={"color": "#777777", "linewidth": 0.7},
                capprops={"color": "#777777", "linewidth": 0.7},
                boxprops={"facecolor": color, "edgecolor": "#666666", "linewidth": 0.7},
            )
            seed = int(hashlib.sha256(feature.encode()).hexdigest()[:8], 16)
            jitter = np.random.default_rng(seed).uniform(-0.19, 0.19, len(values))
            ax.scatter(i + jitter, values, s=5, color="#333333", alpha=0.35, zorder=3, linewidths=0)
            count = int(np.sum(values <= TOP_N))
            ax.text(
                i,
                -4.4,
                f"{count}/{n_splits}\n{100 * count / n_splits:.0f}%",
                ha="center",
                va="center",
                fontsize=6.8,
                color=RED if feature in selected else "#555555",
                linespacing=1.25,
            )
        ax.set_xlim(0.4, len(features) + 0.6)
        ax.set_ylim(max_rank + 1, -8)
        ax.set_yticks([1] + list(range(10, max_rank + 1, 10)))
        ax.set_ylabel("Consensus rank\n(lower = higher rank)")
        ax.set_xticks(
            range(1, len(features) + 1),
            [DISPLAY_LABELS.get(f, f) for f in features],
            rotation=48,
            ha="right",
            rotation_mode="anchor",
            fontsize=7,
        )
        for tick, feature in zip(ax.get_xticklabels(), features):
            tick.set_color(RED if feature in selected else "#555555")
        ax.tick_params(axis="both", length=2.5, width=0.6)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        if panel == "b":
            for position in [3.5, 6.5, 9.5]:
                ax.axvline(position, color="#D8D8D8", linewidth=0.65, zorder=0)
    fig.text(
        0.09,
        0.025,
        f"Ranks are taken from the full candidate set; the dashed line marks rank {TOP_N}.",
        fontsize=7,
        color="#555555",
    )
    stem = args.output_dir / "feature_stability_focused"
    for ext in ["png", "pdf", "svg", "tiff"]:
        kwargs = {"dpi": 200 if ext == "png" else 600}
        if ext == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(f"{stem}.{ext}", **kwargs)
    plt.close(fig)

    rank_lookup = dict(zip(full["feature"], full["consensus_rank"]))
    rows = []
    for panel, features in [("a", selected), ("b", TERRITORY_FEATURES)]:
        for feature in features:
            count = int(np.sum(ranks[feature] <= TOP_N))
            rows.append(
                {
                    "panel": panel,
                    "feature": feature,
                    "final_feature": feature in selected,
                    "full_derivation_rank": rank_lookup[feature],
                    "top13_count": count,
                    "top13_frequency": count / n_splits,
                }
            )
    pd.DataFrame(rows).to_csv(args.output_dir / "displayed_feature_summary.csv", index=False)


if __name__ == "__main__":
    main()

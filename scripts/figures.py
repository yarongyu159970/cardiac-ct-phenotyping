"""Main and supplementary figures: KM/forest panels, radar chart, phenotype heatmaps, CIF curves, OOF diagnostics."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from sklearn.preprocessing import StandardScaler

from common import load_config, load_table, projection_features, semantic_numeric, territory_mean

CLUSTER_COLORS = {1: "#E64B35", 2: "#4DBBD5", 3: "#00A087"}
FOREST_COLORS = {2: "#E64B35", 3: "#4DBBD5"}
MODEL_C_FORMULA = "age + C(gender) + SBP + C(Anti2) + TC + HDL + C(smoking) + C(DM) + C(CADRADS) + C(Cluster)"
PENALIZER = 0.01
RISK_TIMES = np.arange(0, 61, 10, dtype=float)
KEY_COLUMNS = ("MACST", "TIME", "age", "SBP", "TC", "HDL", "CAD-RADS")
ENDPOINT_SPECS = [
    ("hard_mace", "hard_event", "Hard MACE", "Hard MACE-Free Survival"),
    ("composite_mace", "composite_event", "Composite MACE", "MACE-Free Survival"),
]


def set_article_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_base: str | Path, dpi: int = 300) -> list[Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".svg", ".pdf", ".png"):
        fig.savefig(base.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
        paths.append(base.with_suffix(suffix))
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    paths.append(base.with_suffix(".tiff"))
    return paths


def _finish(data: pd.DataFrame) -> pd.DataFrame:
    data["Cluster"] = semantic_numeric(data["Cluster"]).astype(int)
    data["CADRADS"] = data["CAD-RADS"].astype(int)
    data["hard_event"] = data["MACST"].eq(1)
    data["composite_event"] = data["MACST"].ne(0)
    return data


def prepare_derivation(path: str | Path) -> pd.DataFrame:
    data = load_table(path).copy()
    needed = {"ID", "Cluster", "gender", "Anti2", "smoking", "DM", *KEY_COLUMNS}
    missing = sorted(needed.difference(data.columns))
    if missing:
        raise ValueError(f"Derivation table is missing figure variables: {missing}")
    for column in KEY_COLUMNS:
        data[column] = semantic_numeric(data[column]).astype(float)
    if data[["ID", "MACST", "TIME", "CAD-RADS", "Cluster"]].isna().any().any():
        raise ValueError("ID, MACST, TIME, CAD-RADS and Cluster must be complete")
    return _finish(data)


def prepare_external(path: str | Path, phenotype_path: str | Path) -> pd.DataFrame:
    data = load_table(path).copy()
    phenotypes = load_table(phenotype_path)
    columns = ["ID", "Cluster", "P_Cluster1", "P_Cluster2", "P_Cluster3", "Max_probability"]
    missing = sorted(set(columns).difference(phenotypes.columns))
    if missing:
        raise ValueError(f"External phenotype table is missing: {missing}")
    data = data.drop(columns=columns[1:], errors="ignore").merge(
        phenotypes[columns], on="ID", how="left", validate="one_to_one"
    )
    for column in KEY_COLUMNS:
        data[column] = semantic_numeric(data[column]).astype(float)
    if data[["ID", "MACST", "TIME", "CAD-RADS", "Cluster"]].isna().any().any():
        raise ValueError("External figure inputs contain missing key variables")
    return _finish(data)


def fit_model_c(data: pd.DataFrame, event_column: str) -> CoxPHFitter:
    return CoxPHFitter(penalizer=PENALIZER).fit(
        data, duration_col="TIME", event_col=event_column, formula=MODEL_C_FORMULA
    )


def format_p(value: float) -> str:
    return "P <0.001" if value < 0.001 else f"P ={value:.3f}"


def pairwise_p_values(data: pd.DataFrame, event_column: str) -> dict[tuple[int, int], float]:
    result = pairwise_logrank_test(data["TIME"], data["Cluster"], data[event_column])
    return {pair: float(result.summary.loc[pair, "p"]) for pair in ((1, 2), (1, 3), (2, 3))}


def risk_table(data: pd.DataFrame, event_column: str, times: np.ndarray = RISK_TIMES) -> pd.DataFrame:
    records = []
    for cluster in (1, 2, 3):
        subset = data.loc[data["Cluster"].eq(cluster)]
        for time in times:
            before_or_at = subset["TIME"].le(time)
            records.append(
                {
                    "Cluster": cluster,
                    "time": float(time),
                    "at_risk": int(subset["TIME"].ge(time).sum()),
                    "censored": int((before_or_at & ~subset[event_column].astype(bool)).sum()),
                    "events": int((before_or_at & subset[event_column].astype(bool)).sum()),
                }
            )
    return pd.DataFrame(records)


def draw_risk_table(ax: plt.Axes, table: pd.DataFrame, xmax: float) -> None:
    ax.set_xlim(-6.0, xmax)
    ax.set_ylim(-0.7, 11.6)
    ax.axis("off")
    for cluster, y0 in zip((1, 2, 3), (9.8, 5.9, 2.0)):
        ax.text(-1.6, y0 + 0.8, f"Cluster {cluster}", ha="right", va="center", fontsize=6.6)
        for label, offset in (("At risk", 0.0), ("Censored", -0.8), ("Events", -1.6)):
            ax.text(-1.6, y0 + offset, label, ha="right", va="center", fontsize=6.2)
        for row in table.loc[table["Cluster"].eq(cluster)].itertuples(index=False):
            ax.text(row.time, y0, str(row.at_risk), ha="center", va="center", fontsize=6.2)
            ax.text(row.time, y0 - 0.8, str(row.censored), ha="center", va="center", fontsize=6.2)
            ax.text(row.time, y0 - 1.6, str(row.events), ha="center", va="center", fontsize=6.2)


def draw_km_panel(ax: plt.Axes, data: pd.DataFrame, event_column: str, title: str, ylabel: str, xmax: float):
    global_test = multivariate_logrank_test(data["TIME"], data["Cluster"], data[event_column])
    for cluster in (1, 2, 3):
        subset = data["Cluster"].eq(cluster)
        km = KaplanMeierFitter(label=f"Cluster {cluster}").fit(data.loc[subset, "TIME"], data.loc[subset, event_column])
        km.plot_survival_function(
            ax=ax, ci_show=True, color=CLUSTER_COLORS[cluster], linewidth=1.7, ci_alpha=0.16, show_censors=False
        )
    ax.set(
        xlim=(0, xmax),
        ylim=(0, 1.02),
        xlabel="Follow-up Time (months)",
        ylabel=ylabel,
        title=f"{title}\nLog-rank {format_p(float(global_test.p_value))}",
    )
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.24), frameon=True, edgecolor="0.5")
    pairwise = pairwise_p_values(data, event_column)
    lines = ["Pairwise:"] + [f"{a} vs {b}: {format_p(pairwise[(a, b)])}" for a, b in ((1, 2), (1, 3), (2, 3))]
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.55"},
    )
    return float(global_test.p_value), pairwise


def cluster_hr_rows(model: CoxPHFitter, endpoint: str, cohort: str, threshold: float | None = None) -> pd.DataFrame:
    records = []
    for cluster in (2, 3):
        row = model.summary.loc[f"C(Cluster)[T.{cluster}]"]
        records.append(
            {
                "cohort": cohort,
                "endpoint": endpoint,
                "threshold": threshold,
                "comparison": f"{cluster} vs 1",
                "hr": float(row["exp(coef)"]),
                "ci_low": float(row["exp(coef) lower 95%"]),
                "ci_high": float(row["exp(coef) upper 95%"]),
                "p_value": float(row["p"]),
            }
        )
    return pd.DataFrame(records)


def draw_forest(ax: plt.Axes, rows: pd.DataFrame, title: str) -> None:
    ax.axvline(1, color="0.2", linestyle="--", linewidth=0.8)
    ax.scatter([1], [2], marker="D", s=35, color="0.5", zorder=3)
    ax.text(1.03, 2, "Reference", va="center", fontsize=7)
    for y, cluster in ((1, 2), (0, 3)):
        row = rows.loc[rows["comparison"].eq(f"{cluster} vs 1")].iloc[0]
        ax.errorbar(
            row.hr,
            y,
            xerr=[[row.hr - row.ci_low], [row.ci_high - row.hr]],
            fmt="o",
            color=FOREST_COLORS[cluster],
            capsize=4,
            linewidth=1.4,
            markersize=5,
        )
        ax.text(
            row.ci_high,
            y,
            f"HR {row.hr:.2f} ({row.ci_low:.2f}-{row.ci_high:.2f}), {format_p(row.p_value)}",
            va="center",
            ha="left",
            fontsize=6.8,
        )
    upper = max(2.2, float(rows["ci_high"].max()) * 1.55)
    ax.set(
        xlim=(0, upper),
        ylim=(-0.2, 2.2),
        yticks=[2, 1, 0],
        yticklabels=["1 (Ref)", "2 vs 1", "3 vs 1"],
        xlabel="Hazard Ratio (95% CI)",
        title=title,
    )
    ax.grid(False)


def plot_km_forest_figure(data: pd.DataFrame, cohort: str, output_base: str | Path) -> list[Path]:
    """KM curves with risk tables and Model C phenotype forest plots for one cohort."""
    set_article_style()
    cohort_label = "Derivation Cohort" if cohort == "derivation" else "Validation Cohort"
    xmax = 68.0
    fig = plt.figure(figsize=(13.3, 8.1))
    grid = fig.add_gridspec(3, 2, height_ratios=[2.35, 0.95, 1.45], hspace=0.15, wspace=0.10)
    hr_tables, risk_tables, tests = [], [], []
    for column, (endpoint, event_column, short_title, ylabel) in enumerate(ENDPOINT_SPECS):
        global_p, pairwise = draw_km_panel(
            fig.add_subplot(grid[0, column]), data, event_column, f"{cohort_label} - {short_title}", ylabel, xmax
        )
        tests.append(
            {
                "cohort": cohort,
                "endpoint": endpoint,
                "global_p": global_p,
                **{f"p_{a}_vs_{b}": pairwise[(a, b)] for a, b in ((1, 2), (1, 3), (2, 3))},
            }
        )
        risk = risk_table(data, event_column)
        risk["cohort"] = cohort
        risk["endpoint"] = endpoint
        risk_tables.append(risk)
        draw_risk_table(fig.add_subplot(grid[1, column]), risk, xmax)
        hr = cluster_hr_rows(fit_model_c(data, event_column), endpoint, cohort)
        hr_tables.append(hr)
        draw_forest(fig.add_subplot(grid[2, column]), hr, f"{cohort_label} - {short_title}")
    for label, x, y in (("A", 0.012, 0.59), ("B", 0.505, 0.59), ("C", 0.012, 0.035), ("D", 0.505, 0.035)):
        fig.text(x, y, label, fontsize=25, weight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07)
    paths = save_figure(fig, output_base)
    plt.close(fig)
    base = Path(output_base)
    pd.concat(hr_tables, ignore_index=True).to_csv(base.with_name(base.name + "_model_c_hr.csv"), index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(base.with_name(base.name + "_risk_tables.csv"), index=False)
    pd.DataFrame(tests).to_csv(base.with_name(base.name + "_logrank.csv"), index=False)
    return paths


def plot_radar_figure(data: pd.DataFrame, feature_order: list[str], output_base: str | Path) -> list[Path]:
    """Min-max scaled cluster means of the 13 classifier features."""
    set_article_style()
    x = projection_features(data, feature_order).copy()
    means = x.assign(Cluster=data["Cluster"].to_numpy()).groupby("Cluster").mean()
    scaled = ((means - means.min(axis=0)) / (means.max(axis=0) - means.min(axis=0))).fillna(0.5)
    display_order = [
        "whole lesion volume",
        "CT-FFR-LAD",
        "LAD_TTP",
        "FG",
        "HbA1c",
        "low attenuation volume",
        "IMV",
        "LAD_PCBV",
        "RCA_PCBV",
        "CT-FFR-LcX",
        "TC",
        "LAD_MBF",
        "CT-FFR-RCA",
    ]
    display_labels = [
        "Total plaque volume",
        "CT-FFR_LAD",
        "LAD_TTP",
        "Fasting Glucose",
        "HbA1c",
        "Low attenuation plaque volume",
        "IMV",
        "LAD_PCBV",
        "RCA_PCBV",
        "CT-FFR_LCx",
        "TC",
        "LAD_MBF",
        "CT-FFR_RCA",
    ]
    scaled = scaled[display_order]
    theta = np.linspace(0, 2 * np.pi, len(display_order), endpoint=False)
    theta_closed = np.r_[theta, theta[0]]
    fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(0)
    ax.set_theta_direction(-1)
    for cluster in (1, 2, 3):
        values = np.r_[scaled.loc[cluster].to_numpy(float), scaled.loc[cluster].iloc[0]]
        ax.plot(theta_closed, values, color=CLUSTER_COLORS[cluster], linewidth=2, label=f"Cluster {cluster}")
        ax.fill(theta_closed, values, color=CLUSTER_COLORS[cluster], alpha=0.18)
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    for angle, label in zip(theta, display_labels):
        cosine = np.cos(angle)
        horizontal = "left" if cosine > 0.2 else ("right" if cosine < -0.2 else "center")
        ax.text(angle, 1.10, label, fontsize=8, ha=horizontal, va="center", clip_on=False)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)], fontsize=7)
    ax.set_title("Phenotypic Radar Chart (Selected Features)", pad=34, fontsize=12, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.14), frameon=False)
    fig.subplots_adjust(left=0.15, right=0.85, top=0.86, bottom=0.14)
    paths = save_figure(fig, output_base)
    plt.close(fig)
    base = Path(output_base)
    means.to_csv(base.with_name(base.name + "_cluster_means.csv"))
    scaled.to_csv(base.with_name(base.name + "_minmax_scaled.csv"))
    return paths


def phenotype_heatmap_table(
    data: pd.DataFrame, cluster_column: str = "Cluster", cluster_values: tuple[int, ...] = (1, 2, 3)
) -> pd.DataFrame:
    """Cluster means of z-scored continuous variables and one-hot categorical levels."""
    continuous: dict[str, pd.Series] = {}
    for metric in ("MBF", "MBV", "PCBV", "FE", "TTP"):
        for territory in ("LAD", "LCx", "RCA"):
            continuous[f"{territory}_{metric}"] = territory_mean(data, metric, territory)
    for output_name, source_name in (
        ("CT-FFR_LAD", "CT-FFR-LAD"),
        ("CT-FFR_LCx", "CT-FFR-LcX"),
        ("CT-FFR_RCA", "CT-FFR-RCA"),
        ("IMV", "IMV"),
        ("Calcified plaque volume", "calcified volume"),
        ("Low attenuation plaque volume", "low attenuation volume"),
        ("Fibrous plaque volume", "fibrotic volume"),
        ("Fibro-fatty plaque volume", "fibrous fatty volume"),
        ("Total plaque volume", "whole lesion volume"),
        ("Age", "age"),
        ("BMI", "BMI"),
        ("SBP", "SBP"),
        ("Fasting Glucose", "FG"),
        ("HbA1c", "HbA1c"),
        ("LDL", "LDL"),
        ("HDL", "HDL"),
        ("TG", "TG"),
        ("TC", "TC"),
        ("Global MBF", "Global"),
    ):
        continuous[output_name] = semantic_numeric(data[source_name])
    continuous_frame = pd.DataFrame(continuous)
    continuous_frame = continuous_frame.apply(lambda column: column.fillna(column.median()))
    z_continuous = pd.DataFrame(
        StandardScaler().fit_transform(continuous_frame), columns=continuous_frame.columns, index=continuous_frame.index
    )
    categorical_spec = [
        ("CAD-RADS", {i: f"CAD-RADS_{i}" for i in range(6)}),
        ("gender", {0: "Female", 1: "Male"}),
        ("DM", {0: "DM_No", 1: "DM_Yes"}),
        ("HTN", {0: "HTN_No", 1: "HTN_Yes"}),
        ("CCS", {1: "CCS_1", 2: "CCS_2", 3: "CCS_3"}),
        ("CACS", {1: "CACS_1", 2: "CACS_2", 3: "CACS_3", 4: "CACS_4"}),
        ("dislipidemia", {0: "Dyslipidemia_No", 1: "Dyslipidemia_Yes"}),
        ("smoking", {0: "Smoking_No", 1: "Smoking_Yes"}),
        ("Anti1", {0: "Antiplatelet_No", 1: "Antiplatelet_Yes"}),
        ("Anti2", {0: "Antihypertensive_No", 1: "Antihypertensive_Yes"}),
        ("Statin", {0: "Statin_No", 1: "Statin_Yes"}),
        ("Antidiabetic", {0: "Antidiabetic_No", 1: "Antidiabetic_Yes"}),
        ("Nitrates", {0: "Nitrates_0", 1: "Nitrates_1"}),
        ("Antiischemic", {0: "Antiischemic_No", 1: "Antiischemic_Yes"}),
        ("HRP", {0: "HRP_0", 1: "HRP_1"}),
        ("LAP", {0: "LAP_0", 1: "LAP_1"}),
        ("SC", {0: "SC_0", 1: "SC_1"}),
        ("PR", {0: "PR_0", 1: "PR_1"}),
        ("NRS", {0: "NRS_0", 1: "NRS_1"}),
    ]
    categorical_frames = []
    for source, labels in categorical_spec:
        values = semantic_numeric(data[source]).astype(int)
        dummy = pd.DataFrame({name: values.eq(level).astype(float) for level, name in labels.items()})
        categorical_frames.append(
            pd.DataFrame(StandardScaler().fit_transform(dummy), columns=dummy.columns, index=dummy.index)
        )
    all_z = pd.concat([z_continuous, *categorical_frames], axis=1)
    clusters = semantic_numeric(data[cluster_column]).astype(int)
    if set(clusters.unique()) != set(cluster_values):
        raise ValueError(f"{cluster_column} has groups {sorted(clusters.unique())}, expected {list(cluster_values)}")
    means = all_z.assign(_cluster=clusters.to_numpy()).groupby("_cluster").mean().reindex(cluster_values).T
    means.columns = [f"Cluster {value}" for value in cluster_values]
    return means


def plot_heatmap(table: pd.DataFrame, title: str, output_base: str | Path) -> list[Path]:
    set_article_style()
    fig, ax = plt.subplots(figsize=(6.1, max(15.5, 0.255 * len(table))))
    sns.heatmap(
        table,
        cmap="RdBu_r",
        center=0,
        vmin=-2,
        vmax=2,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5.4},
        linewidths=0.35,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Mean standardized value", "shrink": 0.45},
    )
    ax.set_title(title, pad=12, weight="bold")
    ax.set(xlabel="", ylabel="")
    ax.tick_params(axis="y", labelsize=6.2, rotation=0)
    ax.tick_params(axis="x", labelsize=8, rotation=0)
    fig.tight_layout()
    paths = save_figure(fig, output_base)
    plt.close(fig)
    table.to_csv(Path(output_base).with_name(Path(output_base).name + "_source_data.csv"), index_label="Variable")
    return paths


def plot_k3_heatmap(data: pd.DataFrame, output_base: str | Path) -> list[Path]:
    return plot_heatmap(phenotype_heatmap_table(data), "Phenotypic heatmap of the derivation cohort", output_base)


def plot_k2_heatmap(derivation: pd.DataFrame, k2_labels_path: str | Path, output_base: str | Path) -> list[Path]:
    labels = load_table(k2_labels_path)
    if "K2_Cluster" not in labels.columns:
        raise ValueError("K=2 label table must contain K2_Cluster")
    joined = derivation.drop(columns=["K2_Cluster"], errors="ignore").merge(
        labels[["ID", "K2_Cluster"]], on="ID", how="left", validate="one_to_one"
    )
    if joined["K2_Cluster"].isna().any() or len(joined) != len(derivation):
        raise ValueError("K=2 labels do not cover every derivation patient exactly once")
    joined["K2_Cluster"] = semantic_numeric(joined["K2_Cluster"]).astype(int)
    if set(joined["K2_Cluster"].unique()) != {1, 2}:
        raise ValueError("K=2 labels must contain groups 1 and 2")
    table = phenotype_heatmap_table(joined, cluster_column="K2_Cluster", cluster_values=(1, 2))
    return plot_heatmap(table, "Phenotypic heatmap of the K=2 solution", output_base)


def plot_competing_risk_figure(
    derivation: pd.DataFrame,
    external: pd.DataFrame,
    output_base: str | Path,
    derivation_gray_tests: str | Path | None = None,
    validation_gray_tests: str | Path | None = None,
) -> list[Path]:
    """Aalen-Johansen cumulative incidence of hard and soft MACE by phenotype, annotated with Gray's test P values."""

    def gray_p_values(path):
        if path is None:
            return {}
        table = load_table(path)
        missing = sorted({"endpoint", "p_value"}.difference(table.columns))
        if missing:
            raise ValueError(f"Gray-test table is missing columns: {missing}")
        return dict(zip(table["endpoint"], table["p_value"].astype(float)))

    gray = {
        "Derivation cohort": gray_p_values(derivation_gray_tests),
        "Validation cohort": gray_p_values(validation_gray_tests),
    }
    set_article_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    source_rows = []
    for row, (cohort, data, xmax) in enumerate(
        (("Derivation cohort", derivation, 70.0), ("Validation cohort", external, 75.0))
    ):
        for column, (event_code, event_name) in enumerate(((1, "Hard MACE"), (2, "Soft MACE"))):
            ax = axes[row, column]
            for cluster in (1, 2, 3):
                subset = data["Cluster"].eq(cluster)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    aj = AalenJohansenFitter(seed=42).fit(
                        data.loc[subset, "TIME"],
                        data.loc[subset, "MACST"].astype(int),
                        event_of_interest=event_code,
                        label=f"Cluster {cluster}",
                    )
                aj.plot_cumulative_density(
                    ax=ax, ci_show=True, color=CLUSTER_COLORS[cluster], linewidth=1.7, alpha=0.16
                )
                curve = aj.cumulative_density_.reset_index()
                curve.columns = ["time", "cumulative_incidence"]
                curve["cohort"], curve["event"], curve["Cluster"] = cohort, event_name, cluster
                source_rows.append(curve)
            p_value = gray[cohort].get("Hard_MACE" if event_code == 1 else "Soft_MACE")
            label = format_p(p_value) if p_value is not None else "not computed"
            ax.set(
                xlim=(0, xmax),
                ylim=(0, None),
                xlabel="Follow-up Time (months)",
                ylabel="Cumulative incidence",
                title=f"{event_name}\nGray's test {label}",
            )
            ax.legend(frameon=True, loc="upper left")
            ax.grid(False)
    fig.text(
        0.50, 0.975, "Derivation Cohort - Competing Risk Analysis", ha="center", va="top", fontsize=11, weight="bold"
    )
    fig.text(
        0.50, 0.495, "Validation Cohort - Competing Risk Analysis", ha="center", va="top", fontsize=11, weight="bold"
    )
    fig.text(0.012, 0.49, "A", fontsize=24, weight="bold")
    fig.text(0.012, 0.02, "B", fontsize=24, weight="bold")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.94), h_pad=5.0)
    paths = save_figure(fig, output_base)
    plt.close(fig)
    pd.concat(source_rows, ignore_index=True).to_csv(
        Path(output_base).with_name(Path(output_base).name + "_cif_source_data.csv"), index=False
    )
    return paths


def plot_oof_figure(oof: pd.DataFrame, output_base: str | Path) -> list[Path]:
    """Confusion matrix and confidence reliability from the last OOF repeat."""
    set_article_style()
    if "repeat" in oof:
        if 5 not in set(oof["repeat"]):
            raise ValueError("Repeat 5 is required")
        oof = oof.loc[oof["repeat"].eq(5)].copy()
    if "ID" in oof and oof.ID.duplicated().any():
        raise ValueError("One OOF prediction per patient is required")
    truth = oof["Cluster"].to_numpy(int)
    prediction = oof["Predicted_Cluster"].to_numpy(int)
    matrix = pd.crosstab(
        pd.Categorical(truth, categories=[1, 2, 3]), pd.Categorical(prediction, categories=[1, 2, 3]), dropna=False
    ).to_numpy()
    confidence = oof[["P_Cluster1", "P_Cluster2", "P_Cluster3"]].max(axis=1)
    groups = pd.qcut(confidence, q=8, labels=False, duplicates="drop")
    reliability = (
        pd.DataFrame({"confidence": confidence, "correct": truth == prediction, "group": groups})
        .groupby("group", observed=True)
        .agg(
            mean_predicted_confidence=("confidence", "mean"),
            observed_accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [1.2, 1]})
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=True,
        fmt="d",
        cbar=False,
        square=True,
        ax=axes[0],
        xticklabels=[1, 2, 3],
        yticklabels=[1, 2, 3],
    )
    axes[0].set(title="OOF confusion matrix (repeat 5)", xlabel="Predicted label", ylabel="True label")
    axes[1].plot([0, 1], [0, 1], "--", color="#1F77B4")
    axes[1].plot(reliability["mean_predicted_confidence"], reliability["observed_accuracy"], "o-", color="#FF7F0E")
    axes[1].set(
        xlim=(0, 1.02),
        ylim=(0, 1.02),
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        title="OOF confidence reliability (repeat 5)",
    )
    fig.tight_layout()
    paths = save_figure(fig, output_base)
    plt.close(fig)
    base = Path(output_base)
    reliability.to_csv(base.with_name(base.name + "_source_data.csv"))
    pd.DataFrame(matrix, index=["true_1", "true_2", "true_3"], columns=["pred_1", "pred_2", "pred_3"]).to_csv(
        base.with_name(base.name + "_confusion_matrix.csv")
    )
    return paths


def plot_confidence_restricted_figure(external: pd.DataFrame, output_base: str | Path) -> list[Path]:
    """Validation KM/forest panels restricted to patients with assignment probability >= 0.50, 0.60 and 0.70."""
    set_article_style()
    fig = plt.figure(figsize=(13.0, 22.0))
    grid = fig.add_gridspec(
        11, 2, height_ratios=[2.2, 0.9, 1.25, 0.38, 2.2, 0.9, 1.25, 0.38, 2.2, 0.9, 1.25], hspace=0.16, wspace=0.12
    )
    hr_tables, risk_tables, tests, km_axes, forest_axes = [], [], [], [], []
    for block, threshold in enumerate((0.50, 0.60, 0.70)):
        data = external.loc[external["Max_probability"].ge(threshold)].copy()
        for column, (endpoint, event_column, title, ylabel) in enumerate(ENDPOINT_SPECS):
            top = block * 4
            km_ax = fig.add_subplot(grid[top, column])
            km_axes.append(km_ax)
            global_p, pairwise = draw_km_panel(
                km_ax, data, event_column, f"Validation Cohort - {title} (confidence ≥ {threshold:.2f})", ylabel, 75.0
            )
            tests.append(
                {
                    "threshold": threshold,
                    "endpoint": endpoint,
                    "global_p": global_p,
                    **{f"p_{a}_vs_{b}": pairwise[(a, b)] for a, b in ((1, 2), (1, 3), (2, 3))},
                }
            )
            risk = risk_table(data, event_column)
            risk["threshold"], risk["endpoint"] = threshold, endpoint
            risk_tables.append(risk)
            draw_risk_table(fig.add_subplot(grid[top + 1, column]), risk, 75.0)
            hr = cluster_hr_rows(fit_model_c(data, event_column), endpoint, "validation", threshold)
            hr_tables.append(hr)
            forest_ax = fig.add_subplot(grid[top + 2, column])
            forest_axes.append(forest_ax)
            draw_forest(forest_ax, hr, f"Validation Cohort: {title} (confidence ≥ {threshold:.2f})")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.985, bottom=0.025)
    fig.canvas.draw()
    for block in (0, 1):
        upper_bottom = min(forest_axes[block * 2 + col].get_position().y0 for col in (0, 1))
        lower_top = max(km_axes[(block + 1) * 2 + col].get_position().y1 for col in (0, 1))
        y = (upper_bottom + lower_top) / 2
        fig.add_artist(
            plt.Line2D([0.01, 0.99], [y, y], transform=fig.transFigure, color="0.15", linestyle="--", linewidth=1.0)
        )
    for label, index in zip(("A", "B", "C"), (0, 2, 4)):
        fig.text(0.012, km_axes[index].get_position().y1 - 0.012, label, fontsize=25, weight="bold", va="top")
    paths = save_figure(fig, output_base)
    plt.close(fig)
    base = Path(output_base)
    pd.concat(hr_tables, ignore_index=True).to_csv(base.with_name(base.name + "_model_c_hr.csv"), index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(base.with_name(base.name + "_risk_tables.csv"), index=False)
    pd.DataFrame(tests).to_csv(base.with_name(base.name + "_logrank.csv"), index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "derivation", "validation", "phenotypes", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--k2-labels", type=Path)
    parser.add_argument("--derivation-gray-tests", type=Path)
    parser.add_argument("--validation-gray-tests", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derivation = prepare_derivation(args.derivation)
    validation = prepare_external(args.validation, args.phenotypes)
    config = load_config(args.config)
    plot_km_forest_figure(derivation, "derivation", args.output_dir / "Figure_2")
    plot_radar_figure(derivation, config["projection"]["features"], args.output_dir / "Figure_3")
    plot_km_forest_figure(validation, "validation", args.output_dir / "Figure_4")
    plot_k3_heatmap(derivation, args.output_dir / "Supplementary_Figure_3")
    plot_confidence_restricted_figure(validation, args.output_dir / "Supplementary_Figure_7")
    if args.k2_labels:
        plot_k2_heatmap(derivation, args.k2_labels, args.output_dir / "Supplementary_Figure_2")
    if args.derivation_gray_tests and args.validation_gray_tests:
        plot_competing_risk_figure(
            derivation,
            validation,
            args.output_dir / "Supplementary_Figure_4",
            derivation_gray_tests=args.derivation_gray_tests,
            validation_gray_tests=args.validation_gray_tests,
        )


if __name__ == "__main__":
    main()

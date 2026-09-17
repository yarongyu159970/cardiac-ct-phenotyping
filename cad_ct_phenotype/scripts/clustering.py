"""Metric-wise PCA and K-prototypes clustering with seed-repeat and bootstrap stability diagnostics."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import OrdinalEncoder

from clustering_features import build_clustering_matrix
from common import assert_outcome_independent, load_config, load_table, semantic_numeric, validate_patient_table


def stability_resample(
    n_patients: int, sampling: str, fraction: float, preserve_draw_order: bool, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return fit indices and evaluation indices for one stability replicate."""
    if n_patients < 2:
        raise ValueError("Stability analysis requires at least two patients")
    if not 0 < fraction <= 1:
        raise ValueError("stability_fraction must be in (0, 1]")
    requested_n = int(round(fraction * n_patients))
    if sampling == "subsample":
        fit_indices = generator.choice(n_patients, requested_n, replace=False)
        if not preserve_draw_order:
            fit_indices = np.sort(fit_indices)
        evaluation_indices = fit_indices.copy()
    elif sampling == "bootstrap_unique":
        fit_indices = generator.choice(n_patients, requested_n, replace=True)
        if preserve_draw_order:
            _, first_positions = np.unique(fit_indices, return_index=True)
            evaluation_indices = fit_indices[np.sort(first_positions)]
        else:
            evaluation_indices = np.unique(fit_indices)
    else:
        raise ValueError("stability_sampling must be 'subsample' or 'bootstrap_unique'")
    return fit_indices.astype(int), evaluation_indices.astype(int), requested_n


def matched_cluster_jaccard(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    """Mean Jaccard over Hungarian-matched cluster pairs."""
    matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        members = reference == i
        for j in range(k):
            other = candidate == j
            union = np.logical_or(members, other).sum()
            matrix[i, j] = np.logical_and(members, other).sum() / union if union else 0.0
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].mean())


def mean_max_cluster_jaccard(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    """Mean over reference clusters of the best-matching candidate Jaccard (clusterboot convention)."""
    best = []
    for i in range(k):
        members = reference == i
        scores = []
        for j in range(k):
            other = candidate == j
            union = np.logical_or(members, other).sum()
            scores.append(np.logical_and(members, other).sum() / union if union else 0.0)
        best.append(max(scores))
    return float(np.mean(best))


def fit_stability_replicate(
    mixed: np.ndarray,
    categorical_indices: list[int],
    reference_labels: np.ndarray,
    k: int,
    replicate: int,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    requested_n: int,
    sampling: str,
    seed: int,
    init: str,
    n_init: int,
    gamma: float | None,
) -> dict:
    model = KPrototypes(n_clusters=k, init=init, n_init=n_init, gamma=gamma, random_state=seed, verbose=0)
    model.fit_predict(mixed[fit_indices], categorical=categorical_indices)
    candidate_evaluation = model.predict(mixed[evaluation_indices], categorical=categorical_indices).astype(int)
    reference_evaluation = reference_labels[evaluation_indices]
    candidate_full = model.predict(mixed, categorical=categorical_indices).astype(int)
    return {
        "k": k,
        "bootstrap": replicate + 1,
        "seed": seed,
        "sampling": sampling,
        "requested_n": requested_n,
        "fit_sample_n": len(fit_indices),
        "unique_inbag_n": len(np.unique(fit_indices)),
        "evaluation_n": len(evaluation_indices),
        "adjusted_rand_index": adjusted_rand_score(reference_evaluation, candidate_evaluation),
        "mean_clusterwise_jaccard": mean_max_cluster_jaccard(reference_evaluation, candidate_evaluation, k),
        "full_prediction_adjusted_rand_index": adjusted_rand_score(reference_labels, candidate_full),
        "full_prediction_matched_jaccard": matched_cluster_jaccard(reference_labels, candidate_full, k),
    }


def parse_k_values(text: str | None, default: list[int]) -> list[int]:
    if text is None:
        return [int(value) for value in default]
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or min(values) < 2:
        raise ValueError("K values must be integers of at least 2")
    return values


def save_candidate_k_plot(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    silhouette = diagnostics["silhouette_ordinal_euclidean"].copy()
    fixed = diagnostics["fixed_reference_silhouette_ordinal_euclidean"]
    silhouette.loc[fixed.notna()] = fixed.loc[fixed.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].plot(diagnostics["k"], diagnostics["best_cost"], marker="o", color="#2F6B9A")
    axes[0].set(title="Elbow diagnostic", xlabel="Number of clusters (K)", ylabel="K-prototypes cost")
    axes[1].plot(diagnostics["k"], silhouette, marker="o", color="#D97925")
    axes[1].set(title="Silhouette diagnostic", xlabel="Number of clusters (K)", ylabel="Silhouette score")
    for axis in axes:
        axis.set_xticks(diagnostics["k"])
        axis.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output_dir / f"candidate_k_diagnostics{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnostic_matrix(numeric_scaled: pd.DataFrame, categorical: pd.DataFrame) -> np.ndarray:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return np.column_stack([numeric_scaled.to_numpy(dtype=float), encoder.fit_transform(categorical)])


def display_mapping(raw_labels: np.ndarray, ordering_variable: pd.Series) -> tuple[dict[int, int], pd.DataFrame]:
    """Relabel clusters 1..K by increasing mean of the ordering variable."""
    summary = (
        pd.DataFrame(
            {"raw_label": raw_labels.astype(int), "ordering_variable": semantic_numeric(ordering_variable).to_numpy()}
        )
        .groupby("raw_label", as_index=False)
        .agg(n=("raw_label", "size"), mean_ordering_variable=("ordering_variable", "mean"))
        .sort_values(["mean_ordering_variable", "raw_label"])
        .reset_index(drop=True)
    )
    summary["display_label"] = np.arange(1, len(summary) + 1)
    return dict(zip(summary["raw_label"], summary["display_label"])), summary


def reference_overlap_mapping(
    raw_labels: np.ndarray, reference_labels: np.ndarray, k: int
) -> tuple[dict[int, int], pd.DataFrame]:
    """Relabel fitted clusters to maximise overlap with a fixed reference labelling (post hoc only)."""
    expected = set(range(1, k + 1))
    observed = set(pd.Series(reference_labels).astype(int).unique())
    if observed != expected:
        raise ValueError(f"Reference labels for K={k} must be {sorted(expected)}; found {sorted(observed)}")
    counts = np.zeros((k, k), dtype=int)
    for raw in range(k):
        for reference in range(1, k + 1):
            counts[raw, reference - 1] = int(np.logical_and(raw_labels == raw, reference_labels == reference).sum())
    raw_index, reference_index = linear_sum_assignment(-counts)
    mapping = {int(r): int(c + 1) for r, c in zip(raw_index, reference_index)}
    rows = [
        {
            "raw_label": raw,
            "display_label": mapping[raw],
            "n": int((raw_labels == raw).sum()),
            "matched_reference_n": int(counts[raw, mapping[raw] - 1]),
        }
        for raw in range(k)
    ]
    return mapping, pd.DataFrame(rows).sort_values("display_label")


def kprototypes_distance_matrix(numeric_scaled: pd.DataFrame, categorical: pd.DataFrame, gamma: float) -> np.ndarray:
    numeric_distance = pairwise_distances(numeric_scaled.to_numpy(dtype=float), metric="sqeuclidean")
    values = categorical.to_numpy(dtype=object)
    categorical_distance = np.zeros_like(numeric_distance)
    for column in range(values.shape[1]):
        categorical_distance += values[:, column][:, None] != values[:, column][None, :]
    distance = numeric_distance + float(gamma) * categorical_distance
    np.fill_diagonal(distance, 0.0)
    return distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default=None, help="Comma-separated candidate K values")
    parser.add_argument("--seed-repeats", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=None)
    parser.add_argument("--bootstrap-n-init", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Thread workers for stability replicates")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--reference-column",
        default=None,
        help="Existing label column used only to align fitted labels after clustering",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    config = load_config(args.config)
    clustering = config["clustering"]
    id_column = config["endpoint"]["id"]
    frame = load_table(args.input)
    validate_patient_table(frame, id_column)
    if args.row_limit is not None:
        frame = frame.iloc[: args.row_limit].copy()

    mixed, categorical_indices, numeric_scaled, categorical, diagnostics_tables = build_clustering_matrix(
        frame, config["preprocessing"]
    )
    assert_outcome_independent([*numeric_scaled.columns, *categorical.columns])
    diagnostic = diagnostic_matrix(numeric_scaled, categorical)

    reference_labels = None
    if args.reference_column is not None:
        reference = semantic_numeric(frame[args.reference_column])
        if reference.isna().any():
            raise ValueError(f"{args.reference_column} contains missing labels")
        reference_labels = reference.astype(int).to_numpy()

    k_values = parse_k_values(args.k_values, clustering["candidate_k"])
    repeats = args.seed_repeats or int(clustering["seed_repeats"])
    n_init = args.n_init or int(clustering["n_init"])
    bootstrap_repeats = (
        int(clustering["bootstrap_repeats"]) if args.bootstrap_repeats is None else args.bootstrap_repeats
    )
    bootstrap_n_init = args.bootstrap_n_init or int(clustering["bootstrap_n_init"])
    base_seed = int(clustering["random_state"])
    seeds = [base_seed + index for index in range(repeats)]
    gamma = clustering.get("gamma")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(
        [frame[[id_column]].reset_index(drop=True), diagnostics_tables["pca_scores"].reset_index(drop=True)], axis=1
    ).to_csv(output_dir / "pca_scores.csv", index=False)
    for key, filename in (
        ("pca_report", "pca_report.csv"),
        ("feature_types", "feature_types.csv"),
        ("correlation_pairs", "high_correlation_pairs.csv"),
        ("correlation_decisions", "correlation_filter_decisions.csv"),
        ("eta_squared_detail", "eta_squared_detail.csv"),
        ("eta_squared_summary", "eta_squared_summary.csv"),
    ):
        diagnostics_tables[key].to_csv(output_dir / filename, index=False)

    diagnostic_rows = []
    mapping_tables = []
    for k in k_values:
        label_runs, costs, fitted_gammas = [], [], []
        for seed in seeds:
            model = KPrototypes(
                n_clusters=k, init=clustering["init"], n_init=n_init, gamma=gamma, random_state=seed, verbose=0
            )
            label_runs.append(model.fit_predict(mixed, categorical=categorical_indices).astype(int))
            costs.append(float(model.cost_))
            fitted_gammas.append(float(model.gamma))

        reference_seed = clustering.get("reference_seed")
        if reference_seed is None:
            reference_index = int(np.argmin(costs))
        else:
            reference_seed = int(reference_seed)
            if reference_seed not in seeds:
                raise ValueError(f"reference_seed={reference_seed} is outside the evaluated seeds {seeds}")
            reference_index = seeds.index(reference_seed)
        raw_labels = label_runs[reference_index]
        ari_values = [adjusted_rand_score(raw_labels, labels) for labels in label_runs]

        bootstrap_rows = []
        if bootstrap_repeats > 0:
            stability_seed = int(clustering.get("stability_random_state", base_seed))
            generator = np.random.default_rng(stability_seed)
            sampling = clustering.get("stability_sampling", "bootstrap_unique")
            fraction = float(clustering.get("stability_fraction", 1.0))
            preserve_order = bool(clustering.get("stability_preserve_draw_order", False))
            resamples = [
                stability_resample(len(frame), sampling, fraction, preserve_order, generator)
                for _ in range(bootstrap_repeats)
            ]

            def run_replicate(item):
                index, (fit_indices, evaluation_indices, requested_n) = item
                return fit_stability_replicate(
                    mixed,
                    categorical_indices,
                    raw_labels,
                    k,
                    index,
                    fit_indices,
                    evaluation_indices,
                    requested_n,
                    sampling,
                    stability_seed + 10000 * k + index,
                    clustering["init"],
                    bootstrap_n_init,
                    gamma,
                )

            items = list(enumerate(resamples))
            if args.jobs == 1:
                bootstrap_rows = [run_replicate(item) for item in items]
            else:
                with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                    bootstrap_rows = list(executor.map(run_replicate, items))
        bootstrap_table = pd.DataFrame(bootstrap_rows)
        if bootstrap_table.empty:
            mean_ari = mean_jaccard = mean_full_ari = mean_full_jaccard = np.nan
        else:
            bootstrap_table.to_csv(output_dir / f"bootstrap_stability_k{k}.csv", index=False)
            mean_ari = float(bootstrap_table["adjusted_rand_index"].mean())
            mean_jaccard = float(bootstrap_table["mean_clusterwise_jaccard"].mean())
            mean_full_ari = float(bootstrap_table["full_prediction_adjusted_rand_index"].mean())
            mean_full_jaccard = float(bootstrap_table["full_prediction_matched_jaccard"].mean())

        aligned = reference_labels is not None and set(np.unique(reference_labels)) == set(range(1, k + 1))
        if aligned:
            mapping, mapping_table = reference_overlap_mapping(raw_labels, reference_labels, k)
        else:
            mapping, mapping_table = display_mapping(raw_labels, frame[clustering["display_order_variable"]])
        display_labels = pd.Series(raw_labels).map(mapping).astype(int).to_numpy()
        cluster_sizes = pd.Series(display_labels).value_counts().sort_index()

        pd.DataFrame(
            {
                id_column: frame[id_column].to_numpy(),
                f"K{k}_Cluster_Raw": raw_labels + 1,
                f"K{k}_Cluster": display_labels,
            }
        ).to_csv(output_dir / f"labels_k{k}.csv", index=False)

        seed_tables = []
        for seed, cost, fitted_gamma, seed_labels in zip(seeds, costs, fitted_gammas, label_runs):
            if aligned:
                seed_mapping, _ = reference_overlap_mapping(seed_labels, reference_labels, k)
            else:
                seed_mapping, _ = display_mapping(seed_labels, frame[clustering["display_order_variable"]])
            seed_tables.append(
                pd.DataFrame(
                    {
                        id_column: frame[id_column].to_numpy(),
                        "seed": seed,
                        "cost": cost,
                        "gamma": fitted_gamma,
                        f"K{k}_Cluster_Raw": seed_labels + 1,
                        f"K{k}_Cluster": pd.Series(seed_labels).map(seed_mapping).astype(int).to_numpy(),
                    }
                )
            )
        pd.concat(seed_tables, ignore_index=True).to_csv(output_dir / f"labels_k{k}_all_seeds_long.csv", index=False)
        pd.DataFrame(
            {
                "k": k,
                "cluster": cluster_sizes.index.astype(int),
                "n": cluster_sizes.to_numpy(dtype=int),
                "percent": 100 * cluster_sizes.to_numpy(dtype=float) / len(frame),
            }
        ).to_csv(output_dir / f"membership_k{k}.csv", index=False)

        mapping_table.insert(0, "k", k)
        mapping_table["raw_label"] = mapping_table["raw_label"] + 1
        mapping_table["alignment_rule"] = (
            f"maximum overlap with {args.reference_column}"
            if aligned
            else f"increasing mean {clustering['display_order_variable']}"
        )
        mapping_tables.append(mapping_table)

        reference_ari = reference_nmi = fixed_reference_silhouette = np.nan
        if aligned:
            reference_ari = float(adjusted_rand_score(reference_labels, display_labels))
            reference_nmi = float(normalized_mutual_info_score(reference_labels, display_labels))
            fixed_reference_silhouette = float(silhouette_score(diagnostic, reference_labels))
            pd.crosstab(
                pd.Series(reference_labels, name="Reference_Cluster"),
                pd.Series(display_labels, name=f"Fitted_K{k}_Cluster"),
            ).to_csv(output_dir / f"reference_alignment_k{k}.csv")
        distance = kprototypes_distance_matrix(numeric_scaled, categorical, fitted_gammas[reference_index])
        diagnostic_rows.append(
            {
                "k": k,
                "best_seed": seeds[reference_index],
                "best_cost": costs[reference_index],
                "fitted_gamma": fitted_gammas[reference_index],
                "silhouette_kprototypes_distance": float(silhouette_score(distance, raw_labels, metric="precomputed")),
                "silhouette_ordinal_euclidean": float(silhouette_score(diagnostic, raw_labels)),
                "mean_seed_ari": float(np.mean(ari_values)),
                "minimum_seed_ari": float(np.min(ari_values)),
                "mean_bootstrap_ari": mean_ari,
                "mean_cluster_jaccard": mean_jaccard,
                "mean_full_prediction_bootstrap_ari": mean_full_ari,
                "mean_full_prediction_matched_jaccard": mean_full_jaccard,
                "min_cluster_n": int(cluster_sizes.min()),
                "min_cluster_percent": float(100 * cluster_sizes.min() / len(frame)),
                "postfit_reference_ari": reference_ari,
                "postfit_reference_nmi": reference_nmi,
                "fixed_reference_silhouette_ordinal_euclidean": fixed_reference_silhouette,
            }
        )

    diagnostics = pd.DataFrame(diagnostic_rows)
    thresholds = clustering.get("stability_thresholds", {})
    if thresholds:
        diagnostics["meets_stability_thresholds"] = (
            diagnostics["mean_bootstrap_ari"].ge(float(thresholds["mean_bootstrap_ari"]))
            & diagnostics["mean_cluster_jaccard"].ge(float(thresholds["mean_cluster_jaccard"]))
            & diagnostics["min_cluster_percent"].ge(100 * float(thresholds["minimum_cluster_fraction"]))
        )
    diagnostics["reported_primary_solution"] = diagnostics["k"].eq(int(clustering["reported_primary_k"]))
    diagnostics.to_csv(output_dir / "cluster_diagnostics.csv", index=False)
    save_candidate_k_plot(diagnostics, output_dir)
    pd.concat(mapping_tables, ignore_index=True).to_csv(output_dir / "display_label_mapping.csv", index=False)
    settings = {
        "n_patients": len(frame),
        "k_values": k_values,
        "seeds": seeds,
        "reference_seed": None if clustering.get("reference_seed") is None else int(clustering["reference_seed"]),
        "n_init": n_init,
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_n_init": bootstrap_n_init,
        "init": clustering["init"],
        "gamma": gamma,
        "stability_sampling": clustering.get("stability_sampling", "bootstrap_unique"),
        "stability_fraction": float(clustering.get("stability_fraction", 1.0)),
        "numeric_columns": numeric_scaled.columns.tolist(),
        "categorical_columns": categorical.columns.tolist(),
        "correlation_dropped_columns": diagnostics_tables["dropped_numeric"],
        "eta_squared_removed_columns": diagnostics_tables["removed_categorical"],
        "reported_primary_k": int(clustering["reported_primary_k"]),
        "reference_column": args.reference_column,
        "stability_thresholds": thresholds,
    }
    (output_dir / "clustering_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()

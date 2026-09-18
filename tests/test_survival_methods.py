import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibration import grouped_points, weighted_aalen_johansen  # noqa: E402
from competing_cox import weights  # noqa: E402
from graf_ipcw import censor_km, graf_curve, integrate_curve  # noqa: E402
from hard_mace_bootstrap import expand  # noqa: E402
from oof_summary import scores  # noqa: E402
from survival import bootstrap_c_index, fit_models, load_analysis_data  # noqa: E402
from tables import adjust_pairwise_p_values, continuous_tests, performance_table  # noqa: E402


class GrafIpcwTests(unittest.TestCase):
    def test_events_precede_censorings_at_ties(self):
        unique, g = censor_km([1, 1, 2, 3], [1, 0, 1, 0])
        np.testing.assert_array_equal(unique, [1, 2, 3])
        np.testing.assert_allclose(g, [2 / 3, 2 / 3, 0])

    def test_hand_computed_brier_with_ties(self):
        t = np.array([1, 1, 2, 3])
        e = np.array([1, 0, 1, 0])
        np.testing.assert_allclose(graf_curve(t, e, t, e, np.full((4, 2), 0.5), [1, 2]), [0.25, 0.25])

    def test_reduces_to_squared_error_without_censoring(self):
        t = np.array([1, 2, 3, 4])
        e = np.ones(4, bool)
        grid = np.array([1.0, 2.0, 3.0])
        p = np.array([[0.2, 0.1, 0.05], [0.8, 0.4, 0.2], [0.9, 0.8, 0.4], [0.99, 0.9, 0.8]])
        expected = np.mean(((t[:, None] > grid).astype(float) - p) ** 2, axis=0)
        np.testing.assert_allclose(graf_curve(t, e, t, e, p, grid), expected)

    def test_training_censoring_distribution_is_used(self):
        np.testing.assert_allclose(graf_curve([1, 3], [0, 1], [2, 3], [1, 1], np.full((2, 1), 0.5), [1]), [0.5])

    def test_integrated_score(self):
        self.assertAlmostEqual(integrate_curve([0.1, 0.2, 0.3], [3, 4, 5]), 0.2)


class CompetingRiskTests(unittest.TestCase):
    def test_aalen_johansen_treats_competing_event_as_competing(self):
        self.assertAlmostEqual(weighted_aalen_johansen([1, 2, 3, 4], [2, 1, 0, 1], [1, 1, 1, 1], 3), 0.25)
        self.assertAlmostEqual(weighted_aalen_johansen([1, 2, 3, 4], [2, 1, 0, 1], [1, 1, 1, 1], 4), 0.75)
        self.assertAlmostEqual(weighted_aalen_johansen([1, 1, 2], [1, 0, 1], [1, 1, 1], 1), 1 / 3)

    def test_hard_event_target_and_censoring_weights(self):
        frame = pd.DataFrame({"TIME": [1.0, 2.0, 50.0, 70.0], "MACST": [2, 1, 0, 0]})
        _, w, y = weights(frame, frame)[0]
        self.assertFalse(y[0, -1])
        self.assertTrue(y[1, -1])
        np.testing.assert_allclose(w[:, -1], np.ones(4))
        self.assertAlmostEqual(np.mean(w[:, -1] * (y[:, -1] - 0.25) ** 2), 0.1875)

    def test_zero_event_groups_have_no_interval(self):
        frame = pd.DataFrame({"TIME": np.arange(10) + 40.0, "MACST": np.zeros(10, int)})
        points = grouped_points(frame, np.linspace(0.01, 0.1, 10), "hard", "derivation", 10)
        self.assertTrue(points.zero_event.all())
        self.assertTrue(points.observed_risk_ci_low.isna().all())

    def test_paired_differences_are_formed_before_quantiles(self):
        samples = np.array([[[1.0], [2.0], [4.0]], [[10.0], [11.0], [13.0]]])
        x = expand(samples)
        np.testing.assert_array_equal(x[:, 3, 0], [1, 1])
        np.testing.assert_array_equal(x[:, 4, 0], [2, 2])


class SurvivalTests(unittest.TestCase):
    def test_bootstrap_c_index_is_deterministic(self):
        data = pd.DataFrame({"TIME": [1.0, 2.0, 3.0, 4.0, 5.0], "event": [True, False, True, False, True]})
        risk = np.array([0.2, 0.4, 0.6, 0.3, 0.7])
        self.assertEqual(
            bootstrap_c_index(data, "event", risk, 200, 42), bootstrap_c_index(data, "event", risk, 200, 42)
        )

    def test_fractional_categories_are_rejected(self):
        frame = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "TIME": [10, 20, 30],
                "MACET": [0, 1, 2],
                "age": [50, 60, 70],
                "SBP": [110, 120, 130],
                "TC": [4.0, 5.0, 6.0],
                "HDL": [1.0, 1.2, 1.4],
                "gender": [0, 1, 0],
                "Antihypertensive medication": [0, 1, 0],
                "smoking": [1, 0, 1],
                "DM": [0, 1, 0],
                "CAD-RADS": [0.0, 1.5, 5.0],
                "Cluster": [1, 2, 3],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cohort.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "CAD-RADS"):
                load_analysis_data(path, path)

    def test_models_fit_with_balanced_categories(self):
        rng = np.random.default_rng(0)
        n = 120
        frame = pd.DataFrame(
            {
                "ID": np.arange(n),
                "TIME": rng.uniform(3, 80, n),
                "MACET": rng.permutation(np.tile([0, 0, 1, 2], n // 4)),
                "age": rng.uniform(40, 80, n),
                "SBP": rng.uniform(100, 170, n),
                "TC": rng.uniform(3, 8, n),
                "HDL": rng.uniform(0.8, 2, n),
                "gender": rng.permutation(np.arange(n) % 2),
                "Antihypertensive medication": rng.permutation(np.arange(n) % 2),
                "smoking": rng.permutation(np.arange(n) % 2),
                "DM": rng.permutation(np.arange(n) % 2),
                "CAD-RADS": rng.permutation(np.arange(n) % 6),
                "Cluster": rng.permutation(np.arange(n) % 3 + 1),
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cohort.csv"
            frame.to_csv(path, index=False)
            data = load_analysis_data(path, path)
        self.assertEqual(data["CAD-RADS"].median(), 2.5)
        for model in fit_models(data, "composite_event", 0.01).values():
            self.assertTrue(np.isfinite(model.params_.to_numpy()).all())


class TableTests(unittest.TestCase):
    def test_multiclass_brier_sums_over_classes(self):
        frame = pd.DataFrame(
            {
                "Cluster": [1, 2, 3],
                "Predicted_Cluster": [1, 2, 3],
                "P_Cluster1": [0.8, 0.1, 0.1],
                "P_Cluster2": [0.1, 0.8, 0.1],
                "P_Cluster3": [0.1, 0.1, 0.8],
            }
        )
        self.assertAlmostEqual(scores(frame)["multiclass_brier"], 0.06)

    def test_pairwise_tests_use_holm_adjusted_mann_whitney(self):
        from scipy.stats import mannwhitneyu

        frame = pd.DataFrame(
            {
                "Cluster": [1] * 4 + [2] * 4 + [3] * 4,
                "value": [120, 120, 130, 140, 120, 130, 130, 140, 130, 140, 150, 160],
            }
        )
        _, p = continuous_tests(frame, "value", "holm")
        raw = [
            mannwhitneyu(
                frame.loc[frame.Cluster.eq(a), "value"],
                frame.loc[frame.Cluster.eq(b), "value"],
                method="asymptotic",
                use_continuity=True,
                alternative="two-sided",
            ).pvalue
            for a, b in [(1, 2), (1, 3), (2, 3)]
        ]
        np.testing.assert_allclose(p, adjust_pairwise_p_values(raw, "holm"))

    def test_performance_table_rounds_after_differencing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model_comparison").mkdir()
            (root / "hard_competing" / "intervals").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"cohort": "validation", "endpoint": "hard", "model": m, "effective_df_AIC": x}
                    for m, x in zip("ABC", [210.084, 210.566, 201.849])
                ]
            ).to_csv(root / "model_comparison" / "model_AIC_effective_df.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "cohort": "validation",
                        "endpoint": "hard",
                        "comparison": c,
                        "penalized_LR": 1.23,
                        "null_bootstrap_P": 0.123,
                    }
                    for c in ["A_vs_B", "B_vs_C"]
                ]
            ).to_csv(root / "model_comparison" / "penalized_LR_bootstrap_P.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "cohort": "validation",
                        "comparison": m,
                        "metric": k,
                        "estimate": 0.023464,
                        "lower": 0.01,
                        "upper": 0.04,
                    }
                    for m in "ABC"
                    for k in ["C-index", "Brier36", "IBS36", "Brier60", "IBS60"]
                ]
            ).to_csv(root / "hard_competing" / "intervals" / "corrected_table_long.csv", index=False)
            table = performance_table(root, "validation", "hard_mace").set_index("Metric")
        self.assertEqual(len(table), 8)
        self.assertEqual(table.loc["AIC", "A vs B"], "0.48")
        self.assertEqual(table.loc["IBS (60-month horizon)", "Model C"], "0.023")
        self.assertIn("(", table.loc["C-index (95% CI)", "Model A"])


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import (  # noqa: E402
    TERRITORIES,
    attach_phenotypes,
    load_json_model,
    load_table,
    predict_json_model,
    projection_features,
    territory_mean,
    validate_study_categories,
)

MODEL = load_json_model(ROOT / "artifacts" / "final_projection_model.json")
EXAMPLE = pd.read_csv(ROOT / "examples" / "calculator_input.csv")
EXPECTED_FEATURES = [
    "whole lesion volume",
    "CT-FFR-RCA",
    "LAD_MBF",
    "RCA_PCBV",
    "TC",
    "CT-FFR-LcX",
    "LAD_PCBV",
    "IMV",
    "low attenuation volume",
    "HbA1c",
    "LAD_TTP",
    "FG",
    "CT-FFR-LAD",
]


def predict(frame: pd.DataFrame) -> np.ndarray:
    return predict_json_model(projection_features(frame, MODEL["features"]), MODEL)[1]


class FrozenModelTests(unittest.TestCase):
    def test_feature_order(self):
        self.assertEqual(MODEL["features"], EXPECTED_FEATURES)

    def test_probabilities_are_order_invariant(self):
        expected = [[0.9349966809222637, 0.06499551096661657, 0.000007808111119691157]]
        for frame in [EXAMPLE, EXAMPLE[EXAMPLE.columns[::-1]]]:
            probabilities = predict(frame)
            np.testing.assert_allclose(probabilities.sum(1), 1)
            np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-14)

    def test_invalid_measurements_are_rejected(self):
        for field, value in [
            ("IMV", 5.0),
            ("CT-FFR-LAD", 1.1),
            ("CT-FFR-LcX", -1.0),
            ("CT-FFR-RCA", np.inf),
            ("TC", np.nan),
            ("LAD_MBF", -1.0),
            ("HbA1c", "invalid"),
        ]:
            frame = EXAMPLE.astype(object)
            frame.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                predict(frame)

    def test_missing_duplicate_and_empty_inputs_are_rejected(self):
        for frame in [
            EXAMPLE.drop(columns="LAD_TTP"),
            pd.concat([EXAMPLE, EXAMPLE[["IMV"]]], axis=1),
            EXAMPLE.iloc[:0],
        ]:
            with self.assertRaises(ValueError):
                predict(frame)

    def test_duplicate_csv_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.csv"
            pd.concat([EXAMPLE, EXAMPLE[["IMV"]]], axis=1).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate input columns"):
                load_table(path)


class TerritoryTests(unittest.TestCase):
    def test_every_segment_is_required(self):
        for territory, segments in TERRITORIES.items():
            frame = pd.DataFrame({f"{s}_MBF": [float(s), float(s + 2)] for s in segments})
            np.testing.assert_allclose(
                territory_mean(frame, "MBF", territory), [np.mean(segments), np.mean(segments) + 2]
            )
            for segment in segments:
                for value in [np.nan, np.inf, -1.0, "invalid"]:
                    bad = frame.astype(object)
                    bad.loc[1, f"{segment}_MBF"] = value
                    with self.assertRaisesRegex(ValueError, "Partial-segment averaging"):
                        territory_mean(bad, "MBF", territory)
                with self.assertRaisesRegex(ValueError, "missing required columns"):
                    territory_mean(frame.drop(columns=f"{segment}_MBF"), "MBF", territory)

    def test_complete_territory_column_takes_precedence(self):
        np.testing.assert_array_equal(territory_mean(EXAMPLE, "MBF", "LAD"), [150.0])
        frame = pd.DataFrame({f"{s}_MBF": [150.0, 160.0] for s in TERRITORIES["LAD"]})
        frame["LAD_MBF"] = [150.0, np.nan]
        np.testing.assert_array_equal(territory_mean(frame, "MBF", "LAD"), [150.0, 160.0])


class CategoryAndMergeTests(unittest.TestCase):
    def test_study_categories(self):
        frame = pd.DataFrame({k: np.arange(12) % 2 for k in ["gender", "Anti2", "smoking", "DM"]})
        frame["CAD-RADS"] = np.arange(12) % 6
        frame["Cluster"] = np.arange(12) % 3 + 1
        validate_study_categories(frame, require_all_levels=True)
        for field in frame:
            bad = frame.copy()
            bad[field] = 1
            with self.assertRaisesRegex(ValueError, f"{field}.*missing levels"):
                validate_study_categories(bad, require_all_levels=True)
            for value in [np.nan, np.inf, 0.5, 99]:
                bad = frame.astype(float)
                bad.loc[0, field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_study_categories(bad)

    def test_phenotypes_are_matched_by_id(self):
        original = pd.DataFrame({"ID": [1, 2], "Cluster": [3, 3]})
        predicted = pd.DataFrame({"ID": [2, 1], "Cluster": [2, 1]})
        self.assertEqual(attach_phenotypes(original, predicted).Cluster.tolist(), [1, 2])
        with self.assertRaises(ValueError):
            attach_phenotypes(original, predicted.iloc[:1])


if __name__ == "__main__":
    unittest.main()

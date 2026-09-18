import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from candidate_features import (  # noqa: E402
    dummy_encode_candidates,
    encoded_terms_for_sources,
    source_consensus_ranking,
    source_feature_order,
    term_mapping,
)
from common import PERFUSION_METRICS, TERRITORIES  # noqa: E402


def synthetic_candidates() -> pd.DataFrame:
    n = 60
    cacs = np.tile(np.arange(6), 10)
    values = {
        "ID": np.arange(n),
        "Cluster": cacs % 3 + 1,
        "TIME": np.arange(n) + 1,
        "MACET": cacs % 3,
        "CACS": cacs,
        "age": np.arange(n) + 30.0,
        "constant": np.ones(n),
    }
    for ti, territory in enumerate(TERRITORIES):
        for mi, metric in enumerate(PERFUSION_METRICS):
            values[f"{territory}_{metric}"] = np.arange(n) * (mi + 1.1) + ti
    return pd.DataFrame(values)


class FeatureRankingTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_candidates()
        self.encoded, self.original, self.categorical = dummy_encode_candidates(self.frame, 6, False)
        self.mapping = term_mapping(self.original, self.categorical, False)

    def terms(self, cacs_score: int = 2) -> pd.DataFrame:
        table = pd.DataFrame({"term": list(self.mapping), "consensus_rank_sum": np.arange(len(self.mapping)) + 20})
        table.loc[table.term.eq("CACS=4"), "consensus_rank_sum"] = cacs_score
        return table

    def test_outcome_and_constant_columns_are_excluded(self):
        for name in ["ID", "Cluster", "TIME", "MACET", "constant"]:
            self.assertNotIn(name, self.original.columns)
        self.assertIn("LAD_MBF", self.original.columns)

    def test_categorical_terms_map_to_their_source(self):
        self.assertIn("CACS", self.categorical)
        self.assertEqual(list(self.mapping), self.encoded.columns.tolist())
        self.assertEqual([t for t, s in self.mapping.items() if s == "CACS"], [f"CACS={i}" for i in range(6)])

    def test_categorical_source_is_ranked_once_by_its_best_term(self):
        rank = source_consensus_ranking(self.terms(), self.mapping)
        self.assertEqual(rank.iloc[0]["feature"], "CACS")
        self.assertEqual(rank.iloc[0]["term"], "CACS=4")
        self.assertEqual(set(rank.feature), set(self.original.columns))
        self.assertEqual(rank.feature.value_counts()["CACS"], 1)
        self.assertIn("CACS", source_feature_order(rank, self.original.columns)[:13])

    def test_representative_row_is_a_real_term_row(self):
        terms = pd.DataFrame(
            {
                "term": ["CACS=1", "CACS=0", "marker"],
                "lasso_rank": [8, 1, 5],
                "boruta_rank": [1, 8, 3],
                "consensus_rank_sum": [9, 9, 8],
            }
        )
        mapping = {"CACS=0": "CACS", "CACS=1": "CACS", "marker": "marker"}
        rank = source_consensus_ranking(terms, mapping).set_index("feature")
        self.assertEqual(rank.loc["CACS", "term"], "CACS=0")
        self.assertEqual((rank.loc["CACS", "lasso_rank"], rank.loc["CACS", "boruta_rank"]), (1, 8))

    def test_selected_sources_expand_to_all_encoded_terms(self):
        terms = encoded_terms_for_sources(["CACS", "age"], self.mapping)
        self.assertEqual(terms, [f"CACS={i}" for i in range(6)] + ["age"])
        np.testing.assert_array_equal(self.encoded[terms[:6]].sum(axis=1), 1)
        with self.assertRaises(ValueError):
            encoded_terms_for_sources(["absent"], self.mapping)
        with self.assertRaises(ValueError):
            encoded_terms_for_sources(["CACS", "CACS"], self.mapping)

    def test_ambiguous_term_names_are_rejected(self):
        original = pd.DataFrame({"CACS": [0, 1], "CACS=0": [10.0, 20.0]})
        with self.assertRaises(ValueError):
            term_mapping(original, ["CACS"])

    def test_incomplete_rankings_are_rejected(self):
        rank = source_consensus_ranking(self.terms(), self.mapping)
        for bad in [rank[rank.feature.ne("CACS")], pd.concat([rank, rank.head(1)])]:
            with self.assertRaises(ValueError):
                source_feature_order(bad, self.original.columns)
        with self.assertRaises(ValueError):
            source_consensus_ranking(self.terms().iloc[:-1], self.mapping)

    def test_drop_first_keeps_source_mapping(self):
        encoded, original, categorical = dummy_encode_candidates(self.frame, 6, True)
        mapping = term_mapping(original, categorical, True)
        self.assertEqual(list(mapping), encoded.columns.tolist())
        self.assertEqual(len(encoded_terms_for_sources(["CACS"], mapping)), 5)


if __name__ == "__main__":
    unittest.main()

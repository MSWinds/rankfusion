import unittest
import pandas as pd
import numpy as np
from RankFusion import DistanceRankFusion

class TestDistanceRankFusion(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame for testing
        self.df = pd.DataFrame({
            'algo1': [0.9, 0.7, 0.3, 0.1],
            'algo2': [0.8, 0.6, 0.4, 0.2],
            'algo3': [0.95, 0.65, 0.35, 0.15]
        })
        self.score_columns = ['algo1', 'algo2', 'algo3']

    def test_rrf_initialization(self):
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='rrf',
            k=60,
            higher_is_better=True
        )
        result = fusion.get_result()

        # Check if result has expected columns
        expected_columns = set(self.score_columns + ['RRF_score'])
        self.assertEqual(set(result.columns), expected_columns)

        # Check if results are sorted by RRF_score in descending order
        self.assertTrue(result['RRF_score'].is_monotonic_decreasing)

        # Since higher_is_better=True, row with largest original values should appear first.
        # In this small dataset, row 0 has the largest sums (2.65),
        # so after RRF, we expect it near or at the top.
        # We won't enforce an exact index check because .reset_index(drop=True) is used;
        # instead, verify that the *top row* matches the highest-sum original data:
        top_row_in_result = result.loc[0, self.score_columns].values
        top_original_idx = self.df[self.score_columns].sum(axis=1).idxmax()
        self.assertTrue(
            np.allclose(top_row_in_result, self.df.loc[top_original_idx, self.score_columns].values),
            msg="The top row in RRF result is not the row with the largest original sum (when higher_is_better=True)."
        )

    def test_isr_initialization(self):
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='isr',
            k=60,
            higher_is_better=True
        )
        result = fusion.get_result()

        expected_columns = set(self.score_columns + ['ISR_score'])
        self.assertEqual(set(result.columns), expected_columns)
        self.assertTrue(result['ISR_score'].is_monotonic_decreasing)

    def test_rdf_initialization(self):
        weights = [0.4, 0.3, 0.3]
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='rdf',
            weights=weights,
            higher_is_better=True
        )
        result = fusion.get_result()

        expected_columns = set(self.score_columns + ['RDF_min_max_score'])
        self.assertEqual(set(result.columns), expected_columns)
        self.assertTrue(result['RDF_min_max_score'].is_monotonic_decreasing)

    def test_rdfdb_initialization(self):
        weights = [0.4, 0.3, 0.3]
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='rdfdb',
            weights=weights,
            higher_is_better=True
        )
        result = fusion.get_result()

        expected_columns = set(self.score_columns + ['RDF_dist_based_score'])
        self.assertEqual(set(result.columns), expected_columns)
        self.assertTrue(result['RDF_dist_based_score'].is_monotonic_decreasing)

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            DistanceRankFusion(
                df=self.df,
                score_columns=self.score_columns,
                algorithm='invalid_algo'
            )

    def test_invalid_weights(self):
        invalid_weights = [0.5, 0.2, 0.2]  # Sum != 1
        with self.assertRaises(ValueError):
            DistanceRankFusion(
                df=self.df,
                score_columns=self.score_columns,
                algorithm='rdf',
                weights=invalid_weights
            )

    def test_show_details(self):
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='rrf',
            higher_is_better=True
        )
        detailed_result = fusion.get_result(show_details=True)
        simple_result = fusion.get_result(show_details=False)

        # Detailed result should have columns with '_rank' or '_normalized'
        self.assertTrue(len(detailed_result.columns) > len(simple_result.columns))
        self.assertTrue(any('rank' in col for col in detailed_result.columns))
        self.assertFalse(any('rank' in col for col in simple_result.columns))

    def test_higher_is_better_false(self):
        """Check that rows with lower scores become top-ranked when higher_is_better=False."""
        fusion = DistanceRankFusion(
            df=self.df,
            score_columns=self.score_columns,
            algorithm='rrf',
            higher_is_better=False
        )
        result = fusion.get_result()

        # Identify which row originally had the smallest sum of scores
        # Because higher_is_better=False, that row should appear first in the new result.
        lowest_sum_idx = self.df[self.score_columns].sum(axis=1).idxmin()

        # Compare the top row in the new result to that "lowest-sum" row
        top_row_in_result = result.loc[0, self.score_columns].values
        expected_top_row_values = self.df.loc[lowest_sum_idx, self.score_columns].values

        self.assertTrue(
            np.allclose(top_row_in_result, expected_top_row_values),
            msg=(
                "When higher_is_better=False, the row with the lowest original scores "
                "should appear first, but it doesn't."
            )
        )

if __name__ == '__main__':
    unittest.main()

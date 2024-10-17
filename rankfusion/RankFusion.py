import pandas as pd
import numpy as np
from typing import List, Optional

class DistanceRankFusion:
    def __init__(self, df: pd.DataFrame, score_columns: List[str], algorithm: str = 'rrf', weights: Optional[List[float]] = None, k: Optional[int] = None, higher_is_better: bool = False):
        """
        Initialize and execute the chosen algorithm on the DataFrame.

        :param df: DataFrame containing the scores or distances.
        :param score_columns: List of column names containing the scores from different algorithms.
        :param algorithm: The algorithm to use ('rrf', 'isr', 'rdf', 'rdfdb').
        :param weights: Optional list of weights for each algorithm. Must sum to 1.
        :param k: The k parameter for RRF and ISR.
        :param higher_is_better: Boolean indicating if higher score columns are better.
        """
        self.df = df
        self.score_columns = score_columns
        self.algorithm = algorithm
        self.k = k if k is not None else 60  # Default value for k if not provided
        self.higher_is_better = higher_is_better

        if algorithm in ['rrf', 'isr'] and weights is not None:
            raise ValueError("Weights should not be defined for 'rrf' or 'isr' algorithms.")
        if algorithm in ['rdf', 'rdfdb'] and k is not None:
            raise ValueError("'k' parameter should not be defined for 'rdf' or 'rdfdb' algorithms.")

        if algorithm == 'rrf':
            self.result_df = self._reciprocal_rank_fusion(self.k)
        elif algorithm == 'isr':
            self.result_df = self._inverse_square_rank_fusion(self.k)
        elif algorithm == 'rdf':
            self.result_df = self._relative_distance_fusion_min_max(weights)
        elif algorithm == 'rdfdb':
            self.result_df = self._relative_distance_fusion_dist_based(weights)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")

    def _validate_weights(self, weights: Optional[List[float]]) -> List[float]:
        if weights is None:
            weights = [1 / len(self.score_columns)] * len(self.score_columns)
        elif not np.isclose(sum(weights), 1, atol=1e-6):
            raise ValueError("Sum of weights must be close to 1")
        return weights

    def _reciprocal_rank_fusion(self, k: int) -> pd.DataFrame:
        df_copy = self.df.copy()
        for col in self.score_columns:
            rank_col = f"{col}_rank"
            df_copy[rank_col] = df_copy[col].rank(method='min', ascending=not self.higher_is_better)  # rank based on whether higher is better
            df_copy[f"{col}_rrf_transformed"] = 1 / (df_copy[rank_col] + k)
        df_copy['RRF_score'] = df_copy[[f"{col}_rrf_transformed" for col in self.score_columns]].sum(axis=1)
        df_copy = df_copy.drop(columns=[f"{col}_rrf_transformed" for col in self.score_columns])
        return df_copy.sort_values('RRF_score', ascending=False).reset_index(drop=True)

    def _inverse_square_rank_fusion(self, k: int) -> pd.DataFrame:
        df_copy = self.df.copy()
        for col in self.score_columns:
            rank_col = f"{col}_rank"
            df_copy[rank_col] = df_copy[col].rank(method='min', ascending=not self.higher_is_better)  # rank based on whether higher is better
            df_copy[f"{col}_isr_transformed"] = 1 / ((df_copy[rank_col] + k) ** 2)
        df_copy['ISR_score'] = df_copy[[f"{col}_isr_transformed" for col in self.score_columns]].sum(axis=1)
        df_copy = df_copy.drop(columns=[f"{col}_isr_transformed" for col in self.score_columns])
        return df_copy.sort_values('ISR_score', ascending=False).reset_index(drop=True)

    def _relative_distance_fusion_min_max(self, weights: Optional[List[float]]) -> pd.DataFrame:
        weights = self._validate_weights(weights)
        df_copy = self.df.copy()
        for col in self.score_columns:
            normalized_col = f"{col}_normalized"
            max_score = df_copy[col].max()
            min_score = df_copy[col].min()
            epsilon = 1e-9  # To avoid division by zero
            if self.higher_is_better:
                df_copy[normalized_col] = (df_copy[col] - min_score) / (max_score - min_score + epsilon)
            else:
                df_copy[normalized_col] = (max_score - df_copy[col]) / (max_score - min_score + epsilon)  # reverse the scores
        for col, weight in zip(self.score_columns, weights):
            normalized_col = f"{col}_normalized"
            df_copy[normalized_col] *= weight
        df_copy['RDF_min_max_score'] = df_copy[[f"{col}_normalized" for col in self.score_columns]].sum(axis=1)
        return df_copy.sort_values('RDF_min_max_score', ascending=False).reset_index(drop=True)  # sort final score descending

    def _relative_distance_fusion_dist_based(self, weights: Optional[List[float]]) -> pd.DataFrame:
        """
        3 sigma rules
        https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
        """
        weights = self._validate_weights(weights)
        df_copy = self.df.copy()
        for col in self.score_columns:
            normalized_col = f"{col}_normalized"
            mean_score = df_copy[col].mean()
            std_dev = df_copy[col].std()
            min_score = mean_score - 3 * std_dev
            max_score = mean_score + 3 * std_dev
            epsilon = 1e-9  # To avoid division by zero
            if self.higher_is_better:
                df_copy[normalized_col] = (df_copy[col] - min_score) / (max_score - min_score + epsilon)
            else:
                df_copy[normalized_col] = (max_score - df_copy[col]) / (max_score - min_score + epsilon)
        for col, weight in zip(self.score_columns, weights):
            normalized_col = f"{col}_normalized"
            df_copy[normalized_col] *= weight
        df_copy['RDF_dist_based_score'] = df_copy[[f"{col}_normalized" for col in self.score_columns]].sum(axis=1)
        return df_copy.sort_values('RDF_dist_based_score', ascending=False).reset_index(drop=True)  # sort final score descending

    def get_result(self, show_details: bool = False) -> pd.DataFrame:
        """
        Return the result DataFrame processed by the specified algorithm.

        :param show_details: Boolean to decide whether to include rank and normalized columns in the output.
        :return: DataFrame with the final scores sorted accordingly.
        """
        result_df = self.result_df.copy()
        if not show_details:
            columns_to_drop = [col for col in result_df.columns if '_rank' in col or '_normalized' in col]
            result_df = result_df.drop(columns=columns_to_drop)
        return result_df

    
import pandas as pd
import numpy as np
from typing import List, Optional

class DistanceRankFusion:
    def __init__(
        self,
        df: pd.DataFrame,
        score_columns: List[str],
        algorithm: str = 'rrf',
        weights: Optional[List[float]] = None,
        k: Optional[int] = None,
        higher_is_better: bool = False
    ):
        """
        Initialize and execute the chosen algorithm on the DataFrame.

        :param df: DataFrame containing the scores or distances.
        :param score_columns: List of column names containing the scores from different algorithms.
        :param algorithm: The algorithm to use ('rrf', 'isr', 'rdf', 'rdfdb').
        :param weights: Optional list of weights for each algorithm. Must sum to 1.
        :param k: The k parameter for RRF and ISR (ignored if using 'rdf' or 'rdfdb').
        :param higher_is_better: Boolean indicating if higher score columns are better.
        """
        self.df = df
        self.score_columns = score_columns
        self.algorithm = algorithm
        self.k = k if k is not None else 60  # Default value for k if not provided
        self.higher_is_better = higher_is_better

        # Validate parameters
        # If using RDF-based algorithms, 'k' must not be set
        if algorithm in ['rdf', 'rdfdb'] and k is not None:
            raise ValueError("'k' parameter should not be defined for 'rdf' or 'rdfdb' algorithms.")

        # For all algorithms, optionally validate or default the weights
        self.weights = self._validate_weights(weights)

        # Dispatch to the chosen algorithm
        if algorithm == 'rrf':
            self.result_df = self._reciprocal_rank_fusion(self.k)
        elif algorithm == 'isr':
            self.result_df = self._inverse_square_rank_fusion(self.k)
        elif algorithm == 'rdf':
            self.result_df = self._relative_distance_fusion_min_max(self.weights)
        elif algorithm == 'rdfdb':
            self.result_df = self._relative_distance_fusion_dist_based(self.weights)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")

    def _validate_weights(self, weights: Optional[List[float]]) -> List[float]:
        """
        Validate or generate uniform weights, ensuring they sum to 1.
        """
        if weights is None:
            return [1.0 / len(self.score_columns)] * len(self.score_columns)
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError("Sum of weights must be close to 1.")
        return weights

    def _reciprocal_rank_fusion(self, k: int) -> pd.DataFrame:
        """
        Weighted Reciprocal Rank Fusion (RRF).
        Each column's contribution = (1 / (rank + k)) * weight.
        """
        df_copy = self.df.copy()
        # Calculate rank-based transforms for each column
        for col_idx, col in enumerate(self.score_columns):
            rank_col = f"{col}_rank"
            df_copy[rank_col] = df_copy[col].rank(
                method='min', ascending=not self.higher_is_better
            )
            # Weighted RRF transform
            df_copy[f"{col}_rrf_transformed"] = (
                1 / (df_copy[rank_col] + k)
            ) * self.weights[col_idx]

        # Sum up all transforms as final RRF score
        transform_cols = [f"{col}_rrf_transformed" for col in self.score_columns]
        df_copy['RRF_score'] = df_copy[transform_cols].sum(axis=1)

        # Remove the transform columns (but keep rank columns for optional details)
        df_copy = df_copy.drop(columns=transform_cols)

        # Sort descending by RRF_score
        return df_copy.sort_values('RRF_score', ascending=False).reset_index(drop=True)

    def _inverse_square_rank_fusion(self, k: int) -> pd.DataFrame:
        """
        Weighted Inverse Square Rank (ISR).
        Each column's contribution = (1 / (rank + k)^2) * weight.
        """
        df_copy = self.df.copy()
        for col_idx, col in enumerate(self.score_columns):
            rank_col = f"{col}_rank"
            df_copy[rank_col] = df_copy[col].rank(
                method='min', ascending=not self.higher_is_better
            )
            # Weighted ISR transform
            df_copy[f"{col}_isr_transformed"] = (
                1 / ((df_copy[rank_col] + k) ** 2)
            ) * self.weights[col_idx]

        transform_cols = [f"{col}_isr_transformed" for col in self.score_columns]
        df_copy['ISR_score'] = df_copy[transform_cols].sum(axis=1)

        df_copy = df_copy.drop(columns=transform_cols)
        return df_copy.sort_values('ISR_score', ascending=False).reset_index(drop=True)

    def _relative_distance_fusion_min_max(self, weights: List[float]) -> pd.DataFrame:
        """
        Weighted Relative Distance Fusion (min-max normalization).
        """
        df_copy = self.df.copy()
        for col_idx, col in enumerate(self.score_columns):
            normalized_col = f"{col}_normalized"
            max_score = df_copy[col].max()
            min_score = df_copy[col].min()
            epsilon = 1e-9

            if self.higher_is_better:
                df_copy[normalized_col] = (
                    df_copy[col] - min_score
                ) / (max_score - min_score + epsilon)
            else:
                # Reverse scores if lower is better
                df_copy[normalized_col] = (
                    max_score - df_copy[col]
                ) / (max_score - min_score + epsilon)

            # Multiply by that column's weight
            df_copy[normalized_col] *= weights[col_idx]

        norm_cols = [f"{col}_normalized" for col in self.score_columns]
        df_copy['RDF_min_max_score'] = df_copy[norm_cols].sum(axis=1)
        return df_copy.sort_values('RDF_min_max_score', ascending=False).reset_index(drop=True)

    def _relative_distance_fusion_dist_based(self, weights: List[float]) -> pd.DataFrame:
        """
        Weighted Distribution-Based Score Fusion (3-sigma).
        For each column, scale into [0..1] range using mean ± 3*std dev.
        """
        df_copy = self.df.copy()
        for col_idx, col in enumerate(self.score_columns):
            normalized_col = f"{col}_normalized"
            mean_score = df_copy[col].mean()
            std_dev = df_copy[col].std()
            min_score = mean_score - 3 * std_dev
            max_score = mean_score + 3 * std_dev
            epsilon = 1e-9

            if self.higher_is_better:
                df_copy[normalized_col] = (
                    df_copy[col] - min_score
                ) / (max_score - min_score + epsilon)
            else:
                # Reverse scores if lower is better
                df_copy[normalized_col] = (
                    max_score - df_copy[col]
                ) / (max_score - min_score + epsilon)

            # Multiply by weight
            df_copy[normalized_col] *= weights[col_idx]

        norm_cols = [f"{col}_normalized" for col in self.score_columns]
        df_copy['RDF_dist_based_score'] = df_copy[norm_cols].sum(axis=1)
        return df_copy.sort_values('RDF_dist_based_score', ascending=False).reset_index(drop=True)

    def get_result(self, show_details: bool = False) -> pd.DataFrame:
        """
        Return the result DataFrame processed by the specified algorithm.

        :param show_details: Boolean to decide whether to include rank and normalized columns in the output.
        :return: DataFrame with the final scores sorted accordingly.
        """
        result_df = self.result_df.copy()
        if not show_details:
            # Drop rank / normalized columns if not needed
            columns_to_drop = [
                col for col in result_df.columns
                if '_rank' in col or '_normalized' in col
            ]
            result_df = result_df.drop(columns=columns_to_drop)
        return result_df

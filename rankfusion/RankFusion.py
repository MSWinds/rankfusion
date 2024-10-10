import pandas as pd
import numpy as np
from typing import List, Optional, Callable

class DistanceRankFusion:
    """
    A class for performing distance-based rank fusion on multiple scoring algorithms.

    This class implements various rank fusion algorithms including Reciprocal Rank Fusion (RRF),
    Inverse Square Rank (ISR), and Relative Distance Fusion (RDF) with different normalization methods.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the scores.
        score_columns (List[str]): List of column names containing the scores from different algorithms.
        algorithm (str): The chosen fusion algorithm ('rrf', 'isr', 'rdf', 'rdfdb').
        k (int): The k parameter for RRF and ISR algorithms.
        weights (np.ndarray): Weights for each scoring algorithm.
        higher_is_better (bool): Flag indicating whether higher scores are better.
        result_df (pd.DataFrame): The resulting DataFrame after applying the fusion algorithm.
    """
    def __init__(self, df: pd.DataFrame, score_columns: List[str], algorithm: str = 'rrf', 
                 weights: Optional[List[float]] = None, k: int = 60,
                 higher_is_better: bool = False):
        self._validate_inputs(df, score_columns, algorithm, weights)
    
        self.df = df.copy()  # Create a deep copy of the input DataFrame
        self.score_columns = score_columns
        self.algorithm = algorithm
        self.k = k if algorithm in ['rrf', 'isr'] else None
        self.weights = self._validate_weights(weights)
        self.higher_is_better = higher_is_better
        
        self.result_df = getattr(self, f'_{algorithm}')()

    def _validate_inputs(self, df: pd.DataFrame, score_columns: List[str], 
                         algorithm: str, weights: Optional[List[float]]) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
        if not set(score_columns).issubset(df.columns):
            raise KeyError("Some score columns are not in the DataFrame")
        if algorithm not in ['rrf', 'isr', 'rdf', 'rdfdb']:
            raise ValueError(f"Unknown algorithm type: {algorithm}")
        if weights is not None:
            if len(weights) != len(score_columns):
                raise ValueError("Number of weights must match number of score columns")
            if not np.isclose(sum(weights), 1):
                raise ValueError("Weights must sum to 1")

    def _validate_weights(self, weights: Optional[List[float]]) -> np.ndarray:
        if weights is None:
            return np.full(len(self.score_columns), 1 / len(self.score_columns))
        return np.array(weights)

    def _rank_scores(self) -> pd.DataFrame:
        rank_method = 'min' if self.higher_is_better else 'max'
        ranks = self.df[self.score_columns].rank(method=rank_method, ascending=not self.higher_is_better)
        return ranks

    def _normalize_scores(self, method: str = 'min_max') -> pd.DataFrame:
        """
        Normalize the scores using the specified method.

        :param method: Normalization method ('min_max' or 'dist_based').
        :return: DataFrame with normalized scores.
        :raises ValueError: If an unknown normalization method is specified.
        """
        df = self.df[self.score_columns]
        if method == 'min_max':
            normalized = (df - df.min()) / (df.max() - df.min())
        elif method == 'dist_based':
            mean, std = df.mean(), df.std()
            normalized = (df - (mean - 3 * std)) / (6 * std)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized

    def _rrf(self) -> pd.DataFrame:
        ranks = self._rank_scores()
        result = self.df.copy()
        result['RRF_score'] = (1 / (ranks + self.k)).sum(axis=1)
        return result.sort_values('RRF_score', ascending=False).reset_index(drop=True)

    def _isr(self) -> pd.DataFrame:
        ranks = self._rank_scores()
        result = self.df.copy()
        result['ISR_score'] = (1 / (ranks + self.k) ** 2).sum(axis=1)
        return result.sort_values('ISR_score', ascending=False).reset_index(drop=True)

    def _relative_distance_fusion(self, normalize_method: str) -> pd.DataFrame:
        normalized = self._normalize_scores(normalize_method)
        if not self.higher_is_better:
            normalized = 1 - normalized
        
        weights = self.weights.flatten()
        rdf_scores = (normalized * weights).sum(axis=1)
        
        result = self.df.copy()
        score_name = f'RDF_{normalize_method}_score'
        result[score_name] = rdf_scores
        return result.sort_values(score_name, ascending=False).reset_index(drop=True)

    def _rdf(self) -> pd.DataFrame:
        return self._relative_distance_fusion('min_max')

    def _rdfdb(self) -> pd.DataFrame:
        return self._relative_distance_fusion('dist_based')

    def get_result(self, show_details: bool = False, reset_index: bool = True) -> pd.DataFrame:
        """
        Return the result DataFrame processed by the specified algorithm.

        :param show_details: Boolean to decide whether to include additional details in the output.
        :param reset_index: Boolean to decide whether to reset the index of the output DataFrame.
        :return: DataFrame with the final scores sorted accordingly.
        """
        result = self.result_df.copy()
        
        # Determine the score column name based on the algorithm
        if self.algorithm in ['rrf', 'isr']:
            score_column = f'{self.algorithm.upper()}_score'
        elif self.algorithm in ['rdf', 'rdfdb']:
            normalize_method = 'min_max' if self.algorithm == 'rdf' else 'dist_based'
            score_column = f'RDF_{normalize_method}_score'
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Sort the DataFrame by the score column
        result = result.sort_values(by=score_column, ascending=False)
        
        if show_details:
            if self.algorithm in ['rrf', 'isr']:
                # Add only rank columns for RRF and ISR
                ranks = self._rank_scores()
                for col in ranks.columns:
                    result[f"{col}_rank"] = ranks[col]
            elif self.algorithm in ['rdf', 'rdfdb']:
                # Add only normalized columns for RDF and RDFDB
                normalized = self._normalize_scores('min_max' if self.algorithm == 'rdf' else 'dist_based')
                for col in normalized.columns:
                    result[f"{col}_normalized"] = normalized[col]
        else:
            # If not showing details, keep only original columns and the final score column
            original_columns = self.df.columns.tolist()
            result = result[original_columns + [score_column]]
        
        if reset_index:
            result = result.reset_index(drop=True)
        
        return result

    @staticmethod
    def custom_fusion(df: pd.DataFrame, score_columns: List[str], 
                      fusion_func: Callable[[pd.DataFrame], pd.Series], 
                      result_column: str = 'Custom_score',
                      reset_index: bool = True) -> pd.DataFrame:
        """
        Apply a custom fusion function to the input DataFrame.

        :param df: Input DataFrame containing the scores.
        :param score_columns: List of column names containing the scores.
        :param fusion_func: Custom fusion function that takes a DataFrame of scores and returns a Series of fused scores.
        :param result_column: Name of the column to store the custom fusion scores.
        :param reset_index: Boolean to decide whether to reset the index of the output DataFrame.
        :return: DataFrame with custom fusion scores.
        """
        df_copy = df.copy()
        df_copy[result_column] = fusion_func(df_copy[score_columns])
        result = df_copy.sort_values(result_column, ascending=False)
        if reset_index:
            result = result.reset_index(drop=True)
        return result
    
    # Example: Using custom_fusion
    # def geometric_mean_fusion(scores_df):
    #     return np.prod(scores_df, axis=1) ** (1 / scores_df.shape[1])

    # custom_result = DistanceRankFusion.custom_fusion(df, score_columns=['score1', 'score2', 'score3'], 
    #                                                 fusion_func=geometric_mean_fusion, 
    #                                                 result_column='Geometric_Mean_Score', reset_index=True)
    # print("\nCustom Fusion Result (Geometric Mean):")
    # print(custom_result)
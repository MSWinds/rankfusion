import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Union
class DistanceRankFusion:
    def __init__(self, df: pd.DataFrame, score_columns: List[str], algorithm: str = 'rrf', 
                 weights: Optional[List[float]] = None, k: int = 60,
                 higher_is_better: bool = False):
        """
            Initialize and execute the chosen algorithm on the DataFrame.

            Args:
                df (pd.DataFrame): DataFrame containing the scores.
                score_columns (List[str]): List of column names containing the scores from different algorithms.
                algorithm (str): The algorithm to use ('rrf', 'isr', 'rdf', 'rdfdb').
                weights (Optional[List[float]]): Optional list of weights for each algorithm. Must sum to 1.
                k (int): The k parameter for RRF and ISR.
                higher_is_better (bool): If True, assumes higher original scores are better. Default is False.

            Raises:
                ValueError: If the algorithm is unknown or if the weights don't sum to 1.
                KeyError: If any of the score_columns are not in the DataFrame.
        """

        self._validate_inputs(df, score_columns, algorithm, weights)
    
        self.df = df
        self.score_columns = score_columns
        self.algorithm = algorithm
        self.k = k
        self.higher_is_better = higher_is_better
        
        algorithm_map = {
            'rrf': self._reciprocal_rank_fusion,
            'isr': self._inverse_square_rank_fusion,
            'rdf': self._relative_distance_fusion_min_max,
            'rdfdb': self._relative_distance_fusion_dist_based
        }
        
        self.result_df = algorithm_map[algorithm](weights)

    def _validate_inputs(self, df: pd.DataFrame, score_columns: List[str], 
                     algorithm: str, weights: Optional[List[float]]) -> None:
        """Validate input parameters."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
        
        if not set(score_columns).issubset(df.columns):
            raise KeyError("Some score columns are not in the DataFrame")
        
        if algorithm not in ['rrf', 'isr', 'rdf', 'rdfdb']:
            raise ValueError(f"Unknown algorithm type: {algorithm}")
        
        if weights is not None and not np.isclose(sum(weights), 1):
            raise ValueError("Sum of weights must be 1")

    def _validate_weights(self, weights: Optional[List[float]]) -> np.ndarray:
        """Validate and return weights as numpy array."""
        if weights is None:
            return np.full(len(self.score_columns), 1 / len(self.score_columns))
        return np.array(weights)

    def _rank_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank scores for all columns."""
        rank_method = 'min' if self.higher_is_better else 'max'
        return df[self.score_columns].rank(method=rank_method, ascending=not self.higher_is_better)

    def _reciprocal_rank_fusion(self, weights: Optional[List[float]]) -> pd.DataFrame:
        """Implement Reciprocal Rank Fusion."""
        ranks = self._rank_scores(self.df)
        rrf_scores = 1 / (ranks + self.k)
        self.df['RRF_score'] = np.sum(rrf_scores * self._validate_weights(weights), axis=1)
        return self.df.sort_values('RRF_score', ascending=False).reset_index(drop=True)

    def _inverse_square_rank_fusion(self, weights: Optional[List[float]]) -> pd.DataFrame:
        """Implement Inverse Square Rank Fusion."""
        ranks = self._rank_scores(self.df)
        isr_scores = 1 / (ranks + self.k) ** 2
        self.df['ISR_score'] = np.sum(isr_scores * self._validate_weights(weights), axis=1)
        return self.df.sort_values('ISR_score', ascending=False).reset_index(drop=True)

    def _normalize_scores(self, df: pd.DataFrame, method: str = 'min_max') -> pd.DataFrame:
        """Normalize scores using specified method."""
        if method == 'min_max':
            return (df - df.min()) / (df.max() - df.min())
        elif method == 'dist_based':
            mean, std = df.mean(), df.std()
            return (df - (mean - 3 * std)) / (6 * std)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _relative_distance_fusion(self, weights: Optional[List[float]], 
                              normalize_method: str) -> pd.DataFrame:
        """Implement Relative Distance Fusion."""
        normalized = self._normalize_scores(self.df[self.score_columns], normalize_method)
        if not self.higher_is_better:
            normalized = 1 - normalized
        rdf_scores = np.sum(normalized * self._validate_weights(weights).reshape(-1, 1), axis=1)
        score_name = f'RDF_{normalize_method}_score'
        self.df[score_name] = rdf_scores
        return self.df.sort_values(score_name, ascending=False).reset_index(drop=True)

    def _relative_distance_fusion_min_max(self, weights: Optional[List[float]]) -> pd.DataFrame:
        """Implement Relative Distance Fusion with Min-Max normalization."""
        return self._relative_distance_fusion(weights, 'min_max')

    def _relative_distance_fusion_dist_based(self, weights: Optional[List[float]]) -> pd.DataFrame:
        """Implement Relative Distance Fusion with Distribution-based normalization.
        3 sigma rules
        https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
        """
        return self._relative_distance_fusion(weights, 'dist_based')

    def get_result(self, show_details: bool = False, reset_index: bool = True) -> pd.DataFrame:
        """
        Return the result DataFrame processed by the specified algorithm.

        Args:
            show_details (bool): If True, include intermediate columns in the output.
            reset_index (bool): If True, reset the index of the returned DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the final scores sorted accordingly.
        """
        result = self.result_df.copy()
        if not show_details:
            result = result[[col for col in result.columns 
                            if not col.endswith(('_rank', '_normalized'))]]
        if reset_index:
            result = result.reset_index(drop=True)
        return result

    @staticmethod
    def custom_fusion(df: pd.DataFrame, score_columns: List[str], 
                    fusion_func: Callable[[pd.DataFrame], pd.Series], 
                    result_column: str = 'Custom_score',
                    reset_index: bool = True) -> pd.DataFrame:
        """
        Apply a custom fusion function to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            score_columns (List[str]): Columns to use for fusion.
            fusion_func (Callable): Function that takes a DataFrame and returns a Series of scores.
            result_column (str): Name for the resulting score column.
            reset_index (bool): If True, reset the index of the returned DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the custom fusion scores.
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
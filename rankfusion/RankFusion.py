import pandas as pd
import numpy as np
from typing import List, Optional, Callable

class DistanceRankFusion:
    def __init__(self, df: pd.DataFrame, score_columns: List[str], algorithm: str = 'rrf', 
                 weights: Optional[List[float]] = None, k: int = 60,
                 higher_is_better: bool = False):
        self._validate_inputs(df, score_columns, algorithm, weights)
    
        self.df = df
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
        return self.df[self.score_columns].rank(method=rank_method, ascending=not self.higher_is_better)

    def _rrf(self) -> pd.DataFrame:
        ranks = self._rank_scores()
        self.df['RRF_score'] = (1 / (ranks + self.k)).sum(axis=1)
        return self.df.sort_values('RRF_score', ascending=False).reset_index(drop=True)

    def _isr(self) -> pd.DataFrame:
        ranks = self._rank_scores()
        self.df['ISR_score'] = (1 / (ranks + self.k) ** 2).sum(axis=1)
        return self.df.sort_values('ISR_score', ascending=False).reset_index(drop=True)

    def _normalize_scores(self, method: str = 'min_max') -> pd.DataFrame:
        df = self.df[self.score_columns]
        if method == 'min_max':
            return (df - df.min()) / (df.max() - df.min())
        elif method == 'dist_based':
            mean, std = df.mean(), df.std()
            return (df - (mean - 3 * std)) / (6 * std)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _relative_distance_fusion(self, normalize_method: str) -> pd.DataFrame:
        normalized = self._normalize_scores(normalize_method)
        if not self.higher_is_better:
            normalized = 1 - normalized
        
        # Ensure weights are a 1D array with the same length as score_columns
        weights = self.weights.flatten()
        
        # Use pandas DataFrame multiplication for correct alignment
        rdf_scores = (normalized * weights).sum(axis=1)
        
        score_name = f'RDF_{normalize_method}_score'
        self.df[score_name] = rdf_scores
        return self.df.sort_values(score_name, ascending=False).reset_index(drop=True)

    def _rdf(self) -> pd.DataFrame:
        return self._relative_distance_fusion('min_max')

    def _rdfdb(self) -> pd.DataFrame:
        return self._relative_distance_fusion('dist_based')

    def get_result(self, show_details: bool = False, reset_index: bool = True) -> pd.DataFrame:
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
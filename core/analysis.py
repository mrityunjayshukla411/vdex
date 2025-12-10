"""
Analysis tools - Functional style for data analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from core.domain import Dataset


class Analysis:
    """Functional analysis tools for datasets"""
    
    def __init__(self, dataset: Dataset):
        """
        Initialize with a dataset
        
        Args:
            dataset: Dataset to analyze
        """
        self.dataset = dataset
        self.df = dataset.to_dataframe()
    
    def compare(self, metric: str) -> pd.DataFrame:
        """
        Compare metric across simulations and scenes
        
        Args:
            metric: Metric name
            
        Returns:
            Pivot table with scenes as rows, simulations as columns
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        return self.df.pivot(
            index='scene',
            columns='simulation',
            values=metric
        )
    
    def normalize(
        self,
        baseline: str,
        metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize data relative to baseline simulation
        
        Args:
            baseline: Name of baseline simulation
            metric: Specific metric (None = all numeric metrics)
            
        Returns:
            DataFrame with normalized values
        """
        if baseline not in self.df['simulation'].values:
            raise ValueError(f"Baseline '{baseline}' not found")
        
        # Get baseline data
        baseline_df = self.df[self.df['simulation'] == baseline].copy()
        baseline_df = baseline_df.set_index('scene')
        
        # Normalize each simulation
        normalized_dfs = []
        
        for sim_name in self.df['simulation'].unique():
            sim_df = self.df[self.df['simulation'] == sim_name].copy()
            sim_df = sim_df.set_index('scene')
            
            # Get numeric columns
            numeric_cols = sim_df.select_dtypes(include=[np.number]).columns
            
            if metric:
                if metric not in numeric_cols:
                    raise ValueError(f"Metric '{metric}' not numeric")
                numeric_cols = [metric]
            
            # Normalize
            for col in numeric_cols:
                sim_df[f'{col}_normalized'] = sim_df[col] / baseline_df[col]
            
            sim_df['simulation'] = sim_name
            normalized_dfs.append(sim_df.reset_index())
        
        return pd.concat(normalized_dfs, ignore_index=True)
    
    def summarize(self, metric: str) -> pd.DataFrame:
        """
        Get summary statistics for a metric
        
        Args:
            metric: Metric name
            
        Returns:
            Summary statistics per simulation
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        return self.df.groupby('simulation')[metric].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('geomean', lambda x: np.exp(np.mean(np.log(x[x > 0]))))
        ])
    
    def filter_by(
        self,
        predicate: Callable[[pd.Series], bool]
    ) -> 'Analysis':
        """
        Filter dataset by custom predicate
        
        Args:
            predicate: Function that takes a row and returns bool
            
        Returns:
            New Analysis with filtered data
        """
        filtered_df = self.df[self.df.apply(predicate, axis=1)]
        
        # Convert back to Dataset
        from core.domain import ExtractionResult, Simulation, Scene, Metric
        
        results = []
        for _, row in filtered_df.iterrows():
            sim = Simulation(Path("."), row['simulation'])
            scene = Scene(row['scene'])
            
            metrics = {}
            for col in filtered_df.columns:
                if col not in ['simulation', 'scene']:
                    metrics[col] = Metric(col, row[col])
            
            results.append(ExtractionResult(sim, scene, metrics))
        
        return Analysis(Dataset(results))
    
    def apply(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> 'Analysis':
        """
        Apply custom function to DataFrame
        
        Args:
            func: Function that transforms DataFrame
            
        Returns:
            New Analysis with transformed data
        """
        new_df = func(self.df)
        
        # Convert back to Dataset (simplified)
        from core.domain import ExtractionResult, Simulation, Scene, Metric
        from pathlib import Path
        
        results = []
        for _, row in new_df.iterrows():
            sim = Simulation(Path("."), row['simulation'])
            scene = Scene(row['scene'])
            
            metrics = {}
            for col in new_df.columns:
                if col not in ['simulation', 'scene']:
                    metrics[col] = Metric(col, row[col])
            
            results.append(ExtractionResult(sim, scene, metrics))
        
        return Analysis(Dataset(results))
    
    def top_n(self, metric: str, n: int = 5) -> pd.DataFrame:
        """Get top N results for a metric"""
        return self.df.nlargest(n, metric)[['simulation', 'scene', metric]]
    
    def bottom_n(self, metric: str, n: int = 5) -> pd.DataFrame:
        """Get bottom N results for a metric"""
        return self.df.nsmallest(n, metric)[['simulation', 'scene', metric]]
    
    def correlation(self, metric1: str, metric2: str) -> float:
        """Get correlation between two metrics"""
        return self.df[metric1].corr(self.df[metric2])
    
    def metrics(self) -> list:
        """Get list of available metrics"""
        return [col for col in self.df.columns if col not in ['simulation', 'scene']]

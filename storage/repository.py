"""
Storage layer - Repository pattern for data persistence
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import pandas as pd
import warnings

from core.domain import Dataset, Experiment


class Repository(ABC):
    """Abstract repository interface"""
    
    @abstractmethod
    def save(self, experiment: Experiment, dataset: Dataset) -> None:
        """Save dataset for an experiment"""
        pass
    
    @abstractmethod
    def load(self, experiment: Experiment) -> Dataset:
        """Load dataset for an experiment"""
        pass
    
    @abstractmethod
    def exists(self, experiment: Experiment) -> bool:
        """Check if data exists for experiment"""
        pass

    @property
    @abstractmethod
    def data_dir(self) -> Path:
        """Get base data directory"""
        pass


class ParquetRepository(Repository):
    """Repository using Parquet format for storage"""
    
    def __init__(self, base_path: Path = Path(".")):
        """
        Initialize repository
        
        Args:
            base_path: Base directory for all storage
        """
        self.base_path = Path(base_path)
        self._data_dir = self.base_path / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def data_dir(self) -> Path:
        """Get base data directory"""
        return self._data_dir
    
    def save(self, experiment: Experiment, dataset: Dataset) -> None:
        """Save dataset to Parquet"""
        exp_dir = self._experiment_dir(experiment)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and save
        df = dataset.to_dataframe()
        if df.empty:
            warnings.warn(f"Empty dataset for experiment '{experiment.name}'")
            return
        
        file_path = exp_dir / "data.parquet"
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
        
        # Save metadata
        self._save_metadata(experiment, exp_dir)
    
    def load(self, experiment: Experiment) -> Dataset:
        """Load dataset from Parquet"""
        exp_dir = self._experiment_dir(experiment)
        file_path = exp_dir / "data.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for experiment '{experiment.name}'")
        
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        # Convert DataFrame back to Dataset
        # This is a simplified conversion - in practice you'd reconstruct full objects
        from core.domain import ExtractionResult, Simulation, Scene, Metric
        from datetime import datetime
        
        results = []
        for _, row in df.iterrows():
            sim = Simulation(Path("."), row['simulation'])
            scene = Scene(row['scene'])
            
            # Extract metrics from row
            metrics = {}
            for col in df.columns:
                if col not in ['simulation', 'scene']:
                    metrics[col] = Metric(col, row[col])
            
            results.append(ExtractionResult(sim, scene, metrics))
        
        return Dataset(results)
    
    def exists(self, experiment: Experiment) -> bool:
        """Check if data exists"""
        exp_dir = self._experiment_dir(experiment)
        return (exp_dir / "data.parquet").exists()
    
    def _experiment_dir(self, experiment: Experiment) -> Path:
        """Get directory for experiment"""
        return self.data_dir / experiment.name
    
    def _save_metadata(self, experiment: Experiment, exp_dir: Path):
        """Save experiment metadata as JSON"""
        import json
        
        metadata = {
            'name': experiment.name,
            'description': experiment.description,
            'simulations': [s.name for s in experiment.simulations],
            'scenes': [s.name for s in experiment.scenes],
        }
        
        with open(exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class CSVRepository(Repository):
    """Repository using CSV format (for human readability)"""
    
    def __init__(self, base_path: Path = Path(".")):
        self.base_path = Path(base_path)
        self._data_dir = self.base_path / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def data_dir(self) -> Path:
        """Get base data directory"""
        return self._data_dir
    
    def save(self, experiment: Experiment, dataset: Dataset) -> None:
        """Save dataset to CSV"""
        exp_dir = self._experiment_dir(experiment)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        df = dataset.to_dataframe()
        if df.empty:
            warnings.warn(f"Empty dataset for experiment '{experiment.name}'")
            return
        
        file_path = exp_dir / "data.csv"
        df.to_csv(file_path, index=False)
    
    def load(self, experiment: Experiment) -> Dataset:
        """Load dataset from CSV"""
        exp_dir = self._experiment_dir(experiment)
        file_path = exp_dir / "data.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for experiment '{experiment.name}'")
        
        df = pd.read_csv(file_path)
        
        # Convert to Dataset (simplified)
        from core.domain import ExtractionResult, Simulation, Scene, Metric
        
        results = []
        for _, row in df.iterrows():
            sim = Simulation(Path("."), row['simulation'])
            scene = Scene(row['scene'])
            
            metrics = {}
            for col in df.columns:
                if col not in ['simulation', 'scene']:
                    metrics[col] = Metric(col, row[col])
            
            results.append(ExtractionResult(sim, scene, metrics))
        
        return Dataset(results)
    
    def exists(self, experiment: Experiment) -> bool:
        """Check if data exists"""
        exp_dir = self._experiment_dir(experiment)
        return (exp_dir / "data.csv").exists()
    
    def _experiment_dir(self, experiment: Experiment) -> Path:
        return self._data_dir / experiment.name


class CachedRepository(Repository):
    """Repository with in-memory caching"""
    
    def __init__(self, underlying: Repository):
        """
        Wrap another repository with caching
        
        Args:
            underlying: The actual storage repository
        """
        self.underlying = underlying
        self._cache = {}
    
    @property
    def data_dir(self) -> Path:
        """Delegate to underlying repository"""
        return self.underlying.data_dir
    
    def save(self, experiment: Experiment, dataset: Dataset) -> None:
        """Save and cache"""
        self.underlying.save(experiment, dataset)
        self._cache[experiment.name] = dataset
    
    def load(self, experiment: Experiment) -> Dataset:
        """Load with caching"""
        if experiment.name in self._cache:
            return self._cache[experiment.name]
        
        dataset = self.underlying.load(experiment)
        self._cache[experiment.name] = dataset
        return dataset
    
    def exists(self, experiment: Experiment) -> bool:
        """Check existence (check cache first)"""
        if experiment.name in self._cache:
            return True
        return self.underlying.exists(experiment)
    
    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
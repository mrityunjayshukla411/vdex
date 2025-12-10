"""
RT-Cache extractor - extracts per-SM cache statistics
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from core.domain import Simulation, Scene, Dataset, ExtractionResult, Metric
from parsers.rt_cache_parser import RTCacheParser
from storage.repository import Repository, ParquetRepository, CachedRepository

class RTCacheExtractor:
    """Extract RT-Cache statistics per SM"""
    
    def __init__(self, name: str = "rt_cache_experiment"):
        self._name = name
        self._simulations: List[Simulation] = []
        self._scenes: List[Scene] = []
        self._parser = RTCacheParser()
        self._repository: Repository = CachedRepository(ParquetRepository())
        self._dataset: Optional[Dataset] = None
    
    def add_simulation(self, simulation: Simulation) -> 'RTCacheExtractor':
        """Add simulation (fluent)"""
        self._simulations.append(simulation)
        return self
    
    def add_scene(self, scene: Scene) -> 'RTCacheExtractor':
        """Add scene (fluent)"""
        self._scenes.append(scene)
        return self
    
    def extract(self) -> 'RTCacheExtractor':
        """Extract RT-cache per-SM data"""
        print(f"\n{'='*60}")
        print(f"Extracting RT-Cache Per-SM: {self._name}")
        print(f"{'='*60}")
        
        all_dfs = []
        
        for simulation in self._simulations:
            print(f"\n→ {simulation.name}")
            
            for scene in self._scenes:
                try:
                    log_file = scene.log_file(simulation)
                    df = self._parser.parse(log_file)
                    
                    if not df.empty:
                        # Add simulation and scene columns
                        df.insert(0, 'simulation', simulation.name)
                        df.insert(1, 'scene', scene.name)
                        all_dfs.append(df)
                        print(f"  ✓ {scene.name} ({len(df)} SMs)")
                    else:
                        print(f"  ✗ {scene.name}: No RT-cache data")
                
                except Exception as e:
                    print(f"  ✗ {scene.name}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Save using repository
            # (For now, save DataFrame directly - could convert to Dataset)
            self._save_dataframe(combined_df)
            
            print(f"\n{'='*60}")
            print(f"✓ Extracted {len(combined_df)} SM results")
            print(f"{'='*60}\n")
        
        return self
    
    def _save_dataframe(self, df):
        """Save DataFrame as parquet"""
        exp_dir = self._repository.data_dir / self._name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = exp_dir / "rt_cache_per_sm.parquet"
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)
        print(f"✓ Saved to: {file_path}")
    
    def dataframe(self):
        """Load and return DataFrame"""
        file_path = self._repository.data_dir / self._name / "rt_cache_per_sm.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data found: {file_path}")
        
        return pd.read_parquet(file_path, engine='pyarrow')

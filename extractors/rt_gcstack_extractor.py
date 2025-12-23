"""
RT GCStack CPI Stack extractor - extracts performance breakdown statistics
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from core.domain import Simulation, Scene, Dataset, ExtractionResult, Metric
from parsers.rt_gcstack_parser import RTGCStackParser
from storage.repository import Repository, ParquetRepository, CachedRepository


class RTGCStackExtractor:
    """Extract RT GCStack CPI Stack statistics"""

    def __init__(self, name: str = "rt_gcstack_experiment"):
        self._name = name
        self._simulations: List[Simulation] = []
        self._scenes: List[Scene] = []
        self._parser = RTGCStackParser()
        self._repository: Repository = CachedRepository(ParquetRepository())
        self._dataset: Optional[Dataset] = None

    def add_simulation(self, simulation: Simulation) -> 'RTGCStackExtractor':
        """Add simulation (fluent)"""
        self._simulations.append(simulation)
        return self

    def add_scene(self, scene: Scene) -> 'RTGCStackExtractor':
        """Add scene (fluent)"""
        self._scenes.append(scene)
        return self

    def extract(self) -> 'RTGCStackExtractor':
        """Extract RT GCStack CPI Stack data"""
        print(f"\n{'='*60}")
        print(f"Extracting RT GCStack CPI Stack: {self._name}")
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
                        print(f"  ✓ {scene.name}")
                    else:
                        print(f"  ✗ {scene.name}: No RT GCStack data")

                except Exception as e:
                    print(f"  ✗ {scene.name}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Save using repository
            self._save_dataframe(combined_df)

            print(f"\n{'='*60}")
            print(f"✓ Extracted {len(combined_df)} results")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"✗ No RT GCStack data extracted")
            print(f"{'='*60}\n")

        return self

    def _save_dataframe(self, df):
        """Save DataFrame as parquet"""
        exp_dir = self._repository.data_dir / self._name
        exp_dir.mkdir(parents=True, exist_ok=True)

        file_path = exp_dir / "rt_gcstack_cpi.parquet"
        df.to_parquet(file_path, engine='pyarrow', compression='snappy', index=False)

        # Also save metadata
        self._save_metadata(exp_dir, df)

        print(f"✓ Saved to: {file_path}")

    def _save_metadata(self, exp_dir: Path, df: pd.DataFrame):
        """Save experiment metadata as JSON"""
        import json

        metadata = {
            'name': self._name,
            'type': 'gcstack_cpi_stack',
            'simulations': [s.name for s in self._simulations],
            'scenes': [s.name for s in self._scenes],
            'num_results': len(df),
            'metrics': list(df.columns[2:])  # Skip simulation and scene columns
        }

        with open(exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def save(self, also_csv: bool = False) -> 'RTGCStackExtractor':
        """
        Save extracted data (if not already saved)

        Args:
            also_csv: If True, also save as CSV

        Returns:
            self (fluent)
        """
        file_path = self._repository.data_dir / self._name / "rt_gcstack_cpi.parquet"

        if not file_path.exists():
            print("Warning: No data to save. Run extract() first.")
            return self

        if also_csv:
            csv_path = file_path.parent / "rt_gcstack_cpi.csv"
            df = pd.read_parquet(file_path, engine='pyarrow')
            df.to_csv(csv_path, index=False)
            print(f"✓ Also saved as: {csv_path}")

        return self

    def dataframe(self) -> pd.DataFrame:
        """
        Load and return DataFrame

        Returns:
            DataFrame with GCStack CPI data
        """
        file_path = self._repository.data_dir / self._name / "rt_gcstack_cpi.parquet"

        if not file_path.exists():
            raise FileNotFoundError(
                f"No data found: {file_path}\n"
                f"Run extract() first to generate data."
            )

        return pd.read_parquet(file_path, engine='pyarrow')

    def load(self) -> 'RTGCStackExtractor':
        """
        Load previously extracted data

        Returns:
            self (fluent)
        """
        file_path = self._repository.data_dir / self._name / "rt_gcstack_cpi.parquet"

        if not file_path.exists():
            raise FileNotFoundError(
                f"No data found: {file_path}\n"
                f"Run extract() first to generate data."
            )

        print(f"✓ Loaded data from: {file_path}")
        return self

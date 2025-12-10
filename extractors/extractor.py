"""
Extractor - Main orchestration with fluent interface
"""
from typing import List, Optional
from pathlib import Path

from core.domain import Simulation, Scene, Experiment, Dataset, ExtractionResult
from parsers.log_parser import LogParser, ParserFactory
from storage.repository import Repository, ParquetRepository, CachedRepository


class Extractor:
    """
    Main extractor with fluent interface
    
    Example:
        extractor = (Extractor()
            .add_simulation(Simulation(path, "Baseline"))
            .add_simulation(Simulation(path2, "RT-Cache"))
            .add_scene(Scene("sponza"))
            .add_scene(Scene("bunny"))
            .extract()
            .save())
    """
    
    def __init__(self, name: str = "experiment"):
        """
        Initialize extractor
        
        Args:
            name: Name of the experiment
        """
        self._name = name
        self._simulations: List[Simulation] = []
        self._scenes: List[Scene] = []
        self._parser: Optional[LogParser] = None
        self._repository: Repository = CachedRepository(ParquetRepository())
        self._dataset: Optional[Dataset] = None
    
    def add_simulation(self, simulation: Simulation) -> 'Extractor':
        """Add a simulation to extract from (fluent)"""
        self._simulations.append(simulation)
        return self
    
    def add_simulations(self, simulations: List[Simulation]) -> 'Extractor':
        """Add multiple simulations (fluent)"""
        self._simulations.extend(simulations)
        return self
    
    def add_scene(self, scene: Scene) -> 'Extractor':
        """Add a scene to extract (fluent)"""
        self._scenes.append(scene)
        return self
    
    def add_scenes(self, scenes: List[Scene]) -> 'Extractor':
        """Add multiple scenes (fluent)"""
        self._scenes.extend(scenes)
        return self
    
    def with_parser(self, parser: LogParser) -> 'Extractor':
        """Use a specific parser (fluent)"""
        self._parser = parser
        return self
    
    def with_repository(self, repository: Repository) -> 'Extractor':
        """Use a specific repository (fluent)"""
        self._repository = repository
        return self
    
    def extract(self, force: bool = False) -> 'Extractor':
        """
        Extract data from all simulations and scenes
        
        Args:
            force: Force re-extraction even if cached
            
        Returns:
            Self for chaining
        """
        if not self._simulations:
            raise ValueError("No simulations added. Use add_simulation() first.")
        
        if not self._scenes:
            raise ValueError("No scenes added. Use add_scene() first.")
        
        experiment = self._create_experiment()
        
        # Check cache
        if not force and self._repository.exists(experiment):
            print(f"✓ Loading cached data for '{self._name}'")
            self._dataset = self._repository.load(experiment)
            return self
        
        print(f"\n{'='*60}")
        print(f"Extracting: {self._name}")
        print(f"{'='*60}")
        print(f"Simulations: {len(self._simulations)}")
        print(f"Scenes: {len(self._scenes)}")
        
        results = []
        
        for simulation in self._simulations:
            print(f"\n→ {simulation.name}")
            
            for scene in self._scenes:
                try:
                    # Get log file
                    log_file = scene.log_file(simulation)
                    
                    # Get parser
                    parser = self._parser or ParserFactory.create(log_file)
                    
                    # Parse
                    metrics = parser.parse(log_file)
                    
                    # Create result
                    result = ExtractionResult(simulation, scene, metrics)
                    results.append(result)
                    
                    print(f"  ✓ {scene.name} ({len(metrics)} metrics)")
                    
                except Exception as e:
                    print(f"  ✗ {scene.name}: {e}")
        
        self._dataset = Dataset(results)
        
        print(f"\n{'='*60}")
        print(f"✓ Extracted {len(results)} results")
        print(f"{'='*60}\n")
        
        return self
    
    def save(self, also_csv: bool = False) -> 'Extractor':
        """
        Save extracted data
        
        Args:
            also_csv: Also save as CSV for human readability
            
        Returns:
            Self for chaining
        """
        if self._dataset is None:
            raise ValueError("No data to save. Run extract() first.")
        
        experiment = self._create_experiment()
        
        # Save to primary repository
        self._repository.save(experiment, self._dataset)
        print(f"✓ Saved to: data/{self._name}/data.parquet")
        
        # Optionally save CSV
        if also_csv:
            from storage.repository import CSVRepository
            csv_repo = CSVRepository()
            csv_repo.save(experiment, self._dataset)
            print(f"✓ Also saved: data/{self._name}/data.csv")
        
        return self
    
    def load(self) -> 'Extractor':
        """Load previously extracted data"""
        # For loading, we don't need full experiment - just dataset
        from storage.repository import ParquetRepository
        repo = ParquetRepository()
        
        # Load using just the name
        data_path = repo.data_dir / self._name / "data.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"No data found for experiment '{self._name}'")
        
        import pandas as pd
        df = pd.read_parquet(data_path, engine='pyarrow')
        
        # Convert to Dataset (simplified - doesn't need full Simulation/Scene objects)
        from core.domain import ExtractionResult, Simulation, Scene, Metric
        from pathlib import Path
        
        results = []
        for _, row in df.iterrows():
            sim = Simulation(Path("."), row['simulation'])
            scene = Scene(row['scene'])
            
            metrics = {}
            for col in df.columns:
                if col not in ['simulation', 'scene']:
                    metrics[col] = Metric(col, row[col])
            
            results.append(ExtractionResult(sim, scene, metrics))
        
        self._dataset = Dataset(results)
        print(f"✓ Loaded data for '{self._name}' ({len(self._dataset)} results)")
        return self
    
    def dataset(self) -> Dataset:
        """Get the extracted dataset"""
        if self._dataset is None:
            raise ValueError("No data available. Run extract() or load() first.")
        return self._dataset
    
    def dataframe(self):
        """Get data as pandas DataFrame"""
        return self.dataset().to_dataframe()
    
    def filter(self, simulation: Optional[str] = None, scene: Optional[str] = None) -> Dataset:
        """
        Filter dataset
        
        Args:
            simulation: Filter by simulation name
            scene: Filter by scene name
            
        Returns:
            Filtered dataset
        """
        ds = self.dataset()
        
        if simulation:
            ds = ds.filter_simulation(simulation)
        
        if scene:
            ds = ds.filter_scene(scene)
        
        return ds
    
    def _create_experiment(self) -> Experiment:
        """Create experiment object from current configuration"""
        return Experiment(
            name=self._name,
            simulations=self._simulations,
            scenes=self._scenes
        )
    
    def __repr__(self) -> str:
        status = "extracted" if self._dataset else "not extracted"
        return (f"Extractor(name='{self._name}', "
                f"simulations={len(self._simulations)}, "
                f"scenes={len(self._scenes)}, "
                f"status={status})")

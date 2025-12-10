"""
Core domain models - immutable, well-defined business entities
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd


@dataclass(frozen=True)
class Simulation:
    """Immutable simulation configuration"""
    path: Path
    name: str
    description: str = ""
    
    def __post_init__(self):
        if not isinstance(self.path, Path):
            object.__setattr__(self, 'path', Path(self.path))
        if not self.path.exists():
            raise ValueError(f"Simulation path does not exist: {self.path}")
    
    @property
    def log_dir(self) -> Path:
        """Standard location for log files"""
        return self.path


@dataclass(frozen=True)
class Scene:
    """Immutable scene identifier"""
    name: str
    
    def log_file(self, simulation: Simulation) -> Path:
        """Get log file path for this scene in a simulation"""
        # Try multiple possible locations
        candidates = [
            simulation.path / self.name / 'bin' / f'{self.name}.log',
            simulation.path / self.name / f'{self.name}.log',
            simulation.path / f'{self.name}.log',
        ]
        
        for path in candidates:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Log file not found for scene '{self.name}' in {simulation.path}")


@dataclass(frozen=True)
class Metric:
    """Immutable metric definition"""
    name: str
    value: float
    unit: Optional[str] = None
    
    def __str__(self) -> str:
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.name}: {self.value}{unit_str}"


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable extraction result"""
    simulation: Simulation
    scene: Scene
    metrics: Dict[str, Metric]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame"""
        return {
            'simulation': self.simulation.name,
            'scene': self.scene.name,
            **{m.name: m.value for m in self.metrics.values()}
        }


@dataclass
class Dataset:
    """Collection of extraction results"""
    results: List[ExtractionResult] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)
    
    def filter_simulation(self, simulation_name: str) -> 'Dataset':
        """Filter by simulation name"""
        filtered = [r for r in self.results if r.simulation.name == simulation_name]
        return Dataset(filtered)
    
    def filter_scene(self, scene_name: str) -> 'Dataset':
        """Filter by scene name"""
        filtered = [r for r in self.results if r.scene.name == scene_name]
        return Dataset(filtered)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)


@dataclass(frozen=True)
class Experiment:
    """Immutable experiment configuration"""
    name: str
    simulations: List[Simulation]
    scenes: List[Scene]
    description: str = ""
    
    def __post_init__(self):
        if not self.simulations:
            raise ValueError("Experiment must have at least one simulation")
        if not self.scenes:
            raise ValueError("Experiment must have at least one scene")

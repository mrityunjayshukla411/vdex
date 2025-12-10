"""
Vulkan-Sim Analysis Framework - Elegant, Pythonic API

Example usage:
    from vulkansim import Simulation, Scene, Extractor, Analysis
    
    # Extract
    data = (Extractor("my_experiment")
        .add_simulation(Simulation("path/to/sim1", "Baseline"))
        .add_simulation(Simulation("path/to/sim2", "RT-Cache"))
        .add_scene(Scene("sponza"))
        .add_scene(Scene("bunny"))
        .extract()
        .save())
    
    # Analyze
    analysis = Analysis(data.dataset())
    comparison = analysis.compare('gpu_tot_ipc')
    normalized = analysis.normalize('Baseline', 'gpu_tot_sim_cycle')
"""

# Core domain models
from core.domain import (
    Simulation,
    Scene,
    Metric,
    ExtractionResult,
    Dataset,
    Experiment,
)

# Extractor
from extractors.extractor import Extractor

# Analysis
from core.analysis import Analysis

# Parsers (for advanced usage)
from parsers.log_parser import (
    LogParser,
    VulkanSimParser,
    ParserFactory,
)

# Storage (for advanced usage)
from storage.repository import (
    Repository,
    ParquetRepository,
    CSVRepository,
    CachedRepository,
)

# Version
__version__ = "2.0.0"
__all__ = [
    # Core models
    'Simulation',
    'Scene',
    'Metric',
    'ExtractionResult',
    'Dataset',
    'Experiment',
    # Main interfaces
    'Extractor',
    'Analysis',
    # Parsers
    'LogParser',
    'VulkanSimParser',
    'ParserFactory',
    # Storage
    'Repository',
    'ParquetRepository',
    'CSVRepository',
    'CachedRepository',
]


# Convenience functions

def extract(
    name: str,
    simulations: list,
    scenes: list,
    save: bool = True
) -> Extractor:
    """
    Convenience function for quick extraction
    
    Args:
        name: Experiment name
        simulations: List of (path, name) tuples or Simulation objects
        scenes: List of scene names or Scene objects
        save: Whether to save after extraction
        
    Returns:
        Extractor with extracted data
        
    Example:
        data = extract(
            "my_experiment",
            [("path/to/sim1", "Baseline"), ("path/to/sim2", "RT-Cache")],
            ["sponza", "bunny"]
        )
    """
    extractor = Extractor(name)
    
    # Add simulations
    for sim in simulations:
        if isinstance(sim, Simulation):
            extractor.add_simulation(sim)
        else:
            path, sim_name = sim
            extractor.add_simulation(Simulation(path, sim_name))
    
    # Add scenes
    for scene in scenes:
        if isinstance(scene, Scene):
            extractor.add_scene(scene)
        else:
            extractor.add_scene(Scene(scene))
    
    # Extract
    extractor.extract()
    
    if save:
        extractor.save()
    
    return extractor


def load(name: str) -> Extractor:
    """
    Load previously extracted experiment
    
    Args:
        name: Experiment name
        
    Returns:
        Extractor with loaded data
        
    Example:
        data = load("my_experiment")
        df = data.dataframe()
    """
    extractor = Extractor(name)
    extractor.load()
    return extractor


def analyze(dataset: Dataset) -> Analysis:
    """
    Create analysis from dataset
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Analysis object
        
    Example:
        analysis = analyze(extractor.dataset())
        comparison = analysis.compare('gpu_tot_ipc')
    """
    return Analysis(dataset)

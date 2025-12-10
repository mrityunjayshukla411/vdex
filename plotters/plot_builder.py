"""
Plot orchestrator - Fluent interface for creating plots
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from core.domain import Dataset
from core.analysis import Analysis
from plotters.grouped_bar_plotter import GroupedBarPlotter
from plotters.normalized_bar_plotter import NormalizedBarPlotter


class PlotBuilder:
    """
    Fluent interface for creating plots
    
    Example:
        plot = (PlotBuilder(data)
            .metric('gpu_tot_ipc')
            .grouped_bar()
            .save('output.png'))
    """
    
    def __init__(self, data: Union[pd.DataFrame, Dataset, Analysis]):
        """
        Initialize plot builder
        
        Args:
            data: DataFrame, Dataset, or Analysis object
        """
        if isinstance(data, Analysis):
            self.df = data.df
        elif isinstance(data, Dataset):
            self.df = data.to_dataframe()
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise TypeError("data must be DataFrame, Dataset, or Analysis")
        
        # Configuration
        self._metric: Optional[str] = None
        self._baseline: Optional[str] = None
        self._output: Optional[Path] = None
        self._title: Optional[str] = None
        self._xlabel: Optional[str] = None
        self._ylabel: Optional[str] = None
        self._figsize: tuple = (15, 8)
        self._rotation: int = 45
        self._show_values: bool = False
        self._grayscale: bool = False
        self._no_patterns: bool = False
        self._mean_type: str = 'geomean'
    
    def metric(self, name: str) -> 'PlotBuilder':
        """Set metric to plot (fluent)"""
        self._metric = name
        return self
    
    def baseline(self, name: str) -> 'PlotBuilder':
        """Set baseline for normalization (fluent)"""
        self._baseline = name
        return self
    
    def output(self, path: Union[str, Path]) -> 'PlotBuilder':
        """Set output file (fluent)"""
        self._output = Path(path) if isinstance(path, str) else path
        return self
    
    def title(self, text: str) -> 'PlotBuilder':
        """Set custom title (fluent)"""
        self._title = text
        return self
    
    def labels(self, xlabel: str = None, ylabel: str = None) -> 'PlotBuilder':
        """Set axis labels (fluent)"""
        if xlabel:
            self._xlabel = xlabel
        if ylabel:
            self._ylabel = ylabel
        return self
    
    def figsize(self, width: float, height: float) -> 'PlotBuilder':
        """Set figure size (fluent)"""
        self._figsize = (width, height)
        return self
    
    def rotation(self, degrees: int) -> 'PlotBuilder':
        """Set x-axis label rotation (fluent)"""
        self._rotation = degrees
        return self
    
    def show_values(self, show: bool = True) -> 'PlotBuilder':
        """Show values on bars (fluent)"""
        self._show_values = show
        return self
    
    def grayscale(self, enabled: bool = True) -> 'PlotBuilder':
        """Use grayscale palette (fluent)"""
        self._grayscale = enabled
        return self
    
    def no_patterns(self, disabled: bool = True) -> 'PlotBuilder':
        """Disable hatching patterns (fluent)"""
        self._no_patterns = disabled
        return self
    
    def mean_type(self, type_: str) -> 'PlotBuilder':
        """Set mean type: 'mean' or 'geomean' (fluent)"""
        if type_ not in ['mean', 'geomean']:
            raise ValueError("mean_type must be 'mean' or 'geomean'")
        self._mean_type = type_
        return self
    
    def grouped_bar(self) -> 'PlotBuilder':
        """Create grouped bar chart (fluent)"""
        if not self._metric:
            raise ValueError("Must set metric before plotting")
        
        plotter = GroupedBarPlotter()
        plotter.plot(
            self.df,
            self._metric,
            output_file=self._output,
            title=self._title,
            xlabel=self._xlabel,
            ylabel=self._ylabel,
            figsize=self._figsize,
            rotation=self._rotation,
            show_values=self._show_values,
            grayscale=self._grayscale,
            no_patterns=self._no_patterns,
            mean_type=self._mean_type
        )
        return self
    
    def normalized_bar(self) -> 'PlotBuilder':
        """Create normalized bar chart (fluent)"""
        if not self._metric:
            raise ValueError("Must set metric before plotting")
        
        if not self._baseline:
            # Use first simulation as baseline
            self._baseline = self.df['simulation'].iloc[0]
            print(f"Using '{self._baseline}' as baseline")
        
        plotter = NormalizedBarPlotter()
        plotter.plot(
            self.df,
            self._metric,
            self._baseline,
            output_file=self._output,
            title=self._title,
            xlabel=self._xlabel,
            ylabel=self._ylabel,
            figsize=self._figsize,
            grayscale=self._grayscale,
            no_patterns=self._no_patterns,
            mean_type=self._mean_type
        )
        return self
    
    def save(self, path: Union[str, Path]) -> 'PlotBuilder':
        """
        Convenience method: set output and plot
        
        Args:
            path: Output file path
            
        Returns:
            Self for chaining (already plotted)
        """
        self._output = Path(path) if isinstance(path, str) else path
        # Plot was already called in grouped_bar/normalized_bar
        return self


# Convenience functions

def plot(data: Union[pd.DataFrame, Dataset, Analysis]) -> PlotBuilder:
    """
    Create plot builder (convenience function)
    
    Args:
        data: Data to plot
        
    Returns:
        PlotBuilder for fluent interface
        
    Example:
        plot(data).metric('ipc').grouped_bar().save('output.png')
    """
    return PlotBuilder(data)


def quick_plot(
    data: Union[pd.DataFrame, Dataset, Analysis],
    metric: str,
    output: Optional[Union[str, Path]] = None,
    normalized: bool = False,
    baseline: Optional[str] = None
):
    """
    Quick plot without fluent interface
    
    Args:
        data: Data to plot
        metric: Metric name
        output: Output file path
        normalized: Create normalized plot
        baseline: Baseline for normalization
        
    Example:
        quick_plot(data, 'gpu_tot_ipc', 'ipc.png')
        quick_plot(data, 'cycles', 'norm.png', normalized=True, baseline='Baseline')
    """
    builder = PlotBuilder(data).metric(metric)
    
    if output:
        builder.output(output)
    
    if normalized:
        if baseline:
            builder.baseline(baseline)
        builder.normalized_bar()
    else:
        builder.grouped_bar()

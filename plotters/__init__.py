"""
Plotters module - Elegant plotting with fluent interface
"""

from plotters.plotter import Plotter, ColorPalette
from plotters.grouped_bar_plotter import GroupedBarPlotter
from plotters.normalized_bar_plotter import NormalizedBarPlotter
from plotters.plot_builder import PlotBuilder, plot, quick_plot

__all__ = [
    'Plotter',
    'ColorPalette',
    'GroupedBarPlotter',
    'NormalizedBarPlotter',
    'PlotBuilder',
    'plot',
    'quick_plot',
]

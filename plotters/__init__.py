"""
Plotters module - Elegant plotting with fluent interface
"""

from plotters.plotter import Plotter, ColorPalette
from plotters.grouped_bar_plotter import GroupedBarPlotter
from plotters.normalized_bar_plotter import NormalizedBarPlotter
from plotters.gcstack_plotter import GCStackPlotter, plot_gcstack
from plotters.rt_gcstack_plotter import RTGCStackPlotter, plot_rt_gcstack
from plotters.plot_builder import PlotBuilder, plot, quick_plot

__all__ = [
    'Plotter',
    'ColorPalette',
    'GroupedBarPlotter',
    'NormalizedBarPlotter',
    'GCStackPlotter',
    'plot_gcstack',
    'RTGCStackPlotter',
    'plot_rt_gcstack',
    'PlotBuilder',
    'plot',
    'quick_plot',
]

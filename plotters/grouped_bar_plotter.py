"""
Grouped bar chart plotter
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotters.plotter import Plotter, ColorPalette, order_scenes


class GroupedBarPlotter(Plotter):
    """Create grouped bar charts comparing metrics across simulations"""
    
    def plot(
        self,
        data: pd.DataFrame,
        metric: str,
        output_file: Optional[Path] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (15, 8),
        rotation: int = 45,
        show_values: bool = False,
        grayscale: bool = False,
        no_patterns: bool = False,
        mean_type: str = 'geomean'
    ) -> None:
        """
        Create grouped bar chart
        
        Args:
            data: DataFrame with columns: simulation, scene, <metric>
            metric: Metric name to plot
            output_file: Path to save figure
            title: Custom title
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label
            figsize: Figure size (width, height)
            rotation: X-axis label rotation
            show_values: Show values on bars
            grayscale: Use grayscale palette
            no_patterns: Disable hatching patterns
            mean_type: 'mean' or 'geomean' for aggregation
        """
        # Validate
        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        # Prepare data
        scenes = order_scenes(data['scene'].unique())
        simulations = data['simulation'].unique()
        n_scenes = len(scenes)
        n_sims = len(simulations)
        
        # Build data matrix (sims × scenes + mean)
        data_matrix = np.zeros((n_sims, n_scenes + 1))
        
        for sim_idx, sim in enumerate(simulations):
            sim_data = data[data['simulation'] == sim]
            values = []
            
            for scene_idx, scene in enumerate(scenes):
                scene_val = sim_data[sim_data['scene'] == scene][metric]
                if not scene_val.empty:
                    val = scene_val.iloc[0]
                    if pd.notna(val) and val > 0:
                        data_matrix[sim_idx, scene_idx] = val
                        values.append(val)
            
            # Compute mean
            if values:
                if mean_type == 'mean':
                    data_matrix[sim_idx, n_scenes] = np.mean(values)
                else:  # geomean
                    data_matrix[sim_idx, n_scenes] = np.exp(np.mean(np.log(values)))
        
        # Create plot
        fig, ax = self._setup_figure(figsize)
        
        # Get colors and patterns
        colors = ColorPalette.get_palette(n_sims, grayscale)
        patterns = ColorPalette.get_patterns(n_sims, no_patterns)
        
        # Plot bars
        bar_width = 0.8 / n_sims
        x_pos = np.arange(n_scenes + 1)
        
        for sim_idx, sim_name in enumerate(simulations):
            offset = (sim_idx - n_sims/2 + 0.5) * bar_width
            display_name = sim_name.replace('_', ' ')
            
            bars = ax.bar(
                x_pos + offset,
                data_matrix[sim_idx],
                bar_width,
                label=display_name,
                color=colors[sim_idx],
                edgecolor='black',
                linewidth=0.8,
                hatch=patterns[sim_idx],
                alpha=0.85
            )
            
            # Add value labels
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height,
                            f'{height:.1f}',
                            ha='center',
                            va='bottom',
                            fontsize=7
                        )
        
        # Separator line before mean
        ax.axvline(x=n_scenes - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Labels
        scene_labels = scenes + [mean_type.capitalize()]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scene_labels, rotation=rotation, ha='right', fontsize=11)
        
        # Make mean label bold
        labels = ax.get_xticklabels()
        labels[-1].set_weight('bold')
        
        # Apply styling
        self._apply_style(
            ax,
            xlabel or 'Benchmark',
            ylabel or metric,
            title
        )
        
        # Legend
        if n_sims <= 4:
            legend = ax.legend(loc='upper right', fontsize=10, frameon=True,
                              edgecolor='black', framealpha=0.95)
        else:
            legend = ax.legend(loc='center', fontsize=9, frameon=False,
                              ncol=3, bbox_to_anchor=(0.5, 1.06))
        
        if legend.get_frame():
            legend.get_frame().set_linewidth(0.8)
        
        plt.tight_layout()
        self._save_or_show(fig, output_file)

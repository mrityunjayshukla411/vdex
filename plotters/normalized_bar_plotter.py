"""
Normalized bar chart plotter
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotters.plotter import Plotter, ColorPalette


class NormalizedBarPlotter(Plotter):
    """Create normalized bar charts (relative to baseline)"""
    
    def plot(
        self,
        data: pd.DataFrame,
        metric: str,
        baseline: str,
        output_file: Optional[Path] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (15, 8),
        grayscale: bool = False,
        no_patterns: bool = False,
        mean_type: str = 'geomean'
    ) -> None:
        """
        Create normalized bar chart
        
        Args:
            data: DataFrame with columns: simulation, scene, <metric>
            metric: Metric name to plot
            baseline: Baseline simulation name
            output_file: Path to save figure
            title: Custom title
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label
            figsize: Figure size
            grayscale: Use grayscale palette
            no_patterns: Disable patterns
            mean_type: 'mean' or 'geomean'
        """
        # Validate
        if metric not in data.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        if baseline not in data['simulation'].values:
            raise ValueError(f"Baseline '{baseline}' not found")
        
        # Get baseline data
        baseline_data = data[data['simulation'] == baseline]
        baseline_values = {}
        for _, row in baseline_data.iterrows():
            baseline_values[row['scene']] = row[metric] if row[metric] != 0 else 1.0
        
        # Prepare normalized data
        scenes = sorted(data['scene'].unique())
        simulations = [s for s in data['simulation'].unique() if s != baseline]
        n_scenes = len(scenes)
        n_sims = len(simulations)
        
        if n_sims == 0:
            raise ValueError("Only baseline provided, nothing to compare")
        
        data_matrix = np.zeros((n_sims, n_scenes + 1))
        
        for sim_idx, sim in enumerate(simulations):
            sim_data = data[data['simulation'] == sim]
            normalized_values = []
            
            for scene_idx, scene in enumerate(scenes):
                scene_val = sim_data[sim_data['scene'] == scene][metric]
                if not scene_val.empty:
                    val = scene_val.iloc[0]
                    baseline_val = baseline_values.get(scene, 1.0)
                    if pd.notna(val) and baseline_val != 0:
                        normalized = val / baseline_val
                        data_matrix[sim_idx, scene_idx] = normalized
                        normalized_values.append(normalized)
                    else:
                        data_matrix[sim_idx, scene_idx] = 1.0
                else:
                    data_matrix[sim_idx, scene_idx] = 1.0
            
            # Compute mean
            if normalized_values:
                if mean_type == 'mean':
                    data_matrix[sim_idx, n_scenes] = np.mean(normalized_values)
                else:
                    data_matrix[sim_idx, n_scenes] = np.exp(np.mean(np.log(normalized_values)))
            else:
                data_matrix[sim_idx, n_scenes] = 1.0
        
        # Create plot
        fig, ax = self._setup_figure(figsize)
        
        # Colors and patterns
        colors = ColorPalette.get_palette(n_sims, grayscale)
        patterns = ColorPalette.get_patterns(n_sims, no_patterns)
        
        # Plot bars
        bar_width = 0.8 / n_sims
        x_pos = np.arange(n_scenes + 1)
        
        for sim_idx, sim_name in enumerate(simulations):
            offset = (sim_idx - n_sims/2 + 0.5) * bar_width
            display_name = sim_name.replace('_', ' ')
            
            ax.bar(
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
        
        # Baseline line at y=1
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.2, alpha=0.7, zorder=0)
        
        # Separator before mean
        ax.axvline(x=n_scenes - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Labels
        scene_labels = scenes + [mean_type.capitalize()]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scene_labels, rotation=45, ha='right', fontsize=11)
        
        labels = ax.get_xticklabels()
        labels[-1].set_weight('bold')
        
        # Y-axis limits
        min_val = np.min(data_matrix[data_matrix > 0])
        max_val = np.max(data_matrix)
        y_margin = (max_val - min_val) * 0.1
        
        if min_val < 1.0:
            ax.set_ylim(bottom=max(0, min_val - y_margin))
        else:
            ax.set_ylim(bottom=0.9)
        
        # Apply styling
        self._apply_style(
            ax,
            xlabel or 'Benchmark',
            ylabel or f'Normalized {metric}\n(relative to {baseline})',
            title or f'Normalized {metric} (Baseline: {baseline})'
        )
        
        # Legend
        if n_sims <= 4:
            legend = ax.legend(loc='upper right', fontsize=10, frameon=True,
                              edgecolor='black', framealpha=0.95)
        else:
            legend = ax.legend(loc='center', fontsize=9, frameon=False,
                              ncol=2, bbox_to_anchor=(0.5, 1.12))
        
        if legend.get_frame():
            legend.get_frame().set_linewidth(0.8)
        
        plt.tight_layout()
        self._save_or_show(fig, output_file)

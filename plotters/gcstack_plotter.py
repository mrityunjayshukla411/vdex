"""
GCStack CPI Stack plotter - Stacked bar charts for performance breakdown
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotters.plotter import Plotter, ColorPalette, order_scenes


class GCStackPlotter(Plotter):
    """Create stacked bar charts for GCStack CPI breakdown"""

    # GCStack components in order (base first as requested)
    COMPONENTS = ['base', 'memstruct', 'memdata', 'sync',
                  'comstruct', 'comdata', 'control', 'idle']

    # Component display names
    COMPONENT_LABELS = {
        'base': 'Base',
        'memstruct': 'Mem Struct',
        'memdata': 'Mem Data',
        'sync': 'Sync',
        'comstruct': 'Com Struct',
        'comdata': 'Com Data',
        'control': 'Control',
        'idle': 'Idle'
    }

    # Color palette for components (distinct, colorblind-friendly)
    COMPONENT_COLORS = [
        '#4C72B0',  # Base - blue
        '#DD8452',  # MemStruct - orange
        '#C44E52',  # MemData - red
        '#8172B3',  # Sync - purple
        '#55A868',  # ComStruct - green
        '#64B5CD',  # ComData - cyan
        '#937860',  # Control - brown
        '#CCCCCC',  # Idle - gray
    ]

    def plot(
        self,
        data: pd.DataFrame,
        output_file: Optional[Path] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple = (14, 8),
        rotation: int = 0,
        show_legend: bool = True,
        legend_position: str = 'right',
        use_hatches: bool = True,
        normalize: bool = True
    ) -> None:
        """
        Create grouped stacked bar chart for GCStack CPI breakdown

        Args:
            data: DataFrame with columns: simulation, scene, base, memstruct, ...
            output_file: Path to save figure
            title: Custom title
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label (default: "CPI Fraction" or "Cycles")
            figsize: Figure size (width, height)
            rotation: X-axis label rotation
            show_legend: Show legend
            legend_position: 'right' or 'top'
            use_hatches: Use hatch patterns to distinguish simulations
            normalize: Normalize to percentages (0-1 scale)
        """
        # Validate required columns
        required = ['simulation', 'scene', 'total'] + self.COMPONENTS
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Calculate percentages if normalizing
        df = data.copy()
        if normalize:
            for component in self.COMPONENTS:
                df[component] = df[component] / df['total']

        # Get unique scenes and simulations
        scenes = order_scenes(df['scene'].unique())
        simulations = df['simulation'].unique()
        n_scenes = len(scenes)
        n_sims = len(simulations)

        # Create figure
        fig, ax = self._setup_figure(figsize)

        # Calculate bar positions
        bar_width = 0.8 / n_sims
        x_pos = np.arange(n_scenes)

        # Get hatch patterns for simulations
        if use_hatches:
            hatch_patterns = ColorPalette.get_patterns(n_sims)
        else:
            hatch_patterns = [''] * n_sims

        # Plot for each simulation
        for sim_idx, sim_name in enumerate(simulations):
            sim_data = df[df['simulation'] == sim_name]

            # Calculate offset for this simulation's bars
            offset = (sim_idx - n_sims/2 + 0.5) * bar_width
            positions = x_pos + offset

            # Initialize bottom for stacking
            bottom = np.zeros(n_scenes)

            # Plot each component
            for comp_idx, component in enumerate(self.COMPONENTS):
                values = []
                for scene in scenes:
                    scene_data = sim_data[sim_data['scene'] == scene]
                    if not scene_data.empty:
                        val = scene_data[component].iloc[0]
                        values.append(val if pd.notna(val) else 0.0)
                    else:
                        values.append(0.0)

                values = np.array(values)

                # Only show label for first simulation (avoid duplicate legend entries)
                label = self.COMPONENT_LABELS[component] if sim_idx == 0 else None

                # Plot stacked bar
                ax.bar(
                    positions,
                    values,
                    bar_width,
                    bottom=bottom,
                    label=label,
                    color=self.COMPONENT_COLORS[comp_idx],
                    edgecolor='black',
                    linewidth=0.7,
                    hatch=hatch_patterns[sim_idx],
                    alpha=0.9
                )

                # Update bottom for next component
                bottom += values

        # Create custom legend for simulations (using hatches)
        if use_hatches and n_sims > 1:
            # Add simulation legend entries
            from matplotlib.patches import Patch
            sim_legend_elements = []
            for sim_idx, sim_name in enumerate(simulations):
                patch = Patch(
                    facecolor='white',
                    edgecolor='black',
                    hatch=hatch_patterns[sim_idx],
                    label=sim_name.replace('_', ' ')
                )
                sim_legend_elements.append(patch)

            # Create two legends: one for components, one for simulations
            if show_legend:
                # Component legend (colors)
                legend1 = ax.legend(
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=10,
                    frameon=True,
                    edgecolor='black',
                    title='Stall classes',
                    title_fontsize=11
                )
                legend1.get_frame().set_linewidth(0.8)

                # Simulation legend (hatches)
                legend2 = ax.legend(
                    handles=sim_legend_elements,
                    loc='center left',
                    bbox_to_anchor=(1.02, 0.2),
                    fontsize=10,
                    frameon=True,
                    edgecolor='black',
                    title='Configurations',
                    title_fontsize=11
                )
                legend2.get_frame().set_linewidth(0.8)

                # Add first legend back (matplotlib replaces it)
                ax.add_artist(legend1)
        else:
            # Single legend for components
            if show_legend:
                if legend_position == 'top':
                    legend = ax.legend(
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.08),
                        ncol=len(self.COMPONENTS),
                        fontsize=9,
                        frameon=True,
                        edgecolor='black'
                    )
                else:
                    legend = ax.legend(
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=10,
                        frameon=True,
                        edgecolor='black'
                    )
                legend.get_frame().set_linewidth(0.8)

        # Set labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ') for s in scenes],
                           rotation=rotation, ha='center' if rotation == 0 else 'right',
                           fontsize=11)

        # Set y-axis limits
        if normalize:
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

        # Apply styling
        self._apply_style(
            ax,
            xlabel or 'Scene',
            ylabel or ('CPI Fraction' if normalize else 'Cycles'),
            title
        )

        # Adjust layout to make room for legends
        plt.tight_layout()
        if show_legend and legend_position == 'right':
            plt.subplots_adjust(right=0.82)

        # Save or show
        self._save_or_show(fig, output_file)

    def plot_comparison(
        self,
        data: pd.DataFrame,
        baseline_sim: str,
        output_file: Optional[Path] = None,
        title: Optional[str] = None,
        figsize: tuple = (14, 8)
    ) -> None:
        """
        Create side-by-side comparison of baseline vs other simulations

        Args:
            data: DataFrame with GCStack data
            baseline_sim: Name of baseline simulation
            output_file: Path to save figure
            title: Custom title
            figsize: Figure size
        """
        # Get baseline and other simulations
        simulations = data['simulation'].unique()
        if baseline_sim not in simulations:
            raise ValueError(f"Baseline '{baseline_sim}' not found in data")

        other_sims = [s for s in simulations if s != baseline_sim]

        # Create subplots
        n_cols = len(other_sims) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
        if n_cols == 1:
            axes = [axes]

        # Plot baseline
        self._plot_single_sim(axes[0], data, baseline_sim, f"{baseline_sim}\n(Baseline)")

        # Plot other simulations
        for idx, sim in enumerate(other_sims):
            self._plot_single_sim(axes[idx + 1], data, sim, sim)

        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Create shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.05),
                  ncol=len(self.COMPONENTS), fontsize=10, frameon=True,
                  edgecolor='black')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self._save_or_show(fig, output_file)

    def _plot_single_sim(self, ax, data: pd.DataFrame, simulation: str, title: str):
        """Helper to plot a single simulation"""
        sim_data = data[data['simulation'] == simulation].copy()

        # Calculate percentages
        for component in self.COMPONENTS:
            sim_data[component] = sim_data[component] / sim_data['total']

        scenes = order_scenes(sim_data['scene'].unique())
        x_pos = np.arange(len(scenes))

        # Plot stacked bars
        bottom = np.zeros(len(scenes))
        for comp_idx, component in enumerate(self.COMPONENTS):
            values = [sim_data[sim_data['scene'] == scene][component].iloc[0]
                     for scene in scenes]

            ax.bar(
                x_pos,
                values,
                bottom=bottom,
                label=self.COMPONENT_LABELS[component],
                color=self.COMPONENT_COLORS[comp_idx],
                edgecolor='black',
                linewidth=0.7,
                alpha=0.9
            )
            bottom += values

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenes, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
        ax.set_ylabel('CPI Fraction', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)


# Convenience function
def plot_gcstack(
    data: pd.DataFrame,
    output_file: Path,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Quick GCStack plotting function

    Args:
        data: DataFrame with GCStack data
        output_file: Path to save plot
        title: Plot title
        **kwargs: Additional arguments for GCStackPlotter.plot()

    Example:
        plot_gcstack(df, Path('plots/gcstack.png'), title='My Analysis')
    """
    # Ensure plots directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plotter = GCStackPlotter()
    plotter.plot(data, output_file, title=title, **kwargs)

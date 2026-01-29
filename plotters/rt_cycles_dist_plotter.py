"""
RT Cycles Distribution plotter - Stacked bar charts for RT unit cycle distribution
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plotters.plotter import Plotter, ColorPalette, order_scenes


class RTCyclesDistPlotter(Plotter):
    """Create stacked bar charts for RT Cycles Distribution"""

    # Professional color palette (colorblind-friendly, publication-quality)
    BASE_COLORS = [
    '#2F3E46',  # RT Unit 0 - dark slate gray
    '#8D5A4A',  # RT Unit 1 - muted brick
    '#52796F',  # RT Unit 2 - desaturated teal
    '#6C757D',  # RT Unit 3 - neutral gray-blue
    '#7A6C9D',  # RT Unit 4 - muted purple
    '#B08968',  # RT Unit 5 - dull tan
    '#4F6D7A',  # RT Unit 6 - steel blue
    '#3A5A40',  # RT Unit 7 - forest green
    '#9C6644',  # RT Unit 8 - burnt sienna
    '#6B705C',  # RT Unit 9 - olive gray
    '#5E548E',  # RT Unit 10 - dusty indigo
    '#495057',  # RT Unit 11 - charcoal
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
        normalize: bool = False
    ) -> None:
        """
        Create grouped stacked bar chart for RT Cycles Distribution

        Args:
            data: DataFrame with columns: simulation, scene, rt_unit_0, rt_unit_1, ...
            output_file: Path to save figure
            title: Custom title
            xlabel: Custom x-axis label
            ylabel: Custom y-axis label (default: "Cycles" or "Fraction")
            figsize: Figure size (width, height)
            rotation: X-axis label rotation
            show_legend: Show legend
            legend_position: 'right' or 'top'
            use_hatches: Use hatch patterns to distinguish simulations
            normalize: Normalize to percentages (0-1 scale)
        """
        # Copy dataframe
        df = data.copy()

        # Detect RT unit columns dynamically
        rt_unit_cols = [col for col in df.columns if col.startswith('rt_unit_')]
        rt_unit_cols.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by unit number

        if not rt_unit_cols:
            raise ValueError("No rt_unit_* columns found in data")

        n_units = len(rt_unit_cols)

        # Validate required columns
        required = ['simulation', 'scene'] + rt_unit_cols
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Calculate total cycles per row for normalization
        df['total_cycles'] = df[rt_unit_cols].sum(axis=1)

        # Calculate percentages if normalizing
        if normalize:
            for col in rt_unit_cols:
                df[col] = df[col] / df['total_cycles']

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

        # Generate colors for RT units (dynamic based on number of units)
        unit_colors = ColorPalette.generate_colors(self.BASE_COLORS, n_units)

        # Create component labels
        unit_labels = {col: f'RT Unit {col.split("_")[-1]}' for col in rt_unit_cols}

        # Plot for each simulation
        for sim_idx, sim_name in enumerate(simulations):
            sim_data = df[df['simulation'] == sim_name]

            # Calculate offset for this simulation's bars
            offset = (sim_idx - n_sims/2 + 0.5) * bar_width
            positions = x_pos + offset

            # Initialize bottom for stacking
            bottom = np.zeros(n_scenes)

            # Plot each RT unit
            for unit_idx, unit_col in enumerate(rt_unit_cols):
                values = []
                for scene in scenes:
                    scene_data = sim_data[sim_data['scene'] == scene]
                    if not scene_data.empty:
                        val = scene_data[unit_col].iloc[0]
                        values.append(val if pd.notna(val) else 0.0)
                    else:
                        values.append(0.0)

                values = np.array(values)

                # Only show label for first simulation (avoid duplicate legend entries)
                label = unit_labels[unit_col] if sim_idx == 0 else None

                # Plot stacked bar
                ax.bar(
                    positions,
                    values,
                    bar_width,
                    bottom=bottom,
                    label=label,
                    color=unit_colors[unit_idx],
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

            # Create two figure legends: one for RT units, one for simulations
            if show_legend:
                # Get handles and labels from axes for RT units
                handles, labels = ax.get_legend_handles_labels()

                # # RT unit legend (colors) - placed at top right outside plot
                # legend1 = fig.legend(
                #     handles, labels,
                #     loc='upper left',
                #     bbox_to_anchor=(0.03, 1.04),
                #     ncol=n_units,
                #     fontsize=10,
                #     frameon=False,
                #     edgecolor='black',
                #     title_fontsize=11
                # )
                # legend1.get_frame().set_linewidth(0.8)

                # Simulation legend (hatches) - placed below RT units legend
                legend2 = fig.legend(
                    handles=sim_legend_elements,
                    loc='upper left',
                    bbox_to_anchor=(0.3, 1.05),
                    ncol=n_sims,
                    frameon=False,
                    edgecolor='black'
                )
                legend2.get_frame().set_linewidth(0.8)
        else:
            # Single figure legend for RT units
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                if legend_position == 'top':
                    legend = fig.legend(
                        handles, labels,
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.02),
                        ncol=2,
                        frameon=False,
                        edgecolor='black'
                    )
                else:
                    legend = fig.legend(
                        handles, labels,
                        loc='upper left',
                        bbox_to_anchor=(0.85, 0.88),
                        frameon=False,
                        edgecolor='black',
                        title='RT Units'
                    )
                legend.get_frame().set_linewidth(0.8)

        # Set labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ') for s in scenes],
                           rotation=rotation, ha='center' if rotation == 0 else 'right')

        # Remove x-axis padding
        ax.set_xlim(-0.5, n_scenes - 0.5)

        # Set y-axis limits
        if normalize:
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

        # Apply styling
        self._apply_style(
            ax,
            xlabel,
            ylabel or ('Cycle Fraction' if normalize else 'Cycles'),
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
        figsize: tuple = (14, 8),
        normalize: bool = False
    ) -> None:
        """
        Create side-by-side comparison of baseline vs other simulations

        Args:
            data: DataFrame with RT Cycles Distribution data
            baseline_sim: Name of baseline simulation
            output_file: Path to save figure
            title: Custom title
            figsize: Figure size
            normalize: Normalize to percentages
        """
        df = data.copy()

        # Detect RT unit columns dynamically
        rt_unit_cols = [col for col in df.columns if col.startswith('rt_unit_')]
        rt_unit_cols.sort(key=lambda x: int(x.split('_')[-1]))

        if not rt_unit_cols:
            raise ValueError("No rt_unit_* columns found in data")

        n_units = len(rt_unit_cols)

        # Generate colors for RT units
        unit_colors = ColorPalette.generate_colors(self.BASE_COLORS, n_units)
        unit_labels = {col: f'RT Unit {col.split("_")[-1]}' for col in rt_unit_cols}

        # Get baseline and other simulations
        simulations = df['simulation'].unique()
        if baseline_sim not in simulations:
            raise ValueError(f"Baseline '{baseline_sim}' not found in data")

        other_sims = [s for s in simulations if s != baseline_sim]

        # Create subplots
        n_cols = len(other_sims) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
        if n_cols == 1:
            axes = [axes]

        # Plot baseline
        self._plot_single_sim(
            axes[0], df, baseline_sim, f"{baseline_sim}\n(Baseline)",
            rt_unit_cols, unit_colors, unit_labels, normalize
        )

        # Plot other simulations
        for idx, sim in enumerate(other_sims):
            self._plot_single_sim(
                axes[idx + 1], df, sim, sim,
                rt_unit_cols, unit_colors, unit_labels, normalize
            )

        # Add overall title
        if title:
            fig.suptitle(title, y=0.98)

        # Create shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.05),
                  ncol=min(n_units, 6), frameon=True,
                  edgecolor='black')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        self._save_or_show(fig, output_file)

    def _plot_single_sim(
        self, ax, data: pd.DataFrame, simulation: str, title: str,
        rt_unit_cols: List[str], unit_colors: List[str],
        unit_labels: dict, normalize: bool
    ):
        """Helper to plot a single simulation"""
        sim_data = data[data['simulation'] == simulation].copy()

        # Calculate total cycles
        sim_data['total_cycles'] = sim_data[rt_unit_cols].sum(axis=1)

        # Calculate percentages if normalizing
        if normalize:
            for col in rt_unit_cols:
                sim_data[col] = sim_data[col] / sim_data['total_cycles']

        scenes = order_scenes(sim_data['scene'].unique())
        x_pos = np.arange(len(scenes))

        # Plot stacked bars
        bottom = np.zeros(len(scenes))
        for unit_idx, unit_col in enumerate(rt_unit_cols):
            values = [sim_data[sim_data['scene'] == scene][unit_col].iloc[0]
                     for scene in scenes]

            ax.bar(
                x_pos,
                values,
                bottom=bottom,
                label=unit_labels[unit_col],
                color=unit_colors[unit_idx],
                edgecolor='black',
                linewidth=0.7,
                alpha=0.9
            )
            bottom += values

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenes, rotation=45, ha='right')

        if normalize:
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
            ax.set_ylabel('Cycle Fraction')
        else:
            ax.set_ylabel('Cycles')

        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)


# Convenience function
def plot_rt_cycles_dist(
    data: pd.DataFrame,
    output_file: Path,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Quick RT Cycles Distribution plotting function

    Args:
        data: DataFrame with RT Cycles Distribution data
        output_file: Path to save plot
        title: Plot title
        **kwargs: Additional arguments for RTCyclesDistPlotter.plot()

    Example:
        plot_rt_cycles_dist(df, Path('plots/rt_cycles_dist.png'), title='My Analysis')
    """
    # Ensure plots directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plotter = RTCyclesDistPlotter()
    plotter.plot(data, output_file, title=title, **kwargs)

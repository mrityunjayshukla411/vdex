"""
Plotter interface - Strategy pattern for different plot types
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Plotter(ABC):
    """Abstract plotter interface - implement for different plot types"""
    
    @abstractmethod
    def plot(
        self,
        data: pd.DataFrame,
        output_file: Optional[Path] = None,
        **kwargs
    ) -> None:
        """
        Create a plot from data
        
        Args:
            data: DataFrame with data to plot
            output_file: Path to save figure (None = display)
            **kwargs: Plot-specific options
        """
        pass
    
    def _save_or_show(self, fig, output_file: Optional[Path]) -> None:
        """Save figure or display it"""
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to: {output_file}")
        else:
            plt.show()
        plt.close()
    
    def _setup_figure(self, figsize: tuple = (12, 6)):
        """Create figure with standard setup"""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    def _apply_style(self, ax, xlabel: str, ylabel: str, title: str):
        """Apply consistent styling to axes"""
        ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        
        # Ticks
        ax.minorticks_on()
        ax.tick_params(axis='y', which='major', labelsize=10, length=6, width=0.8)
        ax.tick_params(axis='y', which='minor', length=3, width=0.5)
        ax.tick_params(axis='x', which='major', labelsize=10, length=6, width=0.8)


class ColorPalette:
    """Manage color palettes for plots"""
    
    PROFESSIONAL = [
        '#4C72B0',  # Muted blue
        '#DD8452',  # Muted orange
        '#55A868',  # Muted green
        '#C44E52',  # Muted red
        '#8172B3',  # Muted purple
        '#937860',  # Muted brown
    ]
    
    GRAYSCALE = [
        '#2F2F2F',  # Very dark gray
        '#4F4F4F',  # Dark gray
        '#707070',  # Medium gray
        '#909090',  # Medium-light gray
        '#B0B0B0',  # Light gray
        '#D0D0D0',  # Very light gray
    ]
    
    PATTERNS = ['', '///', '...', 'xxx', '\\\\\\', '|||', '---', '+++', 'ooo', '***']
    
    @staticmethod
    def generate_colors(base_colors: List[str], n_needed: int) -> List[str]:
        """Generate enough colors by creating variants if needed"""
        import matplotlib.colors as mcolors
        
        colors = list(base_colors)
        
        if n_needed <= len(base_colors):
            return colors[:n_needed]
        
        # Generate variants
        iteration = 1
        while len(colors) < n_needed:
            for base_color in base_colors:
                if len(colors) >= n_needed:
                    break
                
                rgb = mcolors.hex2color(base_color)
                
                # Alternate between lighter and darker variants
                if iteration % 2 == 1:
                    factor = 0.15 * iteration
                    variant = tuple(min(1.0, c + factor) for c in rgb)
                else:
                    factor = 0.15 * (iteration // 2)
                    variant = tuple(max(0.0, c - factor) for c in rgb)
                
                colors.append(mcolors.rgb2hex(variant))
            
            iteration += 1
        
        return colors[:n_needed]
    
    @classmethod
    def get_palette(cls, n_colors: int, grayscale: bool = False) -> List[str]:
        """Get color palette"""
        base = cls.GRAYSCALE if grayscale else cls.PROFESSIONAL
        return cls.generate_colors(base, n_colors)
    
    @classmethod
    def get_patterns(cls, n_patterns: int, no_patterns: bool = False) -> List[str]:
        """Get hatching patterns"""
        if no_patterns:
            return [''] * n_patterns
        return [cls.PATTERNS[i % len(cls.PATTERNS)] for i in range(n_patterns)]

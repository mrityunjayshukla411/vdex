"""
Example: Plotting GCStack CPI Stack data

This example demonstrates:
1. Loading GCStack data
2. Creating stacked bar plots with grouped configurations
3. Using hatch patterns to distinguish simulations
4. Saving plots to the plots/ directory
"""

from pathlib import Path
from vdex import Simulation, Scene, GCStackExtractor, plot_gcstack, GCStackPlotter


def main():
    print("="*70)
    print("GCStack CPI Stack Plotting Example")
    print("="*70)

    # Step 1: Extract GCStack data (or load existing)
    print("\n[Step 1] Extracting GCStack data...")

    extractor = (GCStackExtractor("gcstack_comparison")
        .add_simulation(Simulation(
            "/home2/mrityujay/raytracing/configs/coopreelet/vulkan-sim-root/coopreelet_simulations/simulations/cooperative_traversal_gcstack",
            "CoopRT"
        ))
        .add_simulation(Simulation(
            "/home2/mrityujay/raytracing/configs/coopreelet/vulkan-sim-root/coopreelet_simulations/simulations/cooperative_treelet_traversal_gcstack",
            "CoopTreeletTraversal"
        ))
        .add_simulation(Simulation(
            "/home2/mrityujay/raytracing/configs/coopreelet/vulkan-sim-root/coopreelet_simulations/simulations/cooperative_treelet_traversal_with_prefetching_gcstack/",
            "CoopTreeletTraversalPrefetching"
        ))
        .add_scene(Scene("bath"))
        .add_scene(Scene("bunny"))
        .add_scene(Scene("car"))
        .add_scene(Scene("chsnt"))
        .add_scene(Scene("crnvl"))
        .add_scene(Scene("fox"))
        .add_scene(Scene("frst"))
        .add_scene(Scene("lands"))
        .add_scene(Scene("park"))
        .add_scene(Scene("party"))
        .add_scene(Scene("ref"))
        .add_scene(Scene("robot"))
        .add_scene(Scene("ship"))
        .add_scene(Scene("spnza"))
        .add_scene(Scene("sprng"))
        .add_scene(Scene("wknd"))
        .extract()
        .save(also_csv=True)
    )

    df = extractor.dataframe()
    print(f"\n✓ Loaded {len(df)} results")

    # Step 2: Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    print(f"\n[Step 2] Creating plots in {plots_dir}/")

    # Step 3: Quick plot using convenience function
    print("\n[Step 3] Creating basic stacked plot...")
    plot_gcstack(
        df,
        plots_dir / "gcstack_basic.png",
        title="GCStack CPI Stack Breakdown - All Scenes"
    )

    # Step 4: Custom plot with hatches to distinguish simulations
    print("\n[Step 4] Creating grouped stacked plot with hatch patterns...")
    plotter = GCStackPlotter()
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_hatched.png",
        title="GCStack CPI Stack Breakdown (Normalized)",
        xlabel="Scene",
        ylabel="CPI Fraction",
        use_hatches=True,
        normalize=True,
        figsize=(14, 8)
    )

    # Step 5: Plot without normalization (absolute cycles)
    print("\n[Step 5] Creating plot with absolute cycles...")
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_absolute.png",
        title="GCStack CPI Stack - Absolute Cycles",
        xlabel="Scene",
        ylabel="Cycles",
        use_hatches=True,
        normalize=False,
        figsize=(14, 8)
    )

    # Step 6: Comparison plot (side-by-side)
    print("\n[Step 6] Creating comparison plot...")
    plotter.plot_comparison(
        df,
        baseline_sim="CoopRT",
        output_file=plots_dir / "gcstack_comparison.png",
        title="GCStack CPI Stack: Baseline vs Optimizations",
        figsize=(16, 7)
    )

    # Step 7: Single scene detailed view
    print("\n[Step 7] Creating single-scene plot...")
    sponza_df = df[df['scene'] == 'spnza']
    plotter.plot(
        sponza_df,
        output_file=plots_dir / "gcstack_sponza_only.png",
        title="GCStack CPI Stack - Sponza Scene Only",
        use_hatches=True,
        normalize=True,
        figsize=(10, 8)
    )

    print("\n" + "="*70)
    print("✓ All plots created successfully!")
    print(f"✓ Plots saved to: {plots_dir.absolute()}/")
    print("="*70)
    print("\nGenerated plots:")
    for plot_file in plots_dir.glob("gcstack_*.png"):
        print(f"  - {plot_file.name}")


def analysis_example():
    """Example showing data analysis before plotting"""
    print("\n" + "="*70)
    print("GCStack Analysis + Plotting Example")
    print("="*70)

    # Load data
    extractor = GCStackExtractor("gcstack_comparison")
    extractor.load()
    df = extractor.dataframe()

    # Calculate useful metrics
    print("\n[Analysis] Calculating derived metrics...")

    # Memory-related CPI
    df['memory_cpi'] = df['memstruct'] + df['memdata']
    df['memory_fraction'] = df['memory_cpi'] / df['total']

    # Compute-related CPI
    df['compute_cpi'] = df['comstruct'] + df['comdata']
    df['compute_fraction'] = df['compute_cpi'] / df['total']

    # Utilization (non-idle)
    df['utilization'] = (df['total'] - df['idle']) / df['total']

    print("\n--- Performance Summary ---")
    print(df[['simulation', 'scene', 'memory_fraction', 'compute_fraction', 'utilization']].to_string(index=False))

    # Plot with analysis
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plotter = GCStackPlotter()
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_analyzed.png",
        title=f"GCStack Analysis (Avg Memory: {df['memory_fraction'].mean():.2%})",
        use_hatches=True,
        normalize=True
    )

    print(f"\n✓ Analysis plot saved to: {plots_dir / 'gcstack_analyzed.png'}")


def custom_styling_example():
    """Example showing custom styling options"""
    print("\n" + "="*70)
    print("GCStack Custom Styling Example")
    print("="*70)

    # Load data
    extractor = GCStackExtractor("gcstack_comparison")
    extractor.load()
    df = extractor.dataframe()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plotter = GCStackPlotter()

    # Example 1: No hatches, legend on top
    print("\n[Custom 1] Plot without hatch patterns...")
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_no_hatches.png",
        title="GCStack CPI Stack (No Hatches)",
        use_hatches=False,
        legend_position='top',
        figsize=(14, 8)
    )

    # Example 2: Rotated labels
    print("\n[Custom 2] Plot with rotated labels...")
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_rotated.png",
        title="GCStack CPI Stack (Rotated Labels)",
        rotation=45,
        use_hatches=True,
        figsize=(14, 8)
    )

    # Example 3: Large figure for presentation
    print("\n[Custom 3] Large figure for presentations...")
    plotter.plot(
        df,
        output_file=plots_dir / "gcstack_presentation.png",
        title="GCStack CPI Stack Breakdown",
        use_hatches=True,
        normalize=True,
        figsize=(18, 10)
    )

    print(f"\n✓ Custom styled plots saved to: {plots_dir.absolute()}/")


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment to run additional examples:
    # analysis_example()
    # custom_styling_example()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)

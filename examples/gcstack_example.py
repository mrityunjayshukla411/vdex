"""
Example: Extracting and analyzing GCStack CPI Stack data

This example demonstrates:
1. Extracting GCStack CPI breakdown statistics
2. Saving as Parquet and CSV
3. Loading and viewing the data
4. Basic analysis of the CPI components
"""

from pathlib import Path
from vdex import Simulation, Scene, GCStackExtractor

# Example usage - adjust paths to match your simulation setup
def main():
    print("="*70)
    print("GCStack CPI Stack Extraction Example")
    print("="*70)

    # Step 1: Create extractor and add simulations/scenes
    print("\n[Step 1] Setting up extraction...")

    # NOTE: Replace these paths with your actual simulation paths
    extractor = (GCStackExtractor("gcstack_comparison")
        .add_simulation(Simulation("/home2/mrityujay/raytracing/configs/coopreelet/vulkan-sim-root/coopreelet_simulations/simulations/cooperative_traversal_gcstack", "CoopRT"))
        .add_simulation(Simulation("/home2/mrityujay/raytracing/configs/coopreelet/vulkan-sim-root/coopreelet_simulations/simulations/cooperative_treelet_traversal_gcstack", "CoopTreeletRT"))
        .add_scene(Scene("spnza"))
        .add_scene(Scene("bunny"))
        .add_scene(Scene("car"))
    )

    # Step 2: Extract data from log files
    print("\n[Step 2] Extracting GCStack data from logs...")
    extractor.extract()

    # Step 3: Save data (Parquet + CSV)
    print("\n[Step 3] Saving extracted data...")
    extractor.save(also_csv=True)

    # Step 4: Load and view the data
    print("\n[Step 4] Loading and viewing data...")
    df = extractor.dataframe()

    print("\n" + "="*70)
    print("Extracted GCStack CPI Stack Data:")
    print("="*70)
    print(df.to_string(index=False))

    # Step 5: Basic analysis
    print("\n" + "="*70)
    print("Basic Analysis:")
    print("="*70)

    # Calculate percentages for each component
    for component in ['base', 'memstruct', 'memdata', 'sync',
                     'comstruct', 'comdata', 'control', 'idle']:
        df[f'{component}_pct'] = (df[component] / df['total']) * 100

    # Show percentage breakdown
    print("\n--- CPI Component Percentages ---")
    pct_cols = ['simulation', 'scene'] + [f'{c}_pct' for c in
                ['base', 'memstruct', 'memdata', 'sync',
                 'comstruct', 'comdata', 'control', 'idle']]
    print(df[pct_cols].to_string(index=False))

    # Calculate useful metrics
    print("\n--- Key Metrics ---")
    df['memory_cpi'] = df['memstruct'] + df['memdata']
    df['compute_cpi'] = df['comstruct'] + df['comdata']
    df['effective_cpi'] = df['total'] - df['idle']

    metrics_cols = ['simulation', 'scene', 'memory_cpi', 'compute_cpi',
                    'effective_cpi', 'idle', 'total']
    print(df[metrics_cols].to_string(index=False))

    # Summary statistics per simulation
    print("\n--- Summary by Simulation ---")
    summary = df.groupby('simulation').agg({
        'total': 'mean',
        'base': 'mean',
        'memory_cpi': 'mean',
        'compute_cpi': 'mean',
        'idle': 'mean',
        'effective_cpi': 'mean'
    }).round(2)
    print(summary)

    print("\n" + "="*70)
    print("✓ Example completed successfully!")
    print(f"✓ Data saved to: data/gcstack_comparison/")
    print("="*70)


# Example: Loading previously extracted data
def load_example():
    """Example of loading previously extracted GCStack data"""
    print("\n" + "="*70)
    print("Loading Previously Extracted Data")
    print("="*70)

    # Create extractor with same name
    extractor = GCStackExtractor("gcstack_comparison")

    # Load the data
    extractor.load()
    df = extractor.dataframe()

    print("\nLoaded data:")
    print(df.head())

    return df


# Quick extraction function
def quick_extract(sim_paths, scenes, experiment_name="gcstack_quick"):
    """
    Quick extraction helper

    Args:
        sim_paths: List of (path, name) tuples
        scenes: List of scene names
        experiment_name: Name for the experiment

    Returns:
        DataFrame with extracted data

    Example:
        df = quick_extract(
            [("path/to/sim1", "Baseline"), ("path/to/sim2", "Optimized")],
            ["sponza", "bunny"]
        )
    """
    extractor = GCStackExtractor(experiment_name)

    # Add simulations
    for path, name in sim_paths:
        extractor.add_simulation(Simulation(path, name))

    # Add scenes
    for scene in scenes:
        extractor.add_scene(Scene(scene))

    # Extract and save
    extractor.extract().save(also_csv=True)

    return extractor.dataframe()


if __name__ == "__main__":
    # Run the main example
    # NOTE: This will fail if you don't have actual simulation data
    # Uncomment when you have real data:
    # main()

    # Or use the quick extract:
    # df = quick_extract(
    #     [("path/to/sim1", "Baseline"), ("path/to/sim2", "Optimized")],
    #     ["sponza", "bunny"]
    # )

    print(__doc__)
    print("\nTo use this example:")
    print("1. Update the simulation paths in main()")
    print("2. Uncomment the main() call at the bottom")
    print("3. Run: python examples/gcstack_example.py")

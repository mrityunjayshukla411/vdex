#!/usr/bin/env python3
"""
Complete Example: Extract, Analyze, Plot using vdex
"""

from vdex import Simulation, Scene, Extractor, Analysis, plot, quick_plot, load, VulkanSimParser

print("="*70)
print("EXAMPLE: Complete Workflow with vdex")
print("="*70)

# =============================================================================
# Step 1: Extract Data
# =============================================================================

print("\n[1/4] Extracting data...")

parser = VulkanSimParser()

data = (Extractor("complete_example")
    .add_simulation(Simulation("test_sim_data/cooperative_treelet_traversal_miss_latencies", "CTT"))
    .add_scene(Scene("spnza"))
    .add_scene(Scene("bunny"))
    .with_parser(parser)
    .extract()
    .save(also_csv=True))

print("✓ Data extracted and saved")

# =============================================================================
# Step 2: View Data
# =============================================================================

print("\n[2/4] Viewing data...")

df = data.dataframe()
print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData preview:")
print(df.head())

# =============================================================================
# Step 3: Analysis
# =============================================================================

print("\n[3/4] Analyzing data...")

analysis = Analysis(data.dataset())

# Available metrics
metrics = analysis.metrics()
print(f"\nAvailable metrics: {len(metrics)}")
print(f"First 5: {metrics[:5]}")

# Summary statistics
if 'gpu_tot_ipc' in metrics:
    summary = analysis.summarize('gpu_tot_ipc')
    print(f"\nIPC Summary:")
    print(summary)

# =============================================================================
# Step 4: Plotting
# =============================================================================

print("\n[4/4] Creating plots...")

# Quick plot
if 'gpu_tot_ipc' in metrics:
    quick_plot(df, 'gpu_tot_ipc', 'ipc_quick.png')
    print("✓ Quick plot: ipc_quick.png")

print("Panduaa")

# Fluent plot with customization
if 'gpu_tot_sim_cycle' in metrics:
    (plot(analysis)
        .metric('gpu_tot_sim_cycle')
        .save('cycles_detailed.png')
        .title('Execution Time')
        .labels(xlabel='Benchmark', ylabel='Cycles')
        .figsize(12, 6)
        .grouped_bar())
    print("✓ Detailed plot: cycles_detailed.png")

# Plot multiple metrics
print("\nGenerating plots for multiple metrics...")
for metric in ['gpu_tot_ipc', 'L1D_MPKI', 'L2_MPKI']:
    if metric in metrics:
        quick_plot(df, metric, f'{metric}.png')
        print(f"  ✓ {metric}.png")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"""
✓ Extracted {len(df)} results
✓ Created analysis with {len(metrics)} metrics  
✓ Generated plots

Files created:
  - data/complete_example/data.parquet
  - data/complete_example/data.csv
  - ipc_quick.png
  - cycles_detailed.png
  - gpu_tot_ipc.png
  - L1D_MPKI.png
  - L2_MPKI.png
""")

# =============================================================================
# Bonus: Load Later
# =============================================================================

print("\n" + "="*70)
print("BONUS: Loading Previously Extracted Data")
print("="*70)

# Load without re-extraction
loaded = load("complete_example")
print(f"✓ Loaded experiment with {len(loaded.dataframe())} results")

# Instant plotting from loaded data
quick_plot(loaded.dataframe(), 'gpu_tot_ipc', 'loaded_plot.png')
print("✓ Created plot from loaded data: loaded_plot.png")

print("\n" + "="*70)
print("All done! Check the .png files for your plots.")
print("="*70)

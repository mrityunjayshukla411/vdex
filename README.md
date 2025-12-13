# Vdex

Author:- Mrityunjay Shukla
**A professional, elegant Python framework for GPU simulation analysis with fluent interfaces, immutable domain models, and design patterns.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Type Hints](https://img.shields.io/badge/type%20hints-100%25-brightgreen.svg)](https://www.python.org/dev/peps/pep-0484/)
[![Design Patterns](https://img.shields.io/badge/design-patterns-purple.svg)](https://en.wikipedia.org/wiki/Software_design_pattern)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Why This Framework?

### Compare the Code

**Before (Manual scripting):**
```python
# 15+ lines of boilerplate
import pandas as pd
import os

results = []
for sim_path in ["sim1", "sim2"]:
    for scene in ["sponza", "bunny"]:
        log_file = f"{sim_path}/{scene}/bin/{scene}.log"
        data = parse_log(log_file)
        results.append(data)

df = pd.DataFrame(results)
df.to_csv("output.csv")

# Then separate script for plotting...
```

**After (Elegant framework):**
```python
from vdex import Simulation, Scene, Extractor, quick_plot

# Extract, save, and plot in 8 lines
data = (Extractor("experiment")
    .add_simulation(Simulation("sim1", "Baseline"))
    .add_simulation(Simulation("sim2", "RT-Cache"))
    .add_scene(Scene("sponza"))
    .add_scene(Scene("bunny"))
    .extract()
    .save())

quick_plot(data.dataframe(), 'gpu_tot_ipc', 'ipc.png')
```

**50-70% less code, fully type-safe, production-ready!**

---

## ✨ Key Features

### Design Excellence
- ✅ **Fluent Interface** - Method chaining for readable code
- ✅ **Immutable Domain Models** - Thread-safe, bug-resistant
- ✅ **Strategy Pattern** - Pluggable parsers and storage backends
- ✅ **Repository Pattern** - Clean data persistence abstraction
- ✅ **Factory Pattern** - Automatic parser selection
- ✅ **100% Type Hints** - Full IDE support and type checking

### Functionality
- ✅ **Multi-Simulation Extraction** - Compare multiple configurations
- ✅ **Automatic Metric Computation** - MPKI, hit rates, derived metrics
- ✅ **Efficient Storage** - Parquet format (5-10x smaller than CSV)
- ✅ **Smart Caching** - Avoid re-extraction
- ✅ **Powerful Analysis** - Compare, normalize, summarize with geomean
- ✅ **Elegant Plotting** - Fluent interface for beautiful visualizations
- ✅ **Extensible Architecture** - Add parsers, storage, plotters easily

---

## 🚀 Quick Start

### Installation

```bash
# Clone or extract the framework
cd vdex

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- pandas
- pyarrow  
- numpy
- matplotlib

### Your First Extraction

```python
from vdex import Simulation, Scene, Extractor

# Extract data with fluent interface
data = (Extractor("my_experiment")
    .add_simulation(Simulation("path/to/simulation", "Baseline"))
    .add_scene(Scene("sponza"))
    .add_scene(Scene("bunny"))
    .extract()
    .save())

# View data
df = data.dataframe()
print(df.head())
```

### Your First Plot

```python
from vdex import quick_plot

# One line to create a plot!
quick_plot(data.dataframe(), 'gpu_tot_ipc', 'ipc_comparison.png')
```

### Your First Analysis

```python
from vdex import Analysis

analysis = Analysis(data.dataset())

# Compare across simulations and scenes
comparison = analysis.compare('gpu_tot_ipc')

# Normalize to baseline
normalized = analysis.normalize('Baseline', 'gpu_tot_sim_cycle')

# Summary statistics (with geomean!)
summary = analysis.summarize('L1D_MPKI')
```

---

## 📖 Complete Examples

### Example 1: Basic Workflow

```python
from vdex import Simulation, Scene, Extractor, Analysis, plot

# Step 1: Extract
data = (Extractor("rt_cache_comparison")
    .add_simulation(Simulation("path/to/baseline", "Baseline"))
    .add_simulation(Simulation("path/to/rt_cache", "RT-Cache"))
    .add_scene(Scene("sponza"))
    .add_scene(Scene("bunny"))
    .add_scene(Scene("car"))
    .extract()
    .save(also_csv=True))

# Step 2: Analyze
analysis = Analysis(data.dataset())
comparison = analysis.compare('gpu_tot_ipc')
print(comparison)

# Step 3: Plot
(plot(analysis)
    .metric('gpu_tot_ipc')
    .title('IPC Comparison')
    .save('ipc.png')
    .grouped_bar())
```

### Example 2: Multiple Metrics

```python
from vdex import load, quick_plot

# Load previously extracted data (instant!)
data = load("rt_cache_comparison")

# Plot multiple metrics
metrics = ['gpu_tot_ipc', 'L1D_MPKI', 'L2_MPKI', 'gpu_tot_sim_cycle']

for metric in metrics:
    quick_plot(data.dataframe(), metric, f'{metric}.png')
    print(f"✓ {metric}.png")
```

### Example 3: Normalized Plots

```python
from vdex import plot

# Normalized to baseline
(plot(data.dataset())
    .metric('gpu_tot_sim_cycle')
    .baseline('Baseline')
    .title('Normalized Execution Time')
    .save('normalized_cycles.png')
    .normalized_bar())
```

### Example 4: Advanced Customization

```python
# Custom styling
(plot(data.dataset())
    .metric('L1D_MPKI')
    .title('L1D Cache Miss Rate')
    .labels(xlabel='Benchmark', ylabel='Misses per Kilo-Instruction')
    .figsize(15, 8)
    .rotation(45)
    .grayscale()
    .mean_type('geomean')
    .save('l1d_mpki.png')
    .grouped_bar())
```

---

## 🏗️ Architecture

### Design Philosophy

> **"Elegance is achieved when everything unnecessary is removed"**

The framework follows SOLID principles and professional design patterns:

```
┌─────────────────────────────────────────┐
│         vdex.py (Public API)            │  ← You import from here
├─────────────────────────────────────────┤
│  Core Domain (Immutable Models)         │  ← Business entities
│  - Simulation, Scene, Metric, Dataset   │
├─────────────────────────────────────────┤
│  Extractors (Orchestration)             │  ← Fluent workflows
│  - Extractor, IntervalExtractor         │
├─────────────────────────────────────────┤
│  Parsers (Strategy Pattern)             │  ← Pluggable parsing
│  - LogParser, VulkanSimParser           │
├─────────────────────────────────────────┤
│  Storage (Repository Pattern)           │  ← Persistence
│  - ParquetRepository, CSVRepository     │
├─────────────────────────────────────────┤
│  Plotters (Strategy Pattern)            │  ← Visualization
│  - GroupedBarPlotter, NormalizedPlotter │
├─────────────────────────────────────────┤
│  Analysis (Pure Functions)              │  ← Data analysis
│  - Compare, normalize, summarize        │
└─────────────────────────────────────────┘
```

### Key Design Patterns

**1. Fluent Interface**
```python
(Extractor("test")
    .add_simulation(sim)
    .add_scene(scene)
    .extract()
    .save())
```

**2. Strategy Pattern**
```python
# Different parsers
VulkanSimParser()
IntervalParser()
CustomParser()

# Different storage
ParquetRepository()
CSVRepository()
DatabaseRepository()
```

**3. Immutable Domain Models**
```python
@dataclass(frozen=True)
class Simulation:
    path: Path
    name: str

sim = Simulation("path", "test")
sim.name = "new"  # ✗ Error! Immutable
```

---

## 🔧 Extending the Framework

### Adding a Custom Parser

```python
# parsers/my_custom_parser.py
from parsers.log_parser import LogParser, ParserFactory

class MyCustomParser(LogParser):
    def can_parse(self, log_file):
        return 'MY_FORMAT' in log_file.read_text()
    
    def parse(self, log_file):
        # Your parsing logic
        return {'metric': Metric('metric', value)}

# Register and use
ParserFactory.register_parser(MyCustomParser)
```

### Adding a Custom Extractor

```python
# extractors/my_extractor.py
class MyExtractor:
    def __init__(self, name):
        self._name = name
    
    def add_simulation(self, sim):
        # ... 
        return self  # Fluent!
    
    def extract(self):
        # Your extraction logic
        return self
```

### Adding a Custom Plotter

```python
# plotters/my_plotter.py
from plotters.plotter import Plotter

class MyPlotter(Plotter):
    def plot(self, data, metric, output_file=None, **kwargs):
        fig, ax = self._setup_figure()
        # Your plotting logic
        self._apply_style(ax, 'X', 'Y', 'Title')
        self._save_or_show(fig, output_file)
```

**See `docs/IMPLEMENTATION_TUTORIAL.md` for detailed guides!**

---

## 📊 Data Format

### Extracted Metrics (Automatic)

- **Performance:** gpu_tot_sim_cycle, gpu_tot_ipc, gpu_tot_sim_insn
- **Cache:** L1D/L2 accesses, misses, MPKI, hit rates
- **RT-specific:** rt_avg_warp_latency, rt_avg_thread_latency  
- **Power:** kernel_avg_power, bwutil

### Storage Locations

- Parquet: `data/experiment_name/data.parquet` (fast, compressed)
- CSV: `data/experiment_name/data.csv` (human-readable)
- Metadata: `data/experiment_name/metadata.json`

---

## 📁 Project Structure

```
vulkan-sim-analysis/
├── vdex.py                    # Main API ⭐
├── core/                      # Domain models
├── extractors/                # Data extraction
├── parsers/                   # Log parsing
├── storage/                   # Persistence
├── plotters/                  # Visualization
├── examples/                  # Usage examples
├── tests/                     # Test suite
├── docs/                      # Documentation
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### "Log file not found"
Framework tries these locations:
1. `sim/scene/bin/scene.log`
2. `sim/scene/scene.log`
3. `sim/scene.log`

### "No suitable parser found"
Use explicit parser:
```python
from parsers.log_parser import VulkanSimParser
extractor.with_parser(VulkanSimParser())
```

### Plot displays instead of saving
**Wrong:** `.grouped_bar().save('out.png')`  
**Right:** `.save('out.png').grouped_bar()`

---

## 🎯 Quick Reference

### Import
```python
from vdex import Simulation, Scene, Extractor, Analysis, plot, quick_plot, load
```

### Extract → Save
```python
Extractor("name").add_simulation(sim).add_scene(scene).extract().save()
```

### Load → Plot
```python
quick_plot(load("name").dataframe(), 'metric', 'out.png')
```

### Fluent Plot
```python
plot(data).metric('ipc').save('out.png').grouped_bar()
```

---

**Ready to build? Start with:**

```bash
python examples/complete_example.py
```


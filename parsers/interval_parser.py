"""
Interval statistics parser - extracts time-series data
"""
from pathlib import Path
import pandas as pd
import re


class IntervalParser:
    """Parse interval statistics blocks"""
    
    def __init__(self):
        self.interval_header_re = re.compile(
            r"=== Interval Stats (\d+) @ (\d+) instructions ==="
        )
        self.interval_range_re = re.compile(
            r"Interval:\s+(\d+)-(\d+)\s+instructions\s+\((\d+)\s+cycles\)"
        )
        
        # Metric patterns
        self.patterns = {
            "occupancy_percent": re.compile(r"Occupancy:\s+([\d.]+)%"),
            "ipc": re.compile(r"IPC:\s+([\d.]+)"),
            "l1d_accesses": re.compile(r"L1D Accesses:\s+(\d+)"),
            "l1d_misses": re.compile(r"L1D Misses:\s+(\d+)"),
            "l1d_hit_rate": re.compile(r"L1D Hit Rate:\s+([\d.]+)%"),
            "l2_accesses": re.compile(r"L2 Accesses:\s+(\d+)"),
            "l2_misses": re.compile(r"L2 Misses:\s+(\d+)"),
            "l2_hit_rate": re.compile(r"L2 Hit Rate:\s+([\d.]+)%"),
            "dram_stalls": re.compile(r"DRAM Stalls:\s+(\d+)"),
        }
    
    def parse(self, log_file: Path) -> pd.DataFrame:
        """Parse all interval blocks"""
        results = []
        curr = {}
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    
                    # Start of block
                    m = self.interval_header_re.search(line)
                    if m:
                        if curr:
                            results.append(curr)
                        
                        curr = {
                            "interval_id": int(m.group(1)),
                            "instruction_marker": int(m.group(2)),
                            **{k: None for k in self.patterns}
                        }
                        continue
                    
                    # Parse interval range
                    m = self.interval_range_re.search(line)
                    if m and curr:
                        curr["start_instr"] = int(m.group(1))
                        curr["end_instr"] = int(m.group(2))
                        curr["cycles"] = int(m.group(3))
                        continue
                    
                    # Parse metrics
                    for key, regex in self.patterns.items():
                        m = regex.search(line)
                        if m and curr:
                            val = m.group(1)
                            curr[key] = float(val) if '.' in val or '%' in key else int(val)
                            break
            
            if curr:
                results.append(curr)
        
        except Exception as e:
            print(f"Warning: Error parsing intervals from {log_file}: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
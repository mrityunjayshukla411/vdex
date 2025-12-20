"""
GCStack CPI Stack statistics parser - extracts performance breakdown data
"""
from pathlib import Path
import pandas as pd
import re


class GCStackParser:
    """
    Parse GCStack CPI Stack Information

    Extracts performance breakdown metrics:
    - Base, MemStruct, MemData, Sync, ComStruct, ComData, Control, Idle, Total
    """

    def __init__(self):
        # Regex pattern to match GCStack metrics
        # Format: GCStack_MetricName:value
        self.gcstack_re = re.compile(r"GCStack_([A-Za-z]+):([\d.]+)")
        self.header_re = re.compile(r"={9,}\s*GCStack CPI Stack Information\s*={9,}")

    def parse(self, log_file: Path) -> pd.DataFrame:
        """
        Parse GCStack CPI Stack statistics from log file

        Returns:
            DataFrame with columns: base, memstruct, memdata, sync, comstruct,
                                   comdata, control, idle, total
        """

        gcstack_data = {}
        in_gcstack_block = False

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Detect start of GCStack block
                    if self.header_re.search(line):
                        in_gcstack_block = True
                        gcstack_data = {}
                        continue

                    # Parse GCStack metrics
                    if in_gcstack_block:
                        match = self.gcstack_re.search(line)
                        if match:
                            metric_name = match.group(1).lower()
                            metric_value = float(match.group(2))
                            gcstack_data[metric_name] = metric_value

                        # End of block (empty line after we have data, or next section starting with ===)
                        if (line == '' and gcstack_data) or (line.startswith('===') and 'GCStack CPI Stack Information' not in line):
                            in_gcstack_block = False
                            # If we have collected data, break (assuming one GCStack block per file)
                            if gcstack_data:
                                break

            # Convert to DataFrame
            if gcstack_data:
                # Ensure all expected columns exist
                expected_cols = ['base', 'memstruct', 'memdata', 'sync',
                               'comstruct', 'comdata', 'control', 'idle', 'total']

                for col in expected_cols:
                    if col not in gcstack_data:
                        gcstack_data[col] = 0.0

                return pd.DataFrame([gcstack_data])
            else:
                print(f"Warning: No GCStack data found in {log_file}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Warning: Error parsing GCStack data from {log_file}: {e}")
            return pd.DataFrame()

    def can_parse(self, log_file: Path) -> bool:
        """
        Check if this parser can handle the log file

        Args:
            log_file: Path to log file

        Returns:
            True if file contains GCStack CPI Stack Information
        """
        try:
            with open(log_file, 'r') as f:
                content = f.read(10000)  # Read first 10KB
                return 'GCStack CPI Stack Information' in content
        except:
            return False

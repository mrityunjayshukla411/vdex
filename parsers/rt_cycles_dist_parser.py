"""
RT Cycles Distribution parser - extracts RT unit cycle distribution data
"""
from pathlib import Path
import pandas as pd
import re


class RTCyclesDistParser:
    """
    Parse RT Cycles Distribution metric

    Extracts cycle count distribution across multiple RT units.
    Format: rt_cycles_dist:   83391   113680  119756  109123  116255  113178  119585  120443

    The number of RT units is dynamic and detected from the data.
    """

    def __init__(self):
        # Regex pattern to match rt_cycles_dist metric
        # Format: rt_cycles_dist: followed by space-separated numbers
        self.rt_cycles_dist_re = re.compile(r"rt_cycles_dist:\s+([\d\s]+)")

    def parse(self, log_file: Path) -> pd.DataFrame:
        """
        Parse RT Cycles Distribution from log file

        Returns:
            DataFrame with columns: rt_unit_0, rt_unit_1, ..., rt_unit_N
            where N is the number of RT units detected (variable)
        """
        rt_cycles_data = {}

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Parse rt_cycles_dist metric
                    match = self.rt_cycles_dist_re.search(line)
                    if match:
                        # Extract all cycle values
                        cycles_str = match.group(1).strip()
                        cycle_values = cycles_str.split()

                        # Create columns for each RT unit
                        for idx, value in enumerate(cycle_values):
                            rt_cycles_data[f'rt_unit_{idx}'] = int(value)

                        # Assuming one rt_cycles_dist line per file, break after finding it
                        break

        except FileNotFoundError:
            print(f"Log file {log_file} not found.")
            return pd.DataFrame()

        # Convert collected data to DataFrame
        if rt_cycles_data:
            df = pd.DataFrame([rt_cycles_data])
            return df
        else:
            return pd.DataFrame()

    def can_parse(self, log_file: Path) -> bool:
        """
        Check if the log file contains rt_cycles_dist metric

        Returns:
            True if rt_cycles_dist is found, False otherwise
        """
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if self.rt_cycles_dist_re.search(line):
                        return True
        except FileNotFoundError:
            print(f"Log file {log_file} not found.")
            return False

        return False

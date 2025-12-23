"""
RT GCStack CPI Stack statistics parser - extracts performance breakdown data
"""
from pathlib import Path
import pandas as pd
import re

class RTGCStackParser:
    """
    Parse RT GCStack CPI Stack Information

    Extracts performance breakdown metrics:
    - Base, TravData, TravStruct, IsectDelay, IsectStruct, MemStruct, Coherence, Idle, EmptyRaySlot, IdleRTUnit, Total
    """

    def __init__(self):
        # Regex pattern to match RT GCStack metrics
        # Format: RT_GCStack_MetricName:value
        self.rt_gcstack_re = re.compile(r"^(RTGCStack_[A-Za-z0-9]+):\s+([0-9]+\.[0-9]+)$")
        self.header_re = re.compile(r"RT Unit GCStack CPI Breakdown:")

    def parse(self, log_file: Path) -> pd.DataFrame:
            """
            Parse RT GCStack CPI Stack statistics from log file

            Returns:
                DataFrame with columns: base, travdata, travstruct, isectdelay,
                                       isectstruct, memstruct, coherence, idle,
                                       emptyrayslot, idlertunit, total
            """

            rt_gcstack_data = {}
            in_rt_gcstack_block = False

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()

                        # Detect start of RT GCStack block
                        if self.header_re.search(line):
                            in_rt_gcstack_block = True
                            rt_gcstack_data = {}
                            continue

                        # Parse RT GCStack metrics
                        if in_rt_gcstack_block:
                            match = self.rt_gcstack_re.search(line)
                            if match:
                                metric_name = match.group(1).lower()
                                metric_value = float(match.group(2))
                                rt_gcstack_data[metric_name] = metric_value

                            # End of block (empty line after we have data, or next section starting with ===)
                            if (line == '' and rt_gcstack_data):
                                in_rt_gcstack_block = False
                                # If we have collected data, break (assuming one RT GCStack block per file)
                                if rt_gcstack_data:
                                    break

            except FileNotFoundError:
                print(f"Log file {log_file} not found.")
                return pd.DataFrame()

            # Convert collected data to DataFrame
            if rt_gcstack_data:
                df = pd.DataFrame([rt_gcstack_data])
                return df
            else:
                return pd.DataFrame()
    def can_parse(self, log_file: Path) -> bool:
        """
        Check if the log file contains RT GCStack CPI Stack Information

        Returns:
            True if RT GCStack section is found, False otherwise
        """
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if self.header_re.search(line):
                        return True
        except FileNotFoundError:
            print(f"Log file {log_file} not found.")
            return False

        return False
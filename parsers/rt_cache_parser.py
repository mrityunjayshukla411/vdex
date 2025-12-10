"""
RT-Cache statistics parser - extracts per-SM cache data
"""
from pathlib import Path    
from typing import List, Dict
import pandas as pd
import re

from core.domain import Metric

class RTCacheParser:
    """
    Parse RTCache statistics per SM
    """
    def __init__(self):
        # Regex patterns
        self.sm_header_re = re.compile(r"RT-Cache Statistics \(SM (\d+)\)")
        self.kv_re = re.compile(r"([A-Za-z ]+):\s+([0-9]+)")
        self.kv_rate_re = re.compile(r"([A-Za-z ]+):\s+([0-9]+)\s+\(([0-9.]+)%\)")

    def parse(self, log_file: Path) -> pd.DataFrame:
        """
        Parse RT-cache statistics from log file
        
        Returns:
            Dataframe with columns: sm_id, total_accesses, hits, misses. internal_*, leaf_*
        """

        sm_data = []
        current_sm = None
        section = None

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Identify SM block
                    sm_match = self.sm_header_re.search(line)
                    if sm_match:
                        # Save previous SM block
                        if current_sm:
                            sm_data.append(current_sm)
                        
                        current_sm = {
                            'sm_id': int(sm_match.group(1)),
                            'total_accesses': 0,
                            'hits': 0,
                            'misses': 0,
                            'hit_rate': 0.0,
                            'internal_node_accesses': 0,
                            'internal_node_hits': 0,
                            'internal_node_misses': 0,
                            'leaf_node_accesses': 0,
                            'leaf_node_hits': 0,
                            'leaf_node_misses': 0
                        }
                        section = 'header'
                        continue
                    
                    if current_sm is None:
                        continue
                    
                    # Identify section
                    if 'Internal Node Statistics' in line:
                        section = 'internal'
                        continue
                    elif 'Leaf Node Statistics' in line:
                        section = 'leaf'
                        continue
                    
                    # Parse stats based on section
                    if section == 'header':
                        self._parse_header_stats(line, current_sm)
                    elif section == 'internal':
                        self._parse_internal_stats(line, current_sm)
                    elif section == 'leaf':
                        self._parse_leaf_stats(line, current_sm)
            
            # Add last SM
            if current_sm:
                sm_data.append(current_sm)
        
        except Exception as e:
            print(f"Warning: Error parsing RT-cache stats from {log_file}: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(sm_data)
    def _parse_header_stats(self, line: str, sm_dict: dict):
        """Parse overall RT-cache stats"""
        # Try with rate pattern first
        m = self.kv_rate_re.search(line)
        if m:
            key = m.group(1).strip().lower().replace(' ', '_')
            count = int(m.group(2))
            rate = float(m.group(3))
            
            if 'hits' in key:
                sm_dict['hits'] = count
                sm_dict['hit_rate'] = rate
            elif 'misses' in key:
                sm_dict['misses'] = count
            return
        
        # Try simple pattern
        m = self.kv_re.search(line)
        if m:
            key = m.group(1).strip().lower().replace(' ', '_')
            count = int(m.group(2))
            
            if 'total_accesses' in key:
                sm_dict['total_accesses'] = count
    
    def _parse_internal_stats(self, line: str, sm_dict: dict):
        """Parse internal node stats"""
        m = self.kv_rate_re.search(line)
        if m:
            key = m.group(1).strip().lower()
            count = int(m.group(2))
            
            if 'hits' in key:
                sm_dict['internal_node_hits'] = count
            elif 'misses' in key:
                sm_dict['internal_node_misses'] = count
            return
        
        m = self.kv_re.search(line)
        if m:
            key = m.group(1).strip().lower()
            count = int(m.group(2))
            
            if 'accesses' in key:
                sm_dict['internal_node_accesses'] = count
    
    def _parse_leaf_stats(self, line: str, sm_dict: dict):
        """Parse leaf node stats"""
        m = self.kv_rate_re.search(line)
        if m:
            key = m.group(1).strip().lower()
            count = int(m.group(2))
            
            if 'hits' in key:
                sm_dict['leaf_node_hits'] = count
            elif 'misses' in key:
                sm_dict['leaf_node_misses'] = count
            return
        
        m = self.kv_re.search(line)
        if m:
            key = m.group(1).strip().lower()
            count = int(m.group(2))
            
            if 'accesses' in key:
                sm_dict['leaf_node_accesses'] = count
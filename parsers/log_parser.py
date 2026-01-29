"""
Parser interface and implementations - Strategy pattern for different log formats
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
import re
import warnings

from core.domain import Metric


class LogParser(ABC):
    """Abstract parser interface - implement for different log formats"""
    
    @abstractmethod
    def parse(self, log_file: Path) -> Dict[str, Metric]:
        """
        Parse a log file and return metrics
        
        Args:
            log_file: Path to log file
            
        Returns:
            Dictionary mapping metric names to Metric objects
        """
        pass
    
    @abstractmethod
    def can_parse(self, log_file: Path) -> bool:
        """Check if this parser can handle the given log file"""
        pass


class VulkanSimParser(LogParser):
    """Parser for standard vulkan-sim log format"""
    
    def __init__(self):
        # Compile regex patterns once
        self._patterns = {
            'gpu_tot_sim_cycle': re.compile(r'gpu_tot_sim_cycle\s*=\s*([\d.]+)'),
            'rt_mem_accesses_queue_avg': re.compile(r'rt_mem_accesses_queue_avg\s*=\s*([\d.]+)'),
            'rt_mem_accesses_queue_max': re.compile(r'rt_mem_accesses_queue_max\s*=\s*([\d.]+)'),
            'gpu_tot_ipc': re.compile(r'gpu_tot_ipc\s*=\s*([\d.]+)'),
            'gpu_stall_dramfull': re.compile(r'gpu_stall_dramfull\s*=\s*([\d.]+)'),
            'gpu_tot_occupancy': re.compile(r'gpu_tot_occupancy\s*=\s*([\d.]+)'),
            'L1D_total_cache_accesses': re.compile(r'L1D_total_cache_accesses\s*=\s*([\d.]+)'),
            'L1D_total_cache_misses': re.compile(r'L1D_total_cache_misses\s*=\s*([\d.]+)'),
            'L2_total_cache_accesses': re.compile(r'L2_total_cache_accesses\s*=\s*([\d.]+)'),
            'L2_total_cache_misses': re.compile(r'L2_total_cache_misses\s*=\s*([\d.]+)'),
            'gpu_tot_sim_insn': re.compile(r'gpu_tot_sim_insn\s*=\s*([\d.]+)'),
            'kernel_avg_power': re.compile(r'kernel_avg_power\s*=\s*([\d.]+)'),
            'bwutil': re.compile(r'bwutil\s*=\s*([\d.]+)'),
            'max_icnt2mem_latency': re.compile(r'max_icnt2mem_latency\s*=\s*([\d.]+)'),
            'avg_icnt2mem_latency': re.compile(r'avg_icnt2mem_latency\s*=\s*([\d.]+)'),
            # L1D Miss Latency metrics
            'L1D_miss_latency_total_accesses': re.compile(r'Global L1D Miss Latency:\s*\n\s*Total Accesses:\s*([\d.]+)'),
            'L1D_miss_latency_avg': re.compile(r'Global L1D Miss Latency:.*?Average Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
            'L1D_miss_latency_max': re.compile(r'Global L1D Miss Latency:.*?Max Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
            'L1D_miss_latency_min': re.compile(r'Global L1D Miss Latency:.*?Min Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
            # L2 Miss Latency metrics
            'L2_miss_latency_total_accesses': re.compile(r'Global L2 Miss Latency:\s*\n\s*Total Accesses:\s*([\d.]+)'),
            'L2_miss_latency_avg': re.compile(r'Global L2 Miss Latency:.*?Average Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
            'L2_miss_latency_max': re.compile(r'Global L2 Miss Latency:.*?Max Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
            'L2_miss_latency_min': re.compile(r'Global L2 Miss Latency:.*?Min Latency:\s*([\d.]+)\s*cycles', re.DOTALL),
        }
        
        # Units for metrics
        self._units = {
            'gpu_tot_sim_cycle': 'cycles',
            'gpu_tot_ipc': 'IPC',
            'gpu_tot_occupancy': '%',
            'kernel_avg_power': 'W',
            'bwutil': '%',
            'L1D_miss_latency_total_accesses': 'accesses',
            'L1D_miss_latency_avg': 'cycles',
            'L1D_miss_latency_max': 'cycles',
            'L1D_miss_latency_min': 'cycles',
            'L2_miss_latency_total_accesses': 'accesses',
            'L2_miss_latency_avg': 'cycles',
            'L2_miss_latency_max': 'cycles',
            'L2_miss_latency_min': 'cycles',
        }
    
    def can_parse(self, log_file: Path) -> bool:
        """Check if file looks like vulkan-sim output"""
        if not log_file.exists():
            return False
        
        try:
            with open(log_file, 'r') as f:
                first_lines = ''.join([f.readline() for _ in range(10)])
                return 'gpu_tot_sim_cycle' in first_lines or 'gpgpu-sim' in first_lines.lower()
        except Exception:
            return False
    
    def parse(self, log_file: Path) -> Dict[str, Metric]:
        """Parse vulkan-sim log file"""
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        metrics = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract metrics using patterns
            for name, pattern in self._patterns.items():
                match = pattern.search(content)
                if match:
                    value = float(match.group(1))
                    unit = self._units.get(name)
                    metrics[name] = Metric(name, value, unit)
            
            # Compute derived metrics
            metrics.update(self._compute_derived(metrics))
            
        except Exception as e:
            warnings.warn(f"Error parsing {log_file}: {e}")
        
        return metrics
    
    def _compute_derived(self, metrics: Dict[str, Metric]) -> Dict[str, Metric]:
        """Compute derived metrics"""
        derived = {}
        
        # L1D MPKI
        if 'L1D_total_cache_misses' in metrics and 'gpu_tot_sim_insn' in metrics:
            insn = metrics['gpu_tot_sim_insn'].value
            if insn > 0:
                mpki = (metrics['L1D_total_cache_misses'].value / insn) * 1000
                derived['L1D_MPKI'] = Metric('L1D_MPKI', mpki, 'MPKI')
        
        # L2 MPKI
        if 'L2_total_cache_misses' in metrics and 'gpu_tot_sim_insn' in metrics:
            insn = metrics['gpu_tot_sim_insn'].value
            if insn > 0:
                mpki = (metrics['L2_total_cache_misses'].value / insn) * 1000
                derived['L2_MPKI'] = Metric('L2_MPKI', mpki, 'MPKI')
        
        # L1D miss rate
        if 'L1D_total_cache_accesses' in metrics and metrics['L1D_total_cache_accesses'].value > 0:
            accesses = metrics['L1D_total_cache_accesses'].value
            misses = metrics['L1D_total_cache_misses'].value
            miss_rate = (misses / accesses) * 100
            hit_rate = 100 - miss_rate
            derived['L1D_miss_rate'] = Metric('L1D_miss_rate', miss_rate, '%')
            derived['L1D_hit_rate'] = Metric('L1D_hit_rate', hit_rate, '%')
        
        # L2 miss rate
        if 'L2_total_cache_accesses' in metrics and metrics['L2_total_cache_accesses'].value > 0:
            accesses = metrics['L2_total_cache_accesses'].value
            misses = metrics['L2_total_cache_misses'].value
            miss_rate = (misses / accesses) * 100
            hit_rate = 100 - miss_rate
            derived['L2_miss_rate'] = Metric('L2_miss_rate', miss_rate, '%')
            derived['L2_hit_rate'] = Metric('L2_hit_rate', hit_rate, '%')
        
        return derived


class ConfigurableParser(LogParser):
    """Parser that reads patterns from YAML config - for extensibility"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize parser with metric configuration
        
        Args:
            config_path: Path to metrics.yaml config file
        """
        self.config_path = config_path
        self._patterns = {}
        
        if config_path and config_path.exists():
            self._load_config()
    
    def can_parse(self, log_file: Path) -> bool:
        """Delegate to VulkanSimParser for now"""
        return VulkanSimParser().can_parse(log_file)
    
    def parse(self, log_file: Path) -> Dict[str, Metric]:
        """Parse using configured patterns"""
        # For now, delegate to standard parser
        # Can be extended to read from config
        return VulkanSimParser().parse(log_file)
    
    def _load_config(self):
        """Load configuration from YAML"""
        import yaml
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Compile patterns from config
        if 'global_metrics' in config:
            for name, cfg in config['global_metrics'].items():
                if 'pattern' in cfg:
                    self._patterns[name] = re.compile(cfg['pattern'])


class ParserFactory:
    """Factory for creating appropriate parser"""
    
    _parsers = [VulkanSimParser]
    
    @classmethod
    def create(cls, log_file: Path) -> LogParser:
        """
        Create appropriate parser for the log file
        
        Args:
            log_file: Path to log file
            
        Returns:
            Parser that can handle this file
            
        Raises:
            ValueError: If no suitable parser found
        """
        for parser_class in cls._parsers:
            parser = parser_class()
            if parser.can_parse(log_file):
                return parser
        
        raise ValueError(f"No suitable parser found for {log_file}")
    
    @classmethod
    def register_parser(cls, parser_class: type):
        """Register a custom parser"""
        cls._parsers.insert(0, parser_class)

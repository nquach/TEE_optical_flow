"""
Configuration objects for optical flow analysis.

This module provides dataclasses to group related parameters, reducing
function parameter counts and centralizing configuration management.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List


@dataclass
class CardiacCycleConfig:
    """Configuration for cardiac cycle detection methods."""
    smooth_fraction: float = 0.2
    pad_len: int = 20
    sys_thres: float = 0.9
    dia_thres: float = 0.5
    rr_sys_ratio: float = 0.333
    sys_extension: int = 2
    t_peak_thres: float = 0.5
    t_min_dist: int = 20
    rr_search_range: List[float] = field(default_factory=lambda: [0.2, 0.75])
    low_peak_thres: float = 0.9
    low_min_dist: int = 50
    high_peak_thres: float = 0.9
    high_min_dist: int = 50
    sys_upstroke_multiplier: int = 2
    sys_upstroke_offset: int = 5


@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting."""
    save_dir: Optional[str] = None
    show_plot: bool = False
    show_img: bool = False
    save_cc_plot: bool = False
    nbins: int = 1000
    invert_rad_yaxis: bool = False
    invert_long_yaxis: bool = False
    fps: int = 30
    colormap_mag: str = 'hot'
    colormap_ang: str = 'viridis'
    colormap_rad: str = 'bwr'
    colormap_long: str = 'BrBG'


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    recalculate: bool = True
    verbose: bool = False
    sampling_rate: Optional[int] = None
    ecg_sampling_rate: int = 500
    art_sampling_rate: int = 125
    cvp_sampling_rate: int = 125
    pap_sampling_rate: int = 125


@dataclass
class PeakDetectionConfig:
    """Configuration for peak detection algorithms."""
    peak_thres: float = 0.2
    min_dist: int = 5
    pick_peak_by_subset: bool = True
    show_all_peaks: bool = False
    smooth_fraction: float = 0.3
    pad_len: int = 20


@dataclass
class AnalysisConfig:
    """Configuration for data analysis operations."""
    percentile: int = 99
    perc_lo: int = 1
    perc_hi: int = 99
    av_filter_flag: bool = True
    av_savgol_window: int = 10
    av_savgol_poly: int = 4
    print_report: bool = False
    return_value: bool = True


@dataclass
class CardiacCycleMethodConfig:
    """Configuration for cardiac cycle detection method selection."""
    method: Literal['angle', 'area', 'ecg', 'ecg_lazy', 'metadata', 'arterial'] = 'angle'
    label: str = 'rv_inner'
    true_sysdia_mode: Literal['radial', 'longitudinal'] = 'radial'
    waveform_data: Optional[object] = None
    show_sysdia: bool = False


# Factory functions for common config presets

def default_cardiac_cycle_config() -> CardiacCycleConfig:
    """Create default cardiac cycle detection configuration."""
    return CardiacCycleConfig()


def default_visualization_config() -> VisualizationConfig:
    """Create default visualization configuration."""
    return VisualizationConfig()


def default_processing_config() -> ProcessingConfig:
    """Create default processing configuration."""
    return ProcessingConfig()


def default_peak_detection_config() -> PeakDetectionConfig:
    """Create default peak detection configuration."""
    return PeakDetectionConfig()


def default_analysis_config() -> AnalysisConfig:
    """Create default analysis configuration."""
    return AnalysisConfig()


def ecg_gated_config() -> CardiacCycleConfig:
    """Create configuration optimized for ECG-gated analysis."""
    config = CardiacCycleConfig()
    config.smooth_fraction = 0.2
    config.pad_len = 20
    config.rr_sys_ratio = 0.333
    return config


def arterial_gated_config() -> CardiacCycleConfig:
    """Create configuration optimized for arterial pressure-gated analysis."""
    config = CardiacCycleConfig()
    config.smooth_fraction = 0.2
    config.pad_len = 20
    config.low_peak_thres = 0.9
    config.low_min_dist = 50
    config.high_peak_thres = 0.9
    config.high_min_dist = 50
    return config


def angle_detection_config() -> CardiacCycleConfig:
    """Create configuration for angle-based cardiac cycle detection."""
    config = CardiacCycleConfig()
    config.smooth_fraction = 0.2
    config.pad_len = 20
    return config


def area_detection_config() -> CardiacCycleConfig:
    """Create configuration for area-based cardiac cycle detection."""
    config = CardiacCycleConfig()
    config.smooth_fraction = 0.3
    config.pad_len = 20
    config.sys_thres = 0.9
    config.dia_thres = 0.5
    return config


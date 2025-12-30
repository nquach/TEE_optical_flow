"""
High-level API for optical flow analysis.

This module provides clean, high-level functions for common operations,
hiding implementation details and providing a simple interface.
"""

from typing import Optional, List
from optical_flow.optical_flow_dataset import OpticalFlowDataset
from optical_flow.config import (
    CardiacCycleConfig, VisualizationConfig, ProcessingConfig,
    AnalysisConfig, PeakDetectionConfig
)
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.visualization import VisualizationManager
from optical_flow.analysis import calculate_3dhist_radlong, calculate_3dhist
from optical_flow.peak_detection import calculate_radlong_peaks


def analyze_optical_flow(dataset: OpticalFlowDataset, param: str, label: str,
                        cc_config: Optional[CardiacCycleConfig] = None,
                        proc_config: Optional[ProcessingConfig] = None,
                        analysis_config: Optional[AnalysisConfig] = None) -> dict:
    """
    Main entry point for optical flow analysis.
    
    Args:
        dataset: OpticalFlowDataset instance
        param: Parameter to analyze ('velocity', 'acceleration', 'PWR')
        label: Label/mask to use
        cc_config: Cardiac cycle configuration
        proc_config: Processing configuration
        analysis_config: Analysis configuration
    
    Returns:
        Dictionary with analysis results
    """
    if not dataset._validate_param(param):
        raise ValueError(f'Invalid parameter: {param}. Must be one of {dataset.accepted_params}')
    
    if not dataset._validate_label(label):
        raise ValueError(f'Invalid label: {label}. Must be one of {dataset.accepted_labels}')
    
    # Use default configs if not provided
    if cc_config is None:
        cc_config = CardiacCycleConfig()
    if proc_config is None:
        proc_config = ProcessingConfig()
    if analysis_config is None:
        analysis_config = AnalysisConfig()
    
    # Perform analysis
    masked_arr = dataset.get_masked_arr(param, label)
    mag, ang, mag_edges, ang_edges, perc_hi = calculate_3dhist(
        masked_arr, dataset.nframes, nbins=analysis_config.av_savgol_window,
        percentile=99
    )
    
    return {
        'magnitude': mag,
        'angle': ang,
        'magnitude_edges': mag_edges,
        'angle_edges': ang_edges,
        'percentile_high': perc_hi
    }


def plot_results(dataset: OpticalFlowDataset, param: str, label: str,
                save_path: str, vis_config: Optional[VisualizationConfig] = None,
                proc_config: Optional[ProcessingConfig] = None,
                analysis_config: Optional[AnalysisConfig] = None):
    """
    Visualization entry point for plotting results.
    
    Args:
        dataset: OpticalFlowDataset instance
        param: Parameter to plot
        label: Label/mask to use
        save_path: Path to save plot
        vis_config: Visualization configuration
        proc_config: Processing configuration
        analysis_config: Analysis configuration
    """
    if vis_config is None:
        vis_config = VisualizationConfig()
    if proc_config is None:
        proc_config = ProcessingConfig()
    if analysis_config is None:
        analysis_config = AnalysisConfig()
    
    vis_manager = VisualizationManager(vis_config, proc_config, analysis_config)
    
    # Get analysis results
    results = analyze_optical_flow(dataset, param, label, proc_config=proc_config,
                                  analysis_config=analysis_config)
    
    # Create plot
    frame_times = dataset.nframes * (1000 / dataset.frame_rate)
    vis_manager.plot_heatmap(
        results['magnitude'], results['angle'],
        results['magnitude_edges'], results['angle_edges'],
        frame_times, param, dataset._param_unit(param),
        dataset.filename, save_path
    )


def batch_process(folder: str, save_dir: str, param_list: List[str],
                 label_list: List[str], process_func: callable,
                 nchunks: int = 10, chunk_index: int = 0,
                 recalculate: bool = False, verbose: bool = True):
    """
    Batch processing entry point.
    
    Args:
        folder: Folder containing HDF5 files
        save_dir: Directory to save results
        param_list: List of parameters to process
        label_list: List of labels to process
        process_func: Function to process each file
        nchunks: Number of chunks
        chunk_index: Chunk index to process
        recalculate: Whether to recalculate
        verbose: Whether to print progress
    """
    from optical_flow.batch_processing import analyze_hdf5_folder
    
    analyze_hdf5_folder(
        folder, save_dir, param_list, label_list, process_func,
        nchunks=nchunks, chunk_index=chunk_index,
        recalculate=recalculate, verbose=verbose
    )


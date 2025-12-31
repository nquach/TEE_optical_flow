# TEE Optical Flow Analysis

A comprehensive Python package for calculating and analyzing optical flow in transesophageal echocardiography (TEE) images, with automatic cardiac cycle detection, radial/longitudinal component analysis, and peak detection. This project was adapted from the finetune-SAM github https://github.com/mazurowski-lab/finetune-SAM

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Examples](#examples)
- [Module Reference](#module-reference)
- [Contributing](#contributing)

## Overview

This codebase provides tools for:

1. **Optical Flow Calculation**: Compute optical flow from DICOM video sequences using DualTVL1 or DeepFlow algorithms
2. **Cardiac Cycle Detection**: Automatically detect systole and diastole using multiple methods:
   - Angle-based detection (optical flow angle analysis)
   - Area-based detection (mask area changes)
   - ECG-gated detection (R-wave and T-wave detection)
   - Arterial pressure-gated detection
   - DICOM metadata-based detection (R-wave times)
3. **Component Analysis**: Decompose optical flow into radial and longitudinal components
4. **Peak Detection**: Identify systolic and diastolic peaks (e', l', a') for both radial/longitudinal and single component data
5. **Visualization**: Generate heatmaps, plots, and video overlays with peak markers, systole/diastole shading, and statistics reporting

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Multiple Detection Methods**: 6 different cardiac cycle detection strategies
- **Configurable**: Extensive configuration options via dataclasses
- **Resource Management**: Proper file handling with context managers
- **Caching Support**: LRU cache for expensive computations
- **Batch Processing**: Process multiple files with error recovery
- **GPU Support**: CUDA acceleration for optical flow calculation
- **Robust Error Handling**: Custom exceptions with clear error messages
- **Logging**: Comprehensive logging instead of print statements
- **Waveform Validation**: Automatic validation of physiological waveforms
- **Type Hints**: Full type annotations for better IDE support and static analysis

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- GDCM library (for DICOM reading)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- OpenCV (with CUDA support for GPU acceleration)
- NumPy
- Matplotlib
- scikit-image
- h5py
- Polars
- NeuroKit2
- tsmoothie
- peakutils
- PyTorch (for SAM model)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TEE_optical_flow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Download SAM model checkpoint for segmentation

## Quick Start

### Basic Usage

```python
from optical_flow.optical_flow_dataset import OpticalFlowDataset
from optical_flow.api import analyze_optical_flow, plot_results

# Load HDF5 file
with OpticalFlowDataset('path/to/file.hdf5') as ds:
    # Analyze optical flow
    results = analyze_optical_flow(ds, param='velocity', label='rv')
    
    # Plot results
    plot_results(ds, param='velocity', label='rv', 
                 save_path='output/velocity_plot.png')
```

### Cardiac Cycle Detection

```python
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.config import default_cardiac_cycle_config

# Create detector
detector = create_detector('angle', 
                          cc_config=default_cardiac_cycle_config())

# Detect cardiac cycle
sys_frames, dia_frames = detector.detect(ds, param='velocity', label='rv_inner')
```

### Calculate Optical Flow from DICOM

```python
from optical_flow.calculate_optical_flow import process_video
from optical_flow.config import OpticalFlowCalculationConfig, default_optical_flow_config

# Create configuration (optional - defaults are used if not provided)
of_config = OpticalFlowCalculationConfig(
    lambda_value=0.15,           # Smoothing factor for DualTVL1
    moving_avg_window=4,         # Window size for mask smoothing
    min_mask_size=500,           # Minimum mask size in pixels
    waveform_flatness_threshold=0.05,  # Threshold for waveform validation
    ecg_sampling_rate=500,       # ECG sampling rate (Hz)
    art_sampling_rate=125        # Arterial pressure sampling rate (Hz)
)

# Process DICOM video and save to HDF5
process_video(
    dcm_path='path/to/dicom.dcm',
    save_path='output/optical_flow.hdf5',
    segmentor_model=your_sam_model,
    mode='RVIO_2class',          # 'A4C', 'RVIO_2class', or 'otsu'
    OF_algo='TVL1',              # 'TVL1' or 'deepflow'
    bkgd_comp='none',            # 'WASE' or 'none'
    no_saliency=False,           # Use saliency for optical flow
    include_waveforms=True,
    waveform_folder='path/to/waveforms',
    config=of_config,            # Optional: use default if not provided
    verbose=True
)
```

### Command-Line Usage

The `calculate_optical_flow.py` module can also be run from the command line:

```bash
python -m optical_flow.calculate_optical_flow \
    --nchunks 10 \
    --dcm_folder /path/to/dicom/files \
    --save_folder /path/to/output \
    --waveform_folder /path/to/waveforms \
    --checkpoint_dir /path/to/model/checkpoint \
    --arch vit_t \
    --verbose \
    --recalculate
```

**Command-line arguments:**
- `--nchunks`: Number of chunks to split processing (required)
- `--dcm_folder`: Folder containing DICOM files (required)
- `--save_folder`: Folder to save HDF5 files (required)
- `--waveform_folder`: Folder containing waveform files (optional)
- `--checkpoint_dir`: Directory containing model checkpoint and args.json (default: predefined path)
- `--arch`: Model architecture (default: 'vit_t')
- `--verbose`: Enable verbose output
- `--recalculate`: Recalculate even if files exist
- `--cuda_device`: CUDA device ID (default: '0')

## Architecture

The codebase is organized into modular components:

```
optical_flow/
├── api.py                      # High-level API functions
├── config.py                   # Configuration dataclasses
├── cardiac_cycle_detection.py  # Cardiac cycle detection classes
├── analysis.py                 # Data analysis functions
├── peak_detection.py           # Peak detection algorithms
├── visualization.py            # Plotting and visualization
├── plotting_utils.py           # Plotting helper functions
├── file_io.py                  # File I/O operations
├── batch_processing.py          # Batch processing utilities
├── cache.py                    # Caching functionality
├── optical_flow_dataset.py     # Dataset class for HDF5 files
├── calculate_optical_flow.py   # Optical flow calculation from DICOM
├── waveform_loader.py          # Waveform loading and validation
├── exceptions.py               # Custom exception classes
├── analyze_optical_flow.py     # Legacy analysis functions (maintained for compatibility)
└── optical_flow_utils.py       # Utility functions
```

### Module Responsibilities

- **`api.py`**: High-level entry points for common operations
- **`config.py`**: Configuration management with dataclasses
- **`cardiac_cycle_detection.py`**: Strategy pattern for cardiac cycle detection
- **`analysis.py`**: Histogram calculations, magnitude projections, centroid finding
- **`peak_detection.py`**: Peak detection in radial/longitudinal components
- **`visualization.py`**: Plotting and video generation
- **`file_io.py`**: HDF5, Pickle, and CSV I/O operations
- **`batch_processing.py`**: Batch processing with error recovery
- **`cache.py`**: LRU caching for expensive operations
- **`optical_flow_dataset.py`**: Dataset class with context manager support
- **`calculate_optical_flow.py`**: DICOM processing, mask prediction, and optical flow calculation
- **`waveform_loader.py`**: Loading and validation of physiological waveforms (ECG, arterial pressure, etc.)
- **`exceptions.py`**: Custom exception classes for error handling

## Usage Guide

### Loading Data

```python
from optical_flow.optical_flow_dataset import OpticalFlowDataset

# Using context manager (recommended)
with OpticalFlowDataset('data.hdf5') as ds:
    # Access data
    velocity = ds.get_velocity('rv')
    mask = ds.get_mask('rv')
    
    # Access metadata
    print(f"Frame rate: {ds.frame_rate} Hz")
    print(f"Pixel spacing: {ds.pixel_spacing} cm/pixel")
    print(f"Number of frames: {ds.nframes}")

# File is automatically closed when exiting context
```

### Cardiac Cycle Detection

```python
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.config import ecg_gated_config, arterial_gated_config

# Angle-based detection
angle_detector = create_detector('angle')
sys_frames, dia_frames = angle_detector.detect(ds, param='velocity', label='rv_inner')

# ECG-gated detection
ecg_detector = create_detector('ecg_lazy', cc_config=ecg_gated_config())
sys_frames, dia_frames = ecg_detector.detect(ds, ecg_arr=ds.ecg, sampling_rate=ds.ecg_sampling_rate)

# Arterial pressure-gated detection
art_detector = create_detector('arterial', cc_config=arterial_gated_config())
sys_frames, dia_frames = art_detector.detect(ds, art_arr=ds.art, sampling_rate=ds.art_sampling_rate)
```

### Analysis

```python
from optical_flow.analysis import calculate_3dhist, calculate_3dhist_radlong

# Calculate magnitude/angle histogram
masked_arr = ds.get_masked_arr('velocity', 'rv')
mag, ang, mag_edges, ang_edges, perc_hi = calculate_3dhist(
    masked_arr, ds.nframes, nbins=1000, percentile=99
)

# Calculate radial/longitudinal histograms
av_masks = ds.get_mask('av')
data_dict = calculate_3dhist_radlong(
    masked_arr, av_masks, ds.nframes,
    nbins=1000, perc_lo=1, perc_hi=99
)
rad_data = data_dict['radial']  # (mag_freq_arr, mag_edges, hi_arr, low_arr)
long_data = data_dict['longitudinal']
```

### Visualization

```python
from optical_flow.visualization import VisualizationManager
from optical_flow.config import VisualizationConfig, PeakDetectionConfig
from optical_flow.peak_detection import calculate_single_peaks
from tsmoothie.smoother import SpectralSmoother

# Create visualization manager
vis_config = VisualizationConfig(
    save_dir='output/plots',
    show_img=False,
    nbins=1000,
    show_sysdia_shading=True,      # Enable systole/diastole shading
    show_peak_annotations=True,     # Show peak value annotations
    print_report=False,            # Print statistics report
    return_statistics=False        # Return statistics tuple
)
vis_manager = VisualizationManager(vis_config)

# Plot heatmap
vis_manager.plot_heatmap(
    mag_arr, ang_arr, mag_edges, ang_edges,
    frame_times, 'velocity', 'cm/s',
    'filename', 'output/heatmap.png'
)

# Plot single component peak line plot
# Option 1: Provide pre-calculated peak data
peak_data = calculate_single_peaks(
    filt_arr, frame_times, sys_frames, dia_frames, ds.nframes,
    cc_method='angle'
)
vis_manager.plot_peak_line(
    filt_arr, frame_times, 'velocity', 'cm/s', 'rv',
    'filename', 'output/peak_line.png',
    peak_data=peak_data,
    sys_frames=sys_frames,
    dia_frames=dia_frames,
    nframes=ds.nframes,
    cc_method='angle',
    show_sysdia=True,              # Show systole/diastole shading
    mode='RVIO_2class',            # Dataset mode (shading only if != 'otsu')
    print_report=True,             # Print statistics report
    return_statistics=False,       # Return statistics tuple
    show_all_peaks=False           # Show all peaks or just cardiac cycle peaks
)

# Option 2: Let plot_peak_line calculate peaks internally
vis_manager.plot_peak_line(
    filt_arr=None,                 # Will be calculated from hi_arr
    frame_times=frame_times,
    param='velocity',
    param_unit='cm/s',
    label='rv',
    filename='filename',
    save_path='output/peak_line.png',
    hi_arr=hi_arr,                 # High percentile array
    sys_frames=sys_frames,
    dia_frames=dia_frames,
    nframes=ds.nframes,
    cc_method='angle',
    peak_config=PeakDetectionConfig(
        smooth_fraction=0.5,
        pad_len=20,
        peak_thres=0.2,
        min_dist=5
    ),
    show_sysdia=True,
    mode='RVIO_2class',
    print_report=True,
    return_statistics=True         # Returns tuple: (peak_sys, mean_sys, peak_e, mean_e, peak_l, mean_l, peak_a, mean_a, n_cycles)
)

# Plot radial/longitudinal peak line plot
vis_manager.plot_peak_line_radlong(
    rad_hi_arr, rad_lo_arr, long_hi_arr, long_lo_arr,
    frame_times, 'velocity', 'cm/s',
    'filename', 'output/radlong_peak_line.png',
    rad_peak_data=rad_peak_data,
    long_peak_data=long_peak_data,
    sys_frames=sys_frames,
    dia_frames=dia_frames,
    nframes=ds.nframes,
    cc_method='angle',
    show_sysdia=True,
    true_sysdia_mode='radial',     # Use radial or longitudinal for shading
    print_report=True,
    return_statistics=True         # Returns tuple of 18 values
)

# Create radial/longitudinal overlay video
vis_manager.visualize_radlong(
    rad_arr, long_arr, echo_arr, centroid_list,
    'filename', 'output/video.mp4', ds.nframes
)
```

### Peak Detection

#### Radial/Longitudinal Peak Detection

```python
from optical_flow.peak_detection import calculate_radlong_peaks
from optical_flow.config import PeakDetectionConfig

peak_config = PeakDetectionConfig(
    peak_thres=0.2,
    min_dist=5,
    pick_peak_by_subset=True
)

# Calculate peaks for radial/longitudinal components
results = calculate_radlong_peaks(
    hi_arr, lo_arr, frame_times,
    sys_frames, dia_frames, ds.nframes,
    cc_method='angle',
    peak_thres=peak_config.peak_thres,
    min_dist=peak_config.min_dist
)

# Access results
sys_px = results['sys_px']  # Systolic peak x-coordinates
e_px = results['e_px']      # e' peak x-coordinates
l_px = results['l_px']      # l' peak x-coordinates
a_px = results['a_px']      # a' peak x-coordinates
```

#### Single Component Peak Detection

```python
from optical_flow.peak_detection import calculate_single_peaks
from tsmoothie.smoother import SpectralSmoother

# First, smooth the high percentile array
smoother = SpectralSmoother(smooth_fraction=0.5, pad_len=20)
smoother.smooth(hi_arr)
filt_arr = smoother.smooth_data[0]

# Calculate peaks for single component
frame_times = np.arange(ds.nframes) * (1000 / ds.frame_rate)
peak_data = calculate_single_peaks(
    filt_arr, frame_times,
    sys_frames, dia_frames, ds.nframes,
    cc_method='angle',
    peak_thres=0.2,
    min_dist=5,
    pick_peak_by_subset=True,
    show_all_peaks=False  # Set to True to return all detected peaks
)

# Access results
sys_px = peak_data['sys_px']  # Systolic peak x-coordinates
sys_py = peak_data['sys_py']  # Systolic peak y-coordinates
e_px = peak_data['e_px']       # e' peak x-coordinates
e_py = peak_data['e_py']      # e' peak y-coordinates
l_px = peak_data['l_px']      # l' peak x-coordinates
l_py = peak_data['l_py']      # l' peak y-coordinates
a_px = peak_data['a_px']      # a' peak x-coordinates
a_py = peak_data['a_py']      # a' peak y-coordinates
true_sys = peak_data['true_sys']  # Systole frame intervals
true_dia = peak_data['true_dia']  # Diastole frame intervals

# If show_all_peaks=True, also get:
if 'all_px' in peak_data:
    all_px = peak_data['all_px']  # All detected peak x-coordinates
    all_py = peak_data['all_py']  # All detected peak y-coordinates
```

### Batch Processing

```python
from optical_flow.batch_processing import BatchProcessor
from optical_flow.file_io import CSVExporter

def process_file(filepath):
    """Custom processing function for each file."""
    with OpticalFlowDataset(filepath) as ds:
        # Your processing logic here
        results = analyze_optical_flow(ds, 'velocity', 'rv')
        # Save results, etc.
    return results

# Process folder
processor = BatchProcessor('input_folder', 'output_folder', verbose=True)
processor.process_chunk(file_list, start_idx=0, end_idx=100, process_func=process_file)
processor.save_errors()

# Aggregate results
CSVExporter.aggregate_pkl_files(
    param_list=['velocity'],
    label_list=['rv'],
    save_dir='output_folder'
)
```

## Configuration

The codebase uses dataclasses for configuration management, reducing parameter counts and improving maintainability.

### Configuration Classes

```python
from optical_flow.config import (
    CardiacCycleConfig,
    VisualizationConfig,
    ProcessingConfig,
    AnalysisConfig,
    PeakDetectionConfig,
    OpticalFlowCalculationConfig
)

# Cardiac cycle configuration
cc_config = CardiacCycleConfig(
    smooth_fraction=0.2,
    pad_len=20,
    rr_sys_ratio=0.333,
    sys_thres=0.9,
    dia_thres=0.5
)

# Visualization configuration
vis_config = VisualizationConfig(
    save_dir='output',
    show_plot=False,
    nbins=1000,
    colormap_mag='hot',
    colormap_ang='viridis',
    show_peak_annotations=True,        # Show peak value annotations on plots
    peak_marker_size=8,                # Size of peak markers
    peak_marker_style='+',             # Style of peak markers ('+', 'o', 'x', etc.)
    peak_annotation_fontsize=8,         # Font size for peak annotations
    peak_annotation_offset=(1.5, 1.5), # (x, y) offset for annotations in points
    radial_peak_color='r',             # Color for radial peak markers
    longitudinal_peak_color='b',        # Color for longitudinal peak markers
    systolic_peak_color='r',           # Color for systolic peak markers
    diastolic_peak_color='b',          # Color for diastolic peak markers
    show_sysdia_shading=False,         # Show systole/diastole shading on plots
    true_sysdia_mode='radial',         # Which component to use for shading ('radial' or 'longitudinal')
    print_report=False,                # Print statistics report to console
    return_statistics=False            # Return statistics tuple from plotting functions
)

# Processing configuration
proc_config = ProcessingConfig(
    recalculate=False,
    verbose=True,
    sampling_rate=500
)

# Analysis configuration
analysis_config = AnalysisConfig(
    percentile=99,
    perc_lo=1,
    perc_hi=99,
    av_filter_flag=True,
    av_savgol_window=10,
    av_savgol_poly=4
)

# Peak detection configuration
peak_config = PeakDetectionConfig(
    peak_thres=0.2,
    min_dist=5,
    pick_peak_by_subset=True
)

# Optical flow calculation configuration
of_config = OpticalFlowCalculationConfig(
    lambda_value=0.15,                    # Smoothing factor for DualTVL1 algorithm
    moving_avg_window=4,                  # Window size for moving average mask filtering
    moving_avg_threshold=0.49,            # Threshold for mask binarization
    min_mask_size=500,                    # Minimum mask size in pixels
    waveform_flatness_threshold=0.05,      # Maximum gradient for flat waveform detection
    pap_max_mean=100.0,                   # Maximum mean for PAP waveform (mmHg)
    cvp_max_mean=50.0,                    # Maximum mean for CVP waveform (mmHg)
    cvp_min_mean=-10.0,                  # Minimum mean for CVP waveform (mmHg)
    ecg_sampling_rate=500,                # ECG sampling rate (Hz)
    art_sampling_rate=125,                # Arterial pressure sampling rate (Hz)
    cvp_sampling_rate=125,                # CVP sampling rate (Hz)
    pap_sampling_rate=125                 # PAP sampling rate (Hz)
)
```

### Preset Configurations

```python
from optical_flow.config import (
    ecg_gated_config,
    arterial_gated_config,
    angle_detection_config,
    area_detection_config
)

# Use preset configurations
ecg_config = ecg_gated_config()
art_config = arterial_gated_config()
```

## Examples

### Example 1: Complete Analysis Pipeline

```python
from optical_flow.optical_flow_dataset import OpticalFlowDataset
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.analysis import calculate_3dhist_radlong
from optical_flow.peak_detection import calculate_radlong_peaks
from optical_flow.visualization import VisualizationManager
from optical_flow.config import *

# Load dataset
with OpticalFlowDataset('data.hdf5') as ds:
    # 1. Detect cardiac cycle
    detector = create_detector('ecg_lazy', cc_config=ecg_gated_config())
    sys_frames, dia_frames = detector.detect(
        ds, ecg_arr=ds.ecg, sampling_rate=ds.ecg_sampling_rate
    )
    
    # 2. Calculate radial/longitudinal components
    param_arr = ds.get_masked_arr('velocity', 'rv')
    av_masks = ds.get_mask('av')
    data_dict = calculate_3dhist_radlong(
        param_arr, av_masks, ds.nframes,
        nbins=1000, perc_lo=1, perc_hi=99
    )
    
    # 3. Detect peaks
    rad_hi = data_dict['radial'][2]  # High percentile array
    rad_lo = data_dict['radial'][3]  # Low percentile array
    frame_times = np.arange(ds.nframes) * (1000 / ds.frame_rate)
    
    peak_results = calculate_radlong_peaks(
        rad_hi, rad_lo, frame_times,
        sys_frames, dia_frames, ds.nframes,
        cc_method='ecg_lazy'
    )
    
    # 4. Visualize
    vis_manager = VisualizationManager(
        VisualizationConfig(save_dir='output', show_sysdia_shading=True),
        ProcessingConfig(verbose=True)
    )
    
    # Plot single component peak line
    hi_arr = data_dict['radial'][2]  # High percentile array
    smoother = SpectralSmoother(smooth_fraction=0.5, pad_len=20)
    smoother.smooth(hi_arr)
    filt_arr = smoother.smooth_data[0]
    frame_times = np.arange(ds.nframes) * (1000 / ds.frame_rate)
    
    peak_data = calculate_single_peaks(
        filt_arr, frame_times, sys_frames, dia_frames, ds.nframes,
        cc_method='ecg_lazy'
    )
    
    vis_manager.plot_peak_line(
        filt_arr, frame_times, 'velocity', 'cm/s', 'rv',
        'filename', 'output/peak_line.png',
        peak_data=peak_data,
        sys_frames=sys_frames,
        dia_frames=dia_frames,
        nframes=ds.nframes,
        cc_method='ecg_lazy',
        show_sysdia=True,
        mode='RVIO_2class',
        print_report=True,
        return_statistics=True
    )
```

### Example 2: Using the High-Level API

```python
from optical_flow.api import analyze_optical_flow, plot_results
from optical_flow.optical_flow_dataset import OpticalFlowDataset

with OpticalFlowDataset('data.hdf5') as ds:
    # Simple analysis
    results = analyze_optical_flow(ds, 'velocity', 'rv')
    
    # Simple plotting
    plot_results(ds, 'velocity', 'rv', 'output/plot.png')
```

### Example 3: Custom Configuration

```python
from optical_flow.config import CardiacCycleConfig, VisualizationConfig, OpticalFlowCalculationConfig
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.calculate_optical_flow import process_video

# Custom cardiac cycle configuration
custom_cc_config = CardiacCycleConfig(
    smooth_fraction=0.3,  # More smoothing
    pad_len=30,           # Longer padding
    rr_sys_ratio=0.35     # Different systole ratio
)

# Custom visualization configuration
custom_vis_config = VisualizationConfig(
    save_dir='custom_output',
    nbins=2000,           # Higher resolution
    colormap_mag='plasma', # Different colormap
    show_plot=True        # Display plots
)

# Custom optical flow calculation configuration
custom_of_config = OpticalFlowCalculationConfig(
    lambda_value=0.2,              # More smoothing for optical flow
    min_mask_size=1000,            # Larger minimum mask size
    waveform_flatness_threshold=0.03  # Stricter waveform validation
)

# Use custom configurations
detector = create_detector('angle', cc_config=custom_cc_config)
sys_frames, dia_frames = detector.detect(ds, param='velocity', label='rv_inner')

# Process video with custom configuration
process_video(
    dcm_path='input.dcm',
    save_path='output.hdf5',
    segmentor_model=model,
    config=custom_of_config,
    verbose=True
)
```

### Example 4: Processing DICOM Files with Waveforms

```python
from optical_flow.calculate_optical_flow import process_video, process_folder
from optical_flow.config import default_optical_flow_config

# Process a single DICOM file with waveforms
process_video(
    dcm_path='patient_001.dcm',
    save_path='output/patient_001.hdf5',
    segmentor_model=sam_model,
    mode='RVIO_2class',
    OF_algo='deepflow',
    include_waveforms=True,
    waveform_folder='waveforms/',
    config=default_optical_flow_config(),
    verbose=True
)

# Process a folder of DICOM files
process_folder(
    dcm_folder='dicom_files/',
    save_folder='output/',
    segmentor_model=sam_model,
    nchunks=10,
    chunk_index=0,
    mode='RVIO_2class',
    include_waveforms=True,
    waveform_folder='waveforms/',
    verbose=True
)
```

## Module Reference

### `api.py`
High-level API functions:
- `analyze_optical_flow()`: Main analysis entry point
- `plot_results()`: Visualization entry point
- `batch_process()`: Batch processing entry point

### `config.py`
Configuration dataclasses:
- `CardiacCycleConfig`: Cardiac cycle detection parameters
- `VisualizationConfig`: Plotting and visualization settings
- `ProcessingConfig`: Processing options
- `AnalysisConfig`: Analysis parameters
- `PeakDetectionConfig`: Peak detection settings
- `OpticalFlowCalculationConfig`: Optical flow calculation and processing parameters

### `cardiac_cycle_detection.py`
Cardiac cycle detection classes:
- `CardiacCycleDetector`: Base class
- `AngleDetector`: Angle-based detection
- `AreaDetector`: Area-based detection
- `ECGDetector`: ECG T-wave detection
- `ECGLazyDetector`: ECG lazy detection
- `ArterialDetector`: Arterial pressure detection
- `RTimeDetector`: DICOM metadata detection
- `create_detector()`: Factory function

### `analysis.py`
Analysis functions:
- `calculate_3dhist()`: Magnitude/angle histogram
- `calculate_3dhist_radlong()`: Radial/longitudinal histograms
- `calc_bidirectional_hist()`: Bidirectional histogram
- `calculate_comp_magnitude()`: Radial/longitudinal components
- `calc_AV_centroid()`: AV centroid calculation

### `peak_detection.py`
Peak detection:
- `PeakDetector`: Peak detection class for radial/longitudinal components
- `calculate_radlong_peaks()`: Calculate peaks for radial/longitudinal data
- `calculate_single_peaks()`: Calculate peaks for single component (non-radial/longitudinal) data

### `visualization.py`
Visualization:
- `VisualizationManager`: Main visualization class
- Methods:
  - `plot_radlong_heatmap()`: Plot radial/longitudinal heatmap
  - `plot_heatmap()`: Plot magnitude/angle heatmap
  - `plot_peak_line()`: Plot single component peak line plot with peak markers, systole/diastole shading, and statistics reporting
  - `plot_peak_line_radlong()`: Plot radial/longitudinal peak line plot with peak markers and statistics
  - `visualize_radlong()`: Create radial/longitudinal overlay video
  - `_calculate_single_peak_statistics()`: Calculate peak statistics for single component (internal helper)
  - `_calculate_peak_statistics()`: Calculate peak statistics for radial/longitudinal components (internal helper)

### `file_io.py`
File I/O:
- `HDF5Reader`: HDF5 file reader with context manager
- `HDF5Writer`: HDF5 file writer with context manager
- `PickleSerializer`: Pickle serialization
- `CSVExporter`: CSV export functionality

### `batch_processing.py`
Batch processing:
- `BatchProcessor`: Batch processing class with error recovery
- `analyze_hdf5_folder()`: Process folder of HDF5 files

### `cache.py`
Caching:
- `ComputationCache`: LRU cache implementation
- `cached_computation()`: Decorator for function caching

### `optical_flow_dataset.py`
Dataset class:
- `OpticalFlowDataset`: Main dataset class with context manager support
- Methods: `get_velocity()`, `get_accel()`, `get_pwr()`, `get_mask()`, `get_masked_arr()`

### `calculate_optical_flow.py`
Optical flow calculation from DICOM:
- `process_video()`: Main function to process DICOM video and calculate optical flow
- `process_folder()`: Process multiple DICOM files in a folder
- `predict_movie()`: Predict masks using SAM model
- `predict_movie_thres()`: Predict masks using Otsu thresholding
- `calculate_optical_flow()`: Calculate optical flow between two frames
- `_read_dicom_file()`: Read and parse DICOM file
- `_extract_dicom_metadata()`: Extract metadata from DICOM dataset
- `_save_optical_flow_to_hdf5()`: Save results to HDF5 file
- `_load_segmentor_model()`: Load SAM model from checkpoint

### `waveform_loader.py`
Waveform loading and validation:
- `load_all_waveforms()`: Load and validate all waveform files for a DICOM file
- `_load_waveform_file()`: Load waveform from numpy file
- `_validate_waveform_flatness()`: Check if waveform is flat
- `_validate_waveform_range()`: Validate waveform is within expected range

### `exceptions.py`
Custom exceptions:
- `OpticalFlowError`: Base exception for optical flow processing
- `DICOMReadError`: Raised when DICOM file cannot be read
- `WaveformLoadError`: Raised when waveform file cannot be loaded
- `WaveformValidationError`: Raised when waveform validation fails
- `OpticalFlowCalculationError`: Raised when optical flow calculation fails
- `ConfigurationError`: Raised when configuration is invalid

## Contributing

When contributing to this codebase:

1. Follow the modular architecture
2. Use configuration dataclasses instead of many parameters
3. Add type hints to all functions
4. Write docstrings for all public functions
5. Maintain backward compatibility when possible
6. Add tests for new functionality

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here]

## Contact

[Add contact information here]


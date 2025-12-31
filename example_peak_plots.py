#!/usr/bin/env python3
"""
Example script for generating peak line plots from HDF5 files.

This script demonstrates how to:
1. Load an HDF5 file containing optical flow data
2. Detect cardiac cycles using various methods
3. Calculate histograms and percentile arrays
4. Detect peaks (both single component and radial/longitudinal)
5. Generate peak line plots with statistics

Usage:
    python example_peak_plots.py <hdf5_filepath> [--output_dir OUTPUT_DIR] [--cc_method METHOD]
    
Arguments:
    hdf5_filepath: Path to the HDF5 file containing optical flow data
    --output_dir: Directory to save output plots (default: 'output')
    --cc_method: Cardiac cycle detection method (default: 'angle')
                 Options: 'angle', 'area', 'ecg', 'ecg_lazy', 'arterial', 'metadata'
    --param: Optical flow parameter to analyze (default: 'velocity')
             Options: 'velocity', 'acceleration', 'PWR'
    --label: Label for the region of interest (default: 'rv')
             Options: 'rv', 'lv', 'rv_inner', 'lv_inner', 'av', etc.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Add the optical_flow directory to the path if needed
sys.path.insert(0, str(Path(__file__).parent))

from optical_flow.optical_flow_dataset import OpticalFlowDataset
from optical_flow.cardiac_cycle_detection import create_detector
from optical_flow.analysis import calculate_3dhist, calculate_3dhist_radlong
from optical_flow.peak_detection import calculate_radlong_peaks, calculate_single_peaks
from optical_flow.visualization import VisualizationManager
from optical_flow.config import (
    VisualizationConfig, ProcessingConfig, PeakDetectionConfig,
    default_cardiac_cycle_config, default_peak_detection_config
)
from tsmoothie.smoother import SpectralSmoother
from optical_flow_utils import safe_makedir


def main():
    """Main function to generate peak line plots."""
    parser = argparse.ArgumentParser(
        description='Generate peak line plots from HDF5 optical flow files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('hdf5_filepath', type=str,
                       help='Path to the HDF5 file containing optical flow data')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save output plots (default: output)')
    parser.add_argument('--cc_method', type=str, default='angle',
                       choices=['angle', 'area', 'ecg', 'ecg_lazy', 'arterial', 'metadata'],
                       help='Cardiac cycle detection method (default: angle)')
    parser.add_argument('--param', type=str, default='velocity',
                       choices=['velocity', 'acceleration', 'PWR'],
                       help='Optical flow parameter to analyze (default: velocity)')
    parser.add_argument('--label', type=str, default='rv',
                       help='Label for the region of interest (default: rv)')
    parser.add_argument('--cc_label', type=str, default='rv_inner',
                       help='Label for cardiac cycle detection (default: rv_inner)')
    parser.add_argument('--percentile', type=int, default=99,
                       help='Percentile for histogram calculation (default: 99)')
    parser.add_argument('--smooth_fraction', type=float, default=0.5,
                       help='Smoothing fraction for peak detection (default: 0.5)')
    parser.add_argument('--show_sysdia', action='store_true',
                       help='Show systole/diastole shading on plots')
    parser.add_argument('--show_all_peaks', action='store_true',
                       help='Show all detected peaks instead of just cardiac cycle peaks')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.hdf5_filepath):
        print(f"Error: HDF5 file not found: {args.hdf5_filepath}")
        sys.exit(1)
    
    # Create output directory
    safe_makedir(args.output_dir)
    
    print("=" * 80)
    print("Peak Line Plot Generation")
    print("=" * 80)
    print(f"Input file: {args.hdf5_filepath}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameter: {args.param}")
    print(f"Label: {args.label}")
    print(f"Cardiac cycle method: {args.cc_method}")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/6] Loading HDF5 file...")
    with OpticalFlowDataset(args.hdf5_filepath) as ds:
        print(f"  - Frame rate: {ds.frame_rate} Hz")
        print(f"  - Number of frames: {ds.nframes}")
        print(f"  - Pixel spacing: {ds.pixel_spacing} cm/pixel")
        print(f"  - Mode: {ds.mode}")
        print(f"  - Available labels: {ds.accepted_labels}")
        
        # Detect cardiac cycle
        print(f"\n[2/6] Detecting cardiac cycle using '{args.cc_method}' method...")
        detector = create_detector(args.cc_method, cc_config=default_cardiac_cycle_config())
        
        # Prepare detection arguments based on method
        if args.cc_method in ['angle', 'area']:
            sys_frames, dia_frames = detector.detect(
                ds, param=args.param, label=args.cc_label
            )
        elif args.cc_method in ['ecg', 'ecg_lazy']:
            if ds.ecg is None:
                print(f"  Warning: ECG data not available. Switching to 'angle' method.")
                detector = create_detector('angle', cc_config=default_cardiac_cycle_config())
                sys_frames, dia_frames = detector.detect(ds, param=args.param, label=args.cc_label)
            else:
                sys_frames, dia_frames = detector.detect(
                    ds, ecg_arr=ds.ecg, sampling_rate=ds.ecg_sampling_rate
                )
        elif args.cc_method == 'arterial':
            if ds.art is None:
                print(f"  Warning: Arterial pressure data not available. Switching to 'angle' method.")
                detector = create_detector('angle', cc_config=default_cardiac_cycle_config())
                sys_frames, dia_frames = detector.detect(ds, param=args.param, label=args.cc_label)
            else:
                sys_frames, dia_frames = detector.detect(
                    ds, art_arr=ds.art, sampling_rate=ds.art_sampling_rate
                )
        elif args.cc_method == 'metadata':
            sys_frames, dia_frames = detector.detect(ds)
        
        print(f"  - Detected {len(sys_frames)} systole intervals")
        print(f"  - Detected {len(dia_frames)} diastole intervals")
        
        # Calculate frame times
        frame_times = np.arange(ds.nframes) * (1000 / ds.frame_rate)
        
        # Calculate single component histogram
        print(f"\n[3/6] Calculating single component histogram...")
        masked_arr = ds.get_masked_arr(args.param, args.label)
        if masked_arr is None:
            print(f"  Error: Could not retrieve {args.param} data for label '{args.label}'")
            sys.exit(1)
        
        mag, ang, mag_edges, ang_edges, hi_arr = calculate_3dhist(
            masked_arr, ds.nframes, nbins=1000, percentile=args.percentile
        )
        print(f"  - Calculated {args.percentile}th percentile array")
        
        # Smooth and detect peaks for single component
        print(f"\n[4/6] Detecting peaks for single component...")
        smoother = SpectralSmoother(
            smooth_fraction=args.smooth_fraction,
            pad_len=20
        )
        smoother.smooth(hi_arr)
        filt_arr = smoother.smooth_data[0]
        
        peak_config = default_peak_detection_config()
        peak_config.smooth_fraction = args.smooth_fraction
        
        single_peak_data = calculate_single_peaks(
            filt_arr, frame_times, sys_frames, dia_frames, ds.nframes,
            cc_method=args.cc_method,
            peak_thres=peak_config.peak_thres,
            min_dist=peak_config.min_dist,
            pick_peak_by_subset=peak_config.pick_peak_by_subset,
            show_all_peaks=args.show_all_peaks
        )
        print(f"  - Detected {len(single_peak_data['sys_px'])} systolic peaks")
        print(f"  - Detected {len(single_peak_data['e_px'])} e' peaks")
        print(f"  - Detected {len(single_peak_data['l_px'])} l' peaks")
        print(f"  - Detected {len(single_peak_data['a_px'])} a' peaks")
        
        # Calculate radial/longitudinal histograms (if AV masks available)
        print(f"\n[5/6] Calculating radial/longitudinal histograms...")
        radlong_data = None
        rad_peak_data = None
        long_peak_data = None
        
        if 'av' in ds.accepted_labels:
            av_masks = ds.get_mask('av')
            if av_masks is not None:
                radlong_data = calculate_3dhist_radlong(
                    masked_arr, av_masks, ds.nframes,
                    nbins=1000, perc_lo=1, perc_hi=args.percentile
                )
                
                rad_hi = radlong_data['radial'][2]
                rad_lo = radlong_data['radial'][3]
                long_hi = radlong_data['longitudinal'][2]
                long_lo = radlong_data['longitudinal'][3]
                
                print(f"  - Calculated radial/longitudinal percentile arrays")
                
                # Detect peaks for radial/longitudinal
                rad_peak_data = calculate_radlong_peaks(
                    rad_hi, rad_lo, frame_times,
                    sys_frames, dia_frames, ds.nframes,
                    cc_method=args.cc_method,
                    smooth_fraction=args.smooth_fraction,
                    pad_len=20,
                    peak_thres=peak_config.peak_thres,
                    min_dist=peak_config.min_dist,
                    pick_peak_by_subset=peak_config.pick_peak_by_subset
                )
                
                long_peak_data = calculate_radlong_peaks(
                    long_hi, long_lo, frame_times,
                    sys_frames, dia_frames, ds.nframes,
                    cc_method=args.cc_method,
                    smooth_fraction=args.smooth_fraction,
                    pad_len=20,
                    peak_thres=peak_config.peak_thres,
                    min_dist=peak_config.min_dist,
                    pick_peak_by_subset=peak_config.pick_peak_by_subset
                )
                
                print(f"  - Detected radial peaks: {len(rad_peak_data['sys_px'])} systolic, "
                      f"{len(rad_peak_data['e_px'])} e', {len(rad_peak_data['l_px'])} l', "
                      f"{len(rad_peak_data['a_px'])} a'")
                print(f"  - Detected longitudinal peaks: {len(long_peak_data['sys_px'])} systolic, "
                      f"{len(long_peak_data['e_px'])} e', {len(long_peak_data['l_px'])} l', "
                      f"{len(long_peak_data['a_px'])} a'")
        else:
            print(f"  - AV masks not available, skipping radial/longitudinal analysis")
        
        # Create visualization manager
        vis_config = VisualizationConfig(
            save_dir=args.output_dir,
            show_img=False,
            show_sysdia_shading=args.show_sysdia,
            show_peak_annotations=True,
            print_report=True,
            return_statistics=True
        )
        proc_config = ProcessingConfig(recalculate=True, verbose=False)
        vis_manager = VisualizationManager(vis_config, proc_config)
        
        # Generate plots
        print(f"\n[6/6] Generating peak line plots...")
        
        # Single component plot
        filename_base = os.path.basename(args.hdf5_filepath).replace('.hdf5', '')
        single_plot_path = os.path.join(
            args.output_dir,
            f"{filename_base}_{args.label}_{args.param}_{args.percentile}_{args.cc_method}_single_peak_line.png"
        )
        
        print(f"  - Generating single component plot: {single_plot_path}")
        single_stats = vis_manager.plot_peak_line(
            filt_arr, frame_times, args.param, ds._param_unit(args.param), args.label,
            filename_base, single_plot_path,
            peak_data=single_peak_data,
            sys_frames=sys_frames,
            dia_frames=dia_frames,
            nframes=ds.nframes,
            cc_method=args.cc_method,
            peak_config=peak_config,
            show_sysdia=args.show_sysdia,
            mode=ds.mode,
            print_report=True,
            return_statistics=True,
            show_all_peaks=args.show_all_peaks
        )
        
        if single_stats is not None:
            print(f"\n  Single Component Statistics:")
            print(f"    Peak Systolic: {single_stats[0]:.2f}, Mean Systolic: {single_stats[1]:.2f}")
            print(f"    Peak e': {single_stats[2]:.2f}, Mean e': {single_stats[3]:.2f}")
            print(f"    Peak l': {single_stats[4]:.2f}, Mean l': {single_stats[5]:.2f}")
            print(f"    Peak a': {single_stats[6]:.2f}, Mean a': {single_stats[7]:.2f}")
            print(f"    Number of cycles: {single_stats[8]}")
        
        # Radial/longitudinal plot (if available)
        if radlong_data is not None and rad_peak_data is not None and long_peak_data is not None:
            # Get arrays from radlong_data
            rad_hi = radlong_data['radial'][2]
            rad_lo = radlong_data['radial'][3]
            long_hi = radlong_data['longitudinal'][2]
            long_lo = radlong_data['longitudinal'][3]
            
            radlong_plot_path = os.path.join(
                args.output_dir,
                f"{filename_base}_{args.label}_{args.param}_{args.percentile}_{args.cc_method}_radlong_peak_line.png"
            )
            
            print(f"  - Generating radial/longitudinal plot: {radlong_plot_path}")
            radlong_stats = vis_manager.plot_peak_line_radlong(
                rad_hi, rad_lo, long_hi, long_lo,
                frame_times, args.param, ds._param_unit(args.param),
                filename_base, radlong_plot_path,
                rad_peak_data=rad_peak_data,
                long_peak_data=long_peak_data,
                sys_frames=sys_frames,
                dia_frames=dia_frames,
                nframes=ds.nframes,
                cc_method=args.cc_method,
                peak_config=peak_config,
                show_sysdia=args.show_sysdia,
                true_sysdia_mode='radial',
                print_report=True,
                return_statistics=True
            )
            
            if radlong_stats is not None:
                print(f"\n  Radial/Longitudinal Statistics:")
                print(f"    Radial - Peak Systolic: {radlong_stats[0]:.2f}, Mean: {radlong_stats[1]:.2f}")
                print(f"    Radial - Peak e': {radlong_stats[2]:.2f}, Mean: {radlong_stats[3]:.2f}")
                print(f"    Radial - Peak l': {radlong_stats[4]:.2f}, Mean: {radlong_stats[5]:.2f}")
                print(f"    Radial - Peak a': {radlong_stats[6]:.2f}, Mean: {radlong_stats[7]:.2f}")
                print(f"    Longitudinal - Peak Systolic: {radlong_stats[8]:.2f}, Mean: {radlong_stats[9]:.2f}")
                print(f"    Longitudinal - Peak e': {radlong_stats[10]:.2f}, Mean: {radlong_stats[11]:.2f}")
                print(f"    Longitudinal - Peak l': {radlong_stats[12]:.2f}, Mean: {radlong_stats[13]:.2f}")
                print(f"    Longitudinal - Peak a': {radlong_stats[14]:.2f}, Mean: {radlong_stats[15]:.2f}")
                print(f"    Radial cycles: {radlong_stats[16]}, Longitudinal cycles: {radlong_stats[17]}")
        
        print("\n" + "=" * 80)
        print("Peak line plot generation complete!")
        print(f"Plots saved to: {args.output_dir}")
        print("=" * 80)


if __name__ == '__main__':
    main()


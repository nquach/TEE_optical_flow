"""
Visualization module for optical flow analysis.

This module provides functions for creating heatmaps, plots, and visualizations
of optical flow data, decoupled from the dataset object.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm
import imageio.v3 as iio
from tqdm import tqdm
import gc
from skimage.color import gray2rgb
from typing import Optional, Dict, List, Tuple, Union

from optical_flow.optical_flow_utils import safe_makedir, fix_ecg
from optical_flow.config import VisualizationConfig, ProcessingConfig, AnalysisConfig, PeakDetectionConfig
from optical_flow.peak_detection import calculate_radlong_peaks, calculate_single_peaks
from tsmoothie.smoother import SpectralSmoother
import peakutils
from optical_flow.plotting_utils import (
    add_systole_diastole_shading, plot_waveform_with_shading,
    create_heatmap_figure, setup_colorbar, annotate_peaks
)


class VisualizationManager:
    """Manages visualization operations for optical flow data."""
    
    def __init__(self, vis_config: Optional[VisualizationConfig] = None,
                 proc_config: Optional[ProcessingConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None):
        self.vis_config = vis_config or VisualizationConfig()
        self.proc_config = proc_config or ProcessingConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
    
    def plot_radlong_heatmap(self, rad_mag_freq_arr: np.ndarray, long_mag_freq_arr: np.ndarray,
                            rad_mag_edges: np.ndarray, long_mag_edges: np.ndarray,
                            frame_times: np.ndarray, param: str, param_unit: str,
                            filename: str, save_path: str,
                            waveform_data: Optional[np.ndarray] = None,
                            waveform_times: Optional[np.ndarray] = None,
                            sampling_rate: Optional[int] = None,
                            sys_frames: Optional[list] = None,
                            dia_frames: Optional[list] = None,
                            nframes: int = None,
                            cc_method: str = 'angle',
                            show_sysdia: bool = False):
        """
        Plot radial/longitudinal heatmap.
        
        Args:
            rad_mag_freq_arr: Radial magnitude frequency array
            long_mag_freq_arr: Longitudinal magnitude frequency array
            rad_mag_edges: Radial magnitude bin edges
            long_mag_edges: Longitudinal magnitude bin edges
            frame_times: Time array for frames
            param: Parameter name
            param_unit: Parameter unit string
            filename: Base filename
            save_path: Full save path
            waveform_data: Optional waveform data
            waveform_times: Optional waveform time array
            sampling_rate: Sampling rate for waveform
            sys_frames: Systole frame intervals
            dia_frames: Diastole frame intervals
            nframes: Total number of frames
            cc_method: Cardiac cycle detection method
            show_sysdia: Whether to show systole/diastole shading
        """
        if os.path.exists(save_path) and not self.proc_config.recalculate:
            print(f'{save_path} already exists, skipping!')
            return None
        
        waveform_exists = waveform_data is not None
        show_waveform = waveform_exists and show_sysdia
        
        fig, axes = create_heatmap_figure(show_waveform=show_waveform, show_sysdia=show_sysdia)
        
        if show_waveform:
            ax1, ax2, ax_t = axes[0], axes[1], axes[2]
            if 'ecg' in cc_method:
                waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
            plot_waveform_with_shading(ax_t, waveform_data, waveform_times,
                                      frame_times, sys_frames, dia_frames, nframes)
        elif show_sysdia:
            ax1, ax2, ax_t = axes[0], axes[1], axes[2]
            add_systole_diastole_shading(ax_t, frame_times, sys_frames, dia_frames, nframes)
            ax_t.set_xlabel('Time (ms)')
        else:
            ax1, ax2 = axes[0], axes[1]
            axes[1].set_xlabel('Time (ms)')
        
        rad_norm = LogNorm(vmin=np.min(rad_mag_freq_arr), vmax=np.max(rad_mag_freq_arr))
        long_norm = LogNorm(vmin=np.min(long_mag_freq_arr), vmax=np.max(long_mag_freq_arr))
        
        ygrid_rad = rad_mag_edges
        ygrid_long = long_mag_edges
        
        # Create frame_times_edges for pcolormesh (needs nframes + 1 elements)
        if len(frame_times) > 1:
            dt = frame_times[1] - frame_times[0]
            frame_times_edges = np.linspace(frame_times[0] - dt/2, frame_times[-1] + dt/2, nframes + 1)
        else:
            # Fallback for edge case
            dt = 1000 / nframes if nframes > 0 else 1
            frame_times_edges = np.linspace(frame_times[0] - dt/2, frame_times[0] + dt/2, nframes + 1)
        
        plt1 = ax1.pcolormesh(frame_times_edges, ygrid_rad, rad_mag_freq_arr.T,
                              norm=rad_norm, cmap=self.vis_config.colormap_mag)
        ax1.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        ax1.set_title('Radial ' + param.capitalize() + ' vs Time (ms)')
        
        plt2 = ax2.pcolormesh(frame_times_edges, ygrid_long, long_mag_freq_arr.T,
                              norm=long_norm, cmap=self.vis_config.colormap_mag)
        ax2.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        ax2.set_title('Longitudinal ' + param.capitalize() + ' vs Time (ms)')
        
        if self.vis_config.invert_rad_yaxis:
            ax1.invert_yaxis()
        if self.vis_config.invert_long_yaxis:
            ax2.invert_yaxis()
        
        setup_colorbar(plt1, ax1, 'log(freq)')
        setup_colorbar(plt2, ax2, 'log(freq)')
        
        fig.savefig(save_path)
        if not self.vis_config.show_img:
            plt.close(fig)
        
        return fig
    
    def plot_heatmap(self, mag_arr: np.ndarray, ang_arr: np.ndarray,
                    mag_edges: np.ndarray, ang_edges: np.ndarray,
                    frame_times: np.ndarray, param: str, param_unit: str,
                    filename: str, save_path: str,
                    waveform_data: Optional[np.ndarray] = None,
                    waveform_times: Optional[np.ndarray] = None,
                    sampling_rate: Optional[int] = None,
                    sys_frames: Optional[list] = None,
                    dia_frames: Optional[list] = None,
                    nframes: int = None,
                    cc_method: str = 'angle',
                    show_sysdia: bool = False):
        """
        Plot magnitude/angle heatmap.
        
        Args:
            mag_arr: Magnitude frequency array
            ang_arr: Angle frequency array
            mag_edges: Magnitude bin edges
            ang_edges: Angle bin edges
            frame_times: Time array for frames
            param: Parameter name
            param_unit: Parameter unit string
            filename: Base filename
            save_path: Full save path
            waveform_data: Optional waveform data
            waveform_times: Optional waveform time array
            sampling_rate: Sampling rate for waveform
            sys_frames: Systole frame intervals
            dia_frames: Diastole frame intervals
            nframes: Total number of frames
            cc_method: Cardiac cycle detection method
            show_sysdia: Whether to show systole/diastole shading
        """
        if os.path.exists(save_path) and not self.proc_config.recalculate:
            print(f'{save_path} already exists, skipping!')
            return None
        
        waveform_exists = waveform_data is not None
        show_waveform = waveform_exists and show_sysdia
        
        fig, axes = create_heatmap_figure(show_waveform=show_waveform, show_sysdia=show_sysdia)
        
        if show_waveform:
            ax1, ax2, ax_t = axes[0], axes[1], axes[2]
            if 'ecg' in cc_method:
                waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
            plot_waveform_with_shading(ax_t, waveform_data, waveform_times,
                                      frame_times, sys_frames, dia_frames, nframes)
        elif show_sysdia:
            ax1, ax2, ax_t = axes[0], axes[1], axes[2]
            add_systole_diastole_shading(ax_t, frame_times, sys_frames, dia_frames, nframes)
            ax_t.legend(loc='lower right')
        else:
            ax1, ax2 = axes[0], axes[1]
            axes[1].set_xlabel('Time (ms)')
        
        mag_norm = LogNorm(vmin=np.min(mag_arr), vmax=np.max(mag_arr))
        ang_norm = LogNorm(vmin=np.min(ang_arr), vmax=np.max(ang_arr))
        
        ygrid_mag = mag_edges
        ygrid_ang = ang_edges * 180 / np.pi
        
        # Create frame_times_edges for pcolormesh (needs nframes + 1 elements)
        if len(frame_times) > 1:
            dt = frame_times[1] - frame_times[0]
            frame_times_edges = np.linspace(frame_times[0] - dt/2, frame_times[-1] + dt/2, nframes + 1)
        else:
            # Fallback for edge case
            dt = 1000 / nframes if nframes > 0 else 1
            frame_times_edges = np.linspace(frame_times[0] - dt/2, frame_times[0] + dt/2, nframes + 1)
        
        plt1 = ax1.pcolormesh(frame_times_edges, ygrid_mag, mag_arr.T,
                              norm=mag_norm, cmap=self.vis_config.colormap_mag)
        ax1.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        ax1.set_title('Magnitude of ' + param.capitalize() + ' vs Time (ms)')
        
        plt2 = ax2.pcolormesh(frame_times_edges, ygrid_ang, ang_arr.T,
                              norm=ang_norm, cmap=self.vis_config.colormap_ang)
        ax2.set_ylabel('Angle (deg)')
        
        if not waveform_exists and not show_sysdia:
            ax2.set_xlabel('Time (ms)')
        else:
            axes[-1].set_xlabel('Time (ms)')
        
        setup_colorbar(plt1, ax1, 'log(freq)')
        setup_colorbar(plt2, ax2, 'log(freq)')
        
        fig.savefig(save_path)
        if not self.vis_config.show_img:
            plt.close(fig)
        
        return fig
    
    def visualize_radlong(self, rad_arr: np.ndarray, long_arr: np.ndarray,
                        echo_arr: np.ndarray, centroid_list: list,
                        filename: str, save_path: str, nframes: int):
        """
        Create radial/longitudinal overlay video.
        
        Args:
            rad_arr: Radial component array
            long_arr: Longitudinal component array
            echo_arr: Echo image array
            centroid_list: List of centroid coordinates
            filename: Base filename
            save_path: Full save path
            nframes: Total number of frames
        """
        pixel_arr = gray2rgb(echo_arr)
        
        rad_list = np.split(rad_arr, rad_arr.shape[0])
        long_list = np.split(long_arr, long_arr.shape[0])
        rad_arr_sqz = [np.squeeze(arr) for arr in rad_list]
        long_arr_sqz = [np.squeeze(arr) for arr in long_list]
        norm = matplotlib.colors.CenteredNorm()
        
        rad_norm_list = []
        long_norm_list = []
        if self.proc_config.verbose:
            print('Converting to colormap...')
        
        for rad_arr_frame, long_arr_frame in zip(rad_arr_sqz, long_arr_sqz):
            rad_norm = norm(rad_arr_frame)
            long_norm = norm(long_arr_frame)
            rad_rgb = plt.cm.get_cmap(self.vis_config.colormap_rad)(rad_norm)
            long_rgb = plt.cm.get_cmap(self.vis_config.colormap_long)(long_norm)
            rad_norm_list.append(rad_rgb[:, :, 0:3])
            long_norm_list.append(long_rgb[:, :, 0:3])
        
        rad_rgb_arr = np.stack(rad_norm_list)
        long_rgb_arr = np.stack(long_norm_list)
        
        # Clean up intermediate arrays
        del rad_norm_list, long_norm_list, rad_list, long_list
        del rad_arr, long_arr, rad_arr_sqz, long_arr_sqz, norm
        gc.collect()
        
        if self.proc_config.verbose:
            print('Overlaying colormaps and greyscale imgs...')
        
        overlay_arr = self._overlay3(pixel_arr[0:nframes, ...], rad_rgb_arr, long_rgb_arr)
        del pixel_arr, rad_rgb_arr, long_rgb_arr
        gc.collect()
        
        safe_makedir(os.path.dirname(save_path))
        writer = iio.get_writer(save_path, fps=self.vis_config.fps)
        
        for i in tqdm(range(nframes), disable=(not self.proc_config.verbose)):
            writer.append_data(overlay_arr[i, ...])
        writer.close()
    
    def _calculate_peak_statistics(self, rad_peak_data: Dict, long_peak_data: Dict) -> Dict:
        """
        Calculate peak statistics for radial and longitudinal components.
        
        Args:
            rad_peak_data: Dictionary with radial peak data
            long_peak_data: Dictionary with longitudinal peak data
        
        Returns:
            Dictionary with statistics keys:
            - rad_peak_sys, rad_mean_sys, rad_n_cycles
            - rad_peak_e, rad_mean_e, rad_peak_l, rad_mean_l, rad_peak_a, rad_mean_a
            - long_peak_sys, long_mean_sys, long_n_cycles
            - long_peak_e, long_mean_e, long_peak_l, long_mean_l, long_peak_a, long_mean_a
        """
        stats = {}
        
        # Radial statistics
        if len(rad_peak_data.get('sys_py', [])) > 0:
            stats['rad_peak_sys'] = np.max(np.abs(rad_peak_data['sys_py']))
            stats['rad_mean_sys'] = np.mean(np.abs(rad_peak_data['sys_py']))
            stats['rad_n_cycles'] = len(rad_peak_data['sys_py'])
        else:
            stats['rad_peak_sys'] = 0.0
            stats['rad_mean_sys'] = 0.0
            stats['rad_n_cycles'] = 0
        
        if len(rad_peak_data.get('e_py', [])) > 0:
            stats['rad_peak_e'] = np.max(np.abs(rad_peak_data['e_py']))
            stats['rad_mean_e'] = np.mean(np.abs(rad_peak_data['e_py']))
        else:
            stats['rad_peak_e'] = 0.0
            stats['rad_mean_e'] = 0.0
        
        if len(rad_peak_data.get('l_py', [])) > 0:
            stats['rad_peak_l'] = np.max(np.abs(rad_peak_data['l_py']))
            stats['rad_mean_l'] = np.mean(np.abs(rad_peak_data['l_py']))
        else:
            stats['rad_peak_l'] = 0.0
            stats['rad_mean_l'] = 0.0
        
        if len(rad_peak_data.get('a_py', [])) > 0:
            stats['rad_peak_a'] = np.max(np.abs(rad_peak_data['a_py']))
            stats['rad_mean_a'] = np.mean(np.abs(rad_peak_data['a_py']))
        else:
            stats['rad_peak_a'] = 0.0
            stats['rad_mean_a'] = 0.0
        
        # Longitudinal statistics
        if len(long_peak_data.get('sys_py', [])) > 0:
            stats['long_peak_sys'] = np.max(np.abs(long_peak_data['sys_py']))
            stats['long_mean_sys'] = np.mean(np.abs(long_peak_data['sys_py']))
            stats['long_n_cycles'] = len(long_peak_data['sys_py'])
        else:
            stats['long_peak_sys'] = 0.0
            stats['long_mean_sys'] = 0.0
            stats['long_n_cycles'] = 0
        
        if len(long_peak_data.get('e_py', [])) > 0:
            stats['long_peak_e'] = np.max(np.abs(long_peak_data['e_py']))
            stats['long_mean_e'] = np.mean(np.abs(long_peak_data['e_py']))
        else:
            stats['long_peak_e'] = 0.0
            stats['long_mean_e'] = 0.0
        
        if len(long_peak_data.get('l_py', [])) > 0:
            stats['long_peak_l'] = np.max(np.abs(long_peak_data['l_py']))
            stats['long_mean_l'] = np.mean(np.abs(long_peak_data['l_py']))
        else:
            stats['long_peak_l'] = 0.0
            stats['long_mean_l'] = 0.0
        
        if len(long_peak_data.get('a_py', [])) > 0:
            stats['long_peak_a'] = np.max(np.abs(long_peak_data['a_py']))
            stats['long_mean_a'] = np.mean(np.abs(long_peak_data['a_py']))
        else:
            stats['long_peak_a'] = 0.0
            stats['long_mean_a'] = 0.0
        
        return stats
    
    def _calculate_single_peak_statistics(self, peak_data: Dict) -> Dict:
        """
        Calculate peak statistics for single component.
        
        Args:
            peak_data: Dictionary with peak data
        
        Returns:
            Dictionary with statistics:
            - peak_sys, mean_sys, n_cycles
            - peak_e, mean_e, peak_l, mean_l, peak_a, mean_a
        """
        stats = {}
        
        if len(peak_data.get('sys_py', [])) > 0:
            stats['peak_sys'] = np.max(peak_data['sys_py'])
            stats['mean_sys'] = np.mean(peak_data['sys_py'])
            stats['n_cycles'] = len(peak_data['sys_py'])
        else:
            stats['peak_sys'] = 0.0
            stats['mean_sys'] = 0.0
            stats['n_cycles'] = 0
        
        if len(peak_data.get('e_py', [])) > 0:
            stats['peak_e'] = np.max(peak_data['e_py'])
            stats['mean_e'] = np.mean(peak_data['e_py'])
        else:
            stats['peak_e'] = 0.0
            stats['mean_e'] = 0.0
        
        if len(peak_data.get('l_py', [])) > 0:
            stats['peak_l'] = np.max(peak_data['l_py'])
            stats['mean_l'] = np.mean(peak_data['l_py'])
        else:
            stats['peak_l'] = 0.0
            stats['mean_l'] = 0.0
        
        if len(peak_data.get('a_py', [])) > 0:
            stats['peak_a'] = np.max(peak_data['a_py'])
            stats['mean_a'] = np.mean(peak_data['a_py'])
        else:
            stats['peak_a'] = 0.0
            stats['mean_a'] = 0.0
        
        return stats
    
    def plot_peak_line_radlong(self, 
                               rad_hi_arr: np.ndarray, rad_lo_arr: np.ndarray,
                               long_hi_arr: np.ndarray, long_lo_arr: np.ndarray,
                               frame_times: np.ndarray, param: str, param_unit: str,
                               filename: str, save_path: str,
                               rad_peak_data: Optional[Dict] = None,
                               long_peak_data: Optional[Dict] = None,
                               waveform_data: Optional[np.ndarray] = None,
                               waveform_times: Optional[np.ndarray] = None,
                               sampling_rate: Optional[int] = None,
                               sys_frames: Optional[List[Tuple[int, int]]] = None,
                               dia_frames: Optional[List[Tuple[int, int]]] = None,
                               nframes: int = None,
                               cc_method: str = 'angle',
                               peak_config: Optional[PeakDetectionConfig] = None,
                               show_sysdia: Optional[bool] = None,
                               true_sysdia_mode: Optional[str] = None,
                               print_report: Optional[bool] = None,
                               return_statistics: Optional[bool] = None) -> Optional[Union[plt.Figure, Tuple[float, ...]]]:
        """
        Plot radial/longitudinal peak line plot with peak markers.
        
        Args:
            rad_hi_arr: Radial high percentile array
            rad_lo_arr: Radial low percentile array
            long_hi_arr: Longitudinal high percentile array
            long_lo_arr: Longitudinal low percentile array
            frame_times: Time array for frames
            param: Parameter name
            param_unit: Parameter unit string
            filename: Base filename
            save_path: Full save path
            rad_peak_data: Dictionary with radial peak data (from calculate_radlong_peaks)
            long_peak_data: Dictionary with longitudinal peak data
            waveform_data: Optional waveform data
            waveform_times: Optional waveform time array
            sampling_rate: Sampling rate for waveform
            sys_frames: Systole frame intervals
            dia_frames: Diastole frame intervals
            nframes: Total number of frames
            cc_method: Cardiac cycle detection method
            peak_config: Peak detection configuration (used if peak_data not provided)
            show_sysdia: Whether to show systole/diastole shading
            true_sysdia_mode: Which component to use for shading ('radial' or 'longitudinal')
            print_report: Whether to print statistics report
            return_statistics: Whether to return statistics tuple
        
        Returns:
            matplotlib Figure object, or tuple of 18 statistics values if return_statistics=True
        """
        if os.path.exists(save_path) and not self.proc_config.recalculate:
            print(f'{save_path} already exists, skipping!')
            return None
        
        # Use config values as defaults if parameters not provided
        if show_sysdia is None:
            show_sysdia = self.vis_config.show_sysdia_shading
        if true_sysdia_mode is None:
            true_sysdia_mode = self.vis_config.true_sysdia_mode
        if print_report is None:
            print_report = self.vis_config.print_report
        if return_statistics is None:
            return_statistics = self.vis_config.return_statistics
        
        if peak_config is None:
            from optical_flow.config import default_peak_detection_config
            peak_config = default_peak_detection_config()
        
        # Calculate peaks if not provided
        if rad_peak_data is None:
            rad_peak_data = calculate_radlong_peaks(
                rad_hi_arr, rad_lo_arr, frame_times,
                sys_frames or [], dia_frames or [], nframes,
                cc_method=cc_method,
                smooth_fraction=peak_config.smooth_fraction,
                pad_len=peak_config.pad_len,
                peak_thres=peak_config.peak_thres,
                min_dist=peak_config.min_dist,
                pick_peak_by_subset=peak_config.pick_peak_by_subset
            )
        
        if long_peak_data is None:
            long_peak_data = calculate_radlong_peaks(
                long_hi_arr, long_lo_arr, frame_times,
                sys_frames or [], dia_frames or [], nframes,
                cc_method=cc_method,
                smooth_fraction=peak_config.smooth_fraction,
                pad_len=peak_config.pad_len,
                peak_thres=peak_config.peak_thres,
                min_dist=peak_config.min_dist,
                pick_peak_by_subset=peak_config.pick_peak_by_subset
            )
        
        waveform_exists = waveform_data is not None
        # Automatically show waveform subplot if cc_method indicates waveform-based detection
        should_show_waveform = (cc_method in ['ecg', 'ecg_lazy', 'arterial']) or waveform_exists
        show_waveform = should_show_waveform and waveform_exists
        
        # Create figure
        if should_show_waveform:
            fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(8, 6))
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        
        # Plot filtered arrays as dotted lines and store handles for legend
        radline, = ax.plot(frame_times, rad_peak_data['filt_hi'], 'r:', label='Radial High')
        ax.plot(frame_times, rad_peak_data['filt_lo'], 'r:', label='Radial Low')
        longline, = ax.plot(frame_times, long_peak_data['filt_hi'], 'c:', label='Longitudinal High')
        ax.plot(frame_times, long_peak_data['filt_lo'], 'c:', label='Longitudinal Low')
        
        # Plot peak markers
        rad_color = self.vis_config.radial_peak_color
        long_color = self.vis_config.longitudinal_peak_color
        sys_color = self.vis_config.systolic_peak_color
        dia_color = self.vis_config.diastolic_peak_color
        
        # Radial peaks (only plot if arrays are not empty)
        if len(rad_peak_data['sys_px']) > 0:
            ax.plot(rad_peak_data['sys_px'], rad_peak_data['sys_py'], 
                   sys_color + self.vis_config.peak_marker_style, 
                   markersize=self.vis_config.peak_marker_size)
        if len(rad_peak_data['e_px']) > 0:
            ax.plot(rad_peak_data['e_px'], rad_peak_data['e_py'], 
                   rad_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        if len(rad_peak_data['l_px']) > 0:
            ax.plot(rad_peak_data['l_px'], rad_peak_data['l_py'], 
                   rad_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        if len(rad_peak_data['a_px']) > 0:
            ax.plot(rad_peak_data['a_px'], rad_peak_data['a_py'], 
                   rad_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        
        # Longitudinal peaks (only plot if arrays are not empty)
        if len(long_peak_data['sys_px']) > 0:
            ax.plot(long_peak_data['sys_px'], long_peak_data['sys_py'], 
                   long_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        if len(long_peak_data['e_px']) > 0:
            ax.plot(long_peak_data['e_px'], long_peak_data['e_py'], 
                   long_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        if len(long_peak_data['l_px']) > 0:
            ax.plot(long_peak_data['l_px'], long_peak_data['l_py'], 
                   long_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        if len(long_peak_data['a_px']) > 0:
            ax.plot(long_peak_data['a_px'], long_peak_data['a_py'], 
                   long_color + self.vis_config.peak_marker_style,
                   markersize=self.vis_config.peak_marker_size)
        
        # Annotate peaks if enabled
        if self.vis_config.show_peak_annotations:
            # Radial annotations
            if len(rad_peak_data['e_px']) > 0:
                annotate_peaks(ax, rad_peak_data['e_px'], rad_peak_data['e_py'],
                              color=rad_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(rad_peak_data['l_px']) > 0:
                annotate_peaks(ax, rad_peak_data['l_px'], rad_peak_data['l_py'],
                              color=rad_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(rad_peak_data['a_px']) > 0:
                annotate_peaks(ax, rad_peak_data['a_px'], rad_peak_data['a_py'],
                              color=rad_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(rad_peak_data['sys_px']) > 0:
                annotate_peaks(ax, rad_peak_data['sys_px'], rad_peak_data['sys_py'],
                              color=rad_color, offset=(self.vis_config.peak_annotation_offset[0], 
                                                      -self.vis_config.peak_annotation_offset[1]),
                              fontsize=self.vis_config.peak_annotation_fontsize)
            
            # Longitudinal annotations
            if len(long_peak_data['e_px']) > 0:
                annotate_peaks(ax, long_peak_data['e_px'], long_peak_data['e_py'],
                              color=long_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(long_peak_data['l_px']) > 0:
                annotate_peaks(ax, long_peak_data['l_px'], long_peak_data['l_py'],
                              color=long_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(long_peak_data['a_px']) > 0:
                annotate_peaks(ax, long_peak_data['a_px'], long_peak_data['a_py'],
                              color=long_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(long_peak_data['sys_px']) > 0:
                annotate_peaks(ax, long_peak_data['sys_px'], long_peak_data['sys_py'],
                              color=long_color, offset=(self.vis_config.peak_annotation_offset[0],
                                                       -self.vis_config.peak_annotation_offset[1]),
                              fontsize=self.vis_config.peak_annotation_fontsize)
        
        # Set labels and title
        ax.set_title(param.capitalize() + ' vs Time')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        
        # Add systole/diastole shading if enabled
        sys_label = None
        dia_label = None
        if show_sysdia:
            # Select which component's true_sys/true_dia to use
            if true_sysdia_mode == 'radial':
                true_sys = rad_peak_data.get('true_sys', [])
                true_dia = rad_peak_data.get('true_dia', [])
            else:
                true_sys = long_peak_data.get('true_sys', [])
                true_dia = long_peak_data.get('true_dia', [])
            
            # Add shading
            if len(true_sys) > 0:
                counter = 0
                for start, stop in true_sys:
                    if nframes is not None and stop >= nframes:
                        stop = nframes - 1
                    if counter == 0:
                        sys_label = ax.axvspan(frame_times[start], frame_times[stop],
                                             facecolor='0.8', alpha=0.5)
                    else:
                        ax.axvspan(frame_times[start], frame_times[stop],
                                  facecolor='0.8', alpha=0.5)
                    counter += 1
            
            if len(true_dia) > 0:
                counter = 0
                for start, stop in true_dia:
                    if nframes is not None and stop >= nframes:
                        stop = nframes - 1
                    if counter == 0:
                        dia_label = ax.axvspan(frame_times[start], frame_times[stop],
                                              facecolor='0.8', alpha=0.25)
                    else:
                        ax.axvspan(frame_times[start], frame_times[stop],
                                  facecolor='0.8', alpha=0.25)
                    counter += 1
        
        # Add custom legend
        if show_sysdia and sys_label is not None and dia_label is not None:
            ax.legend([radline, longline, sys_label, dia_label],
                     ['Radial Component', 'Longitudinal Component', 'Systole', 'Diastole'],
                     loc='lower right')
        else:
            ax.legend([radline, longline],
                     ['Radial Component', 'Longitudinal Component'],
                     loc='lower right')
        
        # Add waveform overlay if provided
        if should_show_waveform:
            if waveform_data is not None:
                if 'ecg' in cc_method:
                    if waveform_times is None:
                        waveform_times = np.arange(waveform_data.size) * (1000 / sampling_rate)
                    waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
                    ax2.plot(waveform_times, waveform_data)
                    ax2.set_ylabel('Voltage (mV)')
                elif cc_method == 'arterial':
                    if waveform_times is None:
                        waveform_times = np.arange(waveform_data.size) * (1000 / sampling_rate)
                    ax2.plot(waveform_times, waveform_data)
                    ax2.set_ylabel('Pressure (mmHg)')
                ax2.set_xlabel('Time (ms)')
            else:
                # Waveform subplot created but no data available
                ax2.text(0.5, 0.5, 'Waveform data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                if 'ecg' in cc_method:
                    ax2.set_ylabel('Voltage (mV)')
                elif cc_method == 'arterial':
                    ax2.set_ylabel('Pressure (mmHg)')
                ax2.set_xlabel('Time (ms)')
        
        # Calculate statistics
        stats = self._calculate_peak_statistics(rad_peak_data, long_peak_data)
        
        # Print report if enabled
        if print_report:
            label = 'rv'  # Default label, could be passed as parameter if needed
            print('=====================')
            print('RADIAL COMPONENT:')
            print('----------------')
            if len(rad_peak_data.get('sys_py', [])) > 0:
                print(f'Global peak systolic {label.upper()} {param}: {stats["rad_peak_sys"]}')
                print(f'Global mean systolic {label.upper()} {param}: {stats["rad_mean_sys"]}')
                print(f'Number of cardiac cycles: {stats["rad_n_cycles"]}')
                print('---------------------')
            if len(rad_peak_data.get('e_py', [])) > 0:
                print(f'Global early peak diastolic {label.upper()} {param}: {stats["rad_peak_e"]}')
                print(f'Global early mean diastolic {label.upper()} {param}: {stats["rad_mean_e"]}')
                print('---------------------')
            if len(rad_peak_data.get('l_py', [])) > 0:
                print(f'Global diastasis peak diastolic {label.upper()} {param}: {stats["rad_peak_l"]}')
                print(f'Global diastasis mean diastolic {label.upper()} {param}: {stats["rad_mean_l"]}')
                print('---------------------')
            if len(rad_peak_data.get('a_py', [])) > 0:
                print(f'Global late peak diastolic {label.upper()} {param}: {stats["rad_peak_a"]}')
                print(f'Global late mean diastolic {label.upper()} {param}: {stats["rad_mean_a"]}')
            print('----------------')
            print('LONGITUDINAL COMPONENT:')
            print('----------------')
            if len(long_peak_data.get('sys_py', [])) > 0:
                print(f'Global peak systolic {label.upper()} {param}: {stats["long_peak_sys"]}')
                print(f'Global mean systolic {label.upper()} {param}: {stats["long_mean_sys"]}')
                print(f'Number of cardiac cycles: {stats["long_n_cycles"]}')
                print('---------------------')
            if len(long_peak_data.get('e_py', [])) > 0:
                print(f'Global early peak diastolic {label.upper()} {param}: {stats["long_peak_e"]}')
                print(f'Global early mean diastolic {label.upper()} {param}: {stats["long_mean_e"]}')
                print('---------------------')
            if len(long_peak_data.get('l_py', [])) > 0:
                print(f'Global diastasis peak diastolic {label.upper()} {param}: {stats["long_peak_l"]}')
                print(f'Global diastasis mean diastolic {label.upper()} {param}: {stats["long_mean_l"]}')
                print('---------------------')
            if len(long_peak_data.get('a_py', [])) > 0:
                print(f'Global late peak diastolic {label.upper()} {param}: {stats["long_peak_a"]}')
                print(f'Global late mean diastolic {label.upper()} {param}: {stats["long_mean_a"]}')
            print('=====================')
        
        safe_makedir(os.path.dirname(save_path))
        fig.tight_layout()
        fig.savefig(save_path)
        if not self.vis_config.show_img:
            plt.close(fig)
        
        # Return statistics tuple if enabled
        if return_statistics:
            result = (stats['rad_peak_sys'], stats['rad_mean_sys'],
                     stats['rad_peak_e'], stats['rad_mean_e'],
                     stats['rad_peak_l'], stats['rad_mean_l'],
                     stats['rad_peak_a'], stats['rad_mean_a'],
                     stats['long_peak_sys'], stats['long_mean_sys'],
                     stats['long_peak_e'], stats['long_mean_e'],
                     stats['long_peak_l'], stats['long_mean_l'],
                     stats['long_peak_a'], stats['long_mean_a'],
                     stats['rad_n_cycles'], stats['long_n_cycles'])
            return result
        
        return fig
    
    def plot_peak_line(self,
                       filt_arr: np.ndarray, frame_times: np.ndarray,
                       param: str, param_unit: str, label: str,
                       filename: str, save_path: str,
                       peak_data: Optional[Dict] = None,
                       hi_arr: Optional[np.ndarray] = None,
                       waveform_data: Optional[np.ndarray] = None,
                       waveform_times: Optional[np.ndarray] = None,
                       sampling_rate: Optional[int] = None,
                       sys_frames: Optional[List[Tuple[int, int]]] = None,
                       dia_frames: Optional[List[Tuple[int, int]]] = None,
                       nframes: int = None,
                       cc_method: str = 'angle',
                       peak_config: Optional[PeakDetectionConfig] = None,
                       show_sysdia: Optional[bool] = None,
                       print_report: Optional[bool] = None,
                       return_statistics: Optional[bool] = None,
                       show_all_peaks: Optional[bool] = None,
                       mode: Optional[str] = None) -> Optional[Union[plt.Figure, Tuple[float, ...]]]:
        """
        Plot single component peak line plot with peak markers.
        
        Args:
            filt_arr: Filtered array (already smoothed) or will be calculated from hi_arr
            frame_times: Time array for frames
            param: Parameter name
            param_unit: Parameter unit string
            label: Label name (e.g., 'rv', 'lv')
            filename: Base filename
            save_path: Full save path
            peak_data: Dictionary with peak data (sys_px, sys_py, e_px, e_py, etc.)
            hi_arr: High percentile array (for internal peak calculation if peak_data not provided)
            waveform_data: Optional waveform data
            waveform_times: Optional waveform time array
            sampling_rate: Sampling rate for waveform
            sys_frames: Systole frame intervals
            dia_frames: Diastole frame intervals
            nframes: Total number of frames
            cc_method: Cardiac cycle detection method
            peak_config: Peak detection configuration
            show_sysdia: Whether to show systole/diastole shading
            print_report: Whether to print statistics report
            return_statistics: Whether to return statistics tuple
            show_all_peaks: Whether to show all detected peaks
            mode: Dataset mode (to check if 'otsu')
        
        Returns:
            matplotlib Figure object, or tuple of 9 statistics values if return_statistics=True
        """
        if os.path.exists(save_path) and not self.proc_config.recalculate:
            print(f'{save_path} already exists, skipping!')
            return None
        
        # Use config values as defaults if parameters are None
        if show_sysdia is None:
            show_sysdia = self.vis_config.show_sysdia_shading
        if print_report is None:
            print_report = self.vis_config.print_report
        if return_statistics is None:
            return_statistics = self.vis_config.return_statistics
        if show_all_peaks is None:
            show_all_peaks = False  # Default to False
        
        if peak_config is None:
            from optical_flow.config import default_peak_detection_config
            peak_config = default_peak_detection_config()
        
        # Calculate peaks internally if not provided
        if peak_data is None:
            if hi_arr is None:
                raise ValueError("Either peak_data or hi_arr must be provided")
            
            # Smooth hi_arr using SpectralSmoother
            smoother = SpectralSmoother(
                smooth_fraction=peak_config.smooth_fraction,
                pad_len=peak_config.pad_len
            )
            smoother.smooth(hi_arr)
            filt_arr = smoother.smooth_data[0]
            
            # Calculate peaks
            if sys_frames is None or dia_frames is None or nframes is None:
                raise ValueError("sys_frames, dia_frames, and nframes must be provided when calculating peaks internally")
            
            peak_data = calculate_single_peaks(
                filt_arr, frame_times, sys_frames, dia_frames, nframes,
                cc_method=cc_method,
                peak_thres=peak_config.peak_thres,
                min_dist=peak_config.min_dist,
                pick_peak_by_subset=peak_config.pick_peak_by_subset,
                show_all_peaks=show_all_peaks
            )
        
        waveform_exists = waveform_data is not None
        # Automatically show waveform subplot if cc_method indicates waveform-based detection
        should_show_waveform = (cc_method in ['ecg', 'ecg_lazy', 'arterial']) or waveform_exists
        show_waveform = should_show_waveform and waveform_exists
        
        # Create figure
        if should_show_waveform:
            fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(8, 6))
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        
        # Plot filtered array as solid line
        ax.plot(frame_times, filt_arr, 'k-', label=param.capitalize())
        
        # Plot peak markers
        sys_color = self.vis_config.systolic_peak_color
        dia_color = self.vis_config.diastolic_peak_color
        
        sys_px = peak_data.get('sys_px', np.array([]))
        sys_py = peak_data.get('sys_py', np.array([]))
        e_px = peak_data.get('e_px', np.array([]))
        e_py = peak_data.get('e_py', np.array([]))
        l_px = peak_data.get('l_px', np.array([]))
        l_py = peak_data.get('l_py', np.array([]))
        a_px = peak_data.get('a_px', np.array([]))
        a_py = peak_data.get('a_py', np.array([]))
        
        # Handle show_all_peaks
        if show_all_peaks and 'all_px' in peak_data and 'all_py' in peak_data:
            all_px = peak_data['all_px']
            all_py = peak_data['all_py']
            if len(all_px) > 0:
                ax.plot(all_px, all_py, 'b+', markersize=self.vis_config.peak_marker_size)
        else:
            # Plot cardiac cycle peaks
            if len(sys_px) > 0:
                ax.plot(sys_px, sys_py,
                       sys_color + self.vis_config.peak_marker_style,
                       markersize=self.vis_config.peak_marker_size)
            if len(e_px) > 0:
                ax.plot(e_px, e_py,
                       dia_color + self.vis_config.peak_marker_style,
                       markersize=self.vis_config.peak_marker_size)
            if len(l_px) > 0:
                ax.plot(l_px, l_py,
                       dia_color + self.vis_config.peak_marker_style,
                       markersize=self.vis_config.peak_marker_size)
            if len(a_px) > 0:
                ax.plot(a_px, a_py,
                       dia_color + self.vis_config.peak_marker_style,
                       markersize=self.vis_config.peak_marker_size)
        
        # Annotate peaks if enabled (only if not show_all_peaks)
        if self.vis_config.show_peak_annotations and not show_all_peaks:
            if len(sys_px) > 0:
                annotate_peaks(ax, sys_px, sys_py,
                              color=sys_color, offset=(self.vis_config.peak_annotation_offset[0],
                                                       -self.vis_config.peak_annotation_offset[1]),
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(e_px) > 0:
                annotate_peaks(ax, e_px, e_py,
                              color=dia_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(l_px) > 0:
                annotate_peaks(ax, l_px, l_py,
                              color=dia_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
            if len(a_px) > 0:
                annotate_peaks(ax, a_px, a_py,
                              color=dia_color, offset=self.vis_config.peak_annotation_offset,
                              fontsize=self.vis_config.peak_annotation_fontsize)
        
        # Set labels and title (matching legacy format)
        ax.set_title(label.upper() + ' ' + param.capitalize() + ' vs Time')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        
        # Add waveform overlay if provided
        if should_show_waveform:
            if waveform_data is not None:
                if 'ecg' in cc_method:
                    if waveform_times is None:
                        waveform_times = np.arange(waveform_data.size) * (1000 / sampling_rate)
                    waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
                    ax2.plot(waveform_times, waveform_data)
                    ax2.set_ylabel('Voltage (mV)')
                elif cc_method == 'arterial':
                    if waveform_times is None:
                        waveform_times = np.arange(waveform_data.size) * (1000 / sampling_rate)
                    ax2.plot(waveform_times, waveform_data)
                    ax2.set_ylabel('Pressure (mmHg)')
                ax2.set_xlabel('Time (ms)')
            else:
                # Waveform subplot created but no data available
                ax2.text(0.5, 0.5, 'Waveform data not available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
                if 'ecg' in cc_method:
                    ax2.set_ylabel('Voltage (mV)')
                elif cc_method == 'arterial':
                    ax2.set_ylabel('Pressure (mmHg)')
                ax2.set_xlabel('Time (ms)')
        
        # Add systole/diastole shading
        sys_label = None
        dia_label = None
        if show_sysdia and mode != 'otsu':
            true_sys = peak_data.get('true_sys', [])
            true_dia = peak_data.get('true_dia', [])
            
            if len(true_sys) > 0:
                counter = 0
                for start, stop in true_sys:
                    if stop >= nframes:
                        stop = nframes - 1  # sanity check
                    if counter == 0:
                        sys_label = ax.axvspan(frame_times[start], frame_times[stop],
                                             facecolor='0.8', alpha=0.5, label='Systole')
                    else:
                        ax.axvspan(frame_times[start], frame_times[stop],
                                 facecolor='0.8', alpha=0.5)
                    counter += 1
            
            if len(true_dia) > 0:
                counter = 0
                for start, stop in true_dia:
                    if stop >= nframes:
                        stop = nframes - 1  # sanity check
                    if counter == 0:
                        dia_label = ax.axvspan(frame_times[start], frame_times[stop],
                                             facecolor='0.8', alpha=0.25, label='Diastole')
                    else:
                        ax.axvspan(frame_times[start], frame_times[stop],
                                 facecolor='0.8', alpha=0.25)
                    counter += 1
        
        # Add legend if shading enabled
        if show_sysdia and mode != 'otsu' and (sys_label is not None or dia_label is not None):
            ax.legend(loc='lower right')
        
        # Calculate and print/return statistics
        stats = None
        if print_report or return_statistics:
            stats = self._calculate_single_peak_statistics(peak_data)
            
            if print_report:
                print('=====================')
                if stats['n_cycles'] > 0:
                    print(f'Global peak systolic {label.upper()} {param}: {stats["peak_sys"]:.2f}')
                    print(f'Global mean systolic {label.upper()} {param}: {stats["mean_sys"]:.2f}')
                    print(f'Number of cardiac cycles: {stats["n_cycles"]}')
                    print('---------------------')
                if stats['peak_e'] > 0:
                    print(f'Global peak early diastolic {label.upper()} {param}: {stats["peak_e"]:.2f}')
                    print(f'Global mean early diastolic {label.upper()} {param}: {stats["mean_e"]:.2f}')
                    print('---------------------')
                if stats['peak_l'] > 0:
                    print(f'Global peak diastasis diastolic {label.upper()} {param}: {stats["peak_l"]:.2f}')
                    print(f'Global mean diastasis diastolic {label.upper()} {param}: {stats["mean_l"]:.2f}')
                    print('---------------------')
                if stats['peak_a'] > 0:
                    print(f'Global peak late diastolic {label.upper()} {param}: {stats["peak_a"]:.2f}')
                    print(f'Global mean late diastolic {label.upper()} {param}: {stats["mean_a"]:.2f}')
                print('=====================')
        
        # Always save the file
        safe_makedir(os.path.dirname(save_path))
        fig.tight_layout()
        fig.savefig(save_path)
        if not self.vis_config.show_img:
            plt.close(fig)
        
        # Return statistics if requested
        if return_statistics and stats is not None:
            if len(sys_py) == 0:
                print(f'ERROR not complete cardiac cycle: systolic cycles={len(sys_py)}')
            result_tuple = (
                stats['peak_sys'], stats['mean_sys'],
                stats['peak_e'], stats['mean_e'],
                stats['peak_l'], stats['mean_l'],
                stats['peak_a'], stats['mean_a'],
                stats['n_cycles']
            )
            return result_tuple
        
        return fig
    
    @staticmethod
    def _overlay3(dcm_arr: np.ndarray, rad_arr: np.ndarray, long_arr: np.ndarray) -> np.ndarray:
        """Overlay radial and longitudinal components on DICOM array."""
        x1 = np.concatenate([dcm_arr, dcm_arr], axis=2)
        x2 = np.concatenate([rad_arr, long_arr], axis=2)
        x = (0.5 * (x1 / np.max(x1)) + 0.5 * (x2 / np.max(x2))) * 255
        return x.astype(np.uint8)


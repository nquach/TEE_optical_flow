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

from optical_flow_utils import safe_makedir, fix_ecg
from optical_flow.config import VisualizationConfig, ProcessingConfig, AnalysisConfig
from optical_flow.plotting_utils import (
    add_systole_diastole_shading, plot_waveform_with_shading,
    create_heatmap_figure, setup_colorbar
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
        
        plt1 = ax1.pcolormesh(frame_times, ygrid_rad, rad_mag_freq_arr.T,
                              norm=rad_norm, cmap=self.vis_config.colormap_mag)
        ax1.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        ax1.set_title('Radial ' + param.capitalize() + ' vs Time (ms)')
        
        plt2 = ax2.pcolormesh(frame_times, ygrid_long, long_mag_freq_arr.T,
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
        
        plt1 = ax1.pcolormesh(frame_times, ygrid_mag, mag_arr.T,
                              norm=mag_norm, cmap=self.vis_config.colormap_mag)
        ax1.set_ylabel(param.capitalize() + ' (' + param_unit + ')')
        ax1.set_title('Magnitude of ' + param.capitalize() + ' vs Time (ms)')
        
        plt2 = ax2.pcolormesh(frame_times, ygrid_ang, ang_arr.T,
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
    
    @staticmethod
    def _overlay3(dcm_arr: np.ndarray, rad_arr: np.ndarray, long_arr: np.ndarray) -> np.ndarray:
        """Overlay radial and longitudinal components on DICOM array."""
        x1 = np.concatenate([dcm_arr, dcm_arr], axis=2)
        x2 = np.concatenate([rad_arr, long_arr], axis=2)
        x = (0.5 * (x1 / np.max(x1)) + 0.5 * (x2 / np.max(x2))) * 255
        return x.astype(np.uint8)


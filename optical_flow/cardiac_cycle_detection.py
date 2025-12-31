"""
Cardiac cycle detection module.

This module provides classes for detecting cardiac cycle phases (systole/diastole)
using various methods: angle-based, area-based, ECG, arterial pressure, and DICOM metadata.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from tsmoothie.smoother import SpectralSmoother
import neurokit2 as nk
import peakutils
import skimage.measure
from scipy.stats import mode
import cv2

from optical_flow.optical_flow_utils import safe_makedir, find_start_stop, timeinterval2index, frame2time, index_smallest_positive
from optical_flow.config import CardiacCycleConfig, VisualizationConfig, ProcessingConfig


class CardiacCycleDetector(ABC):
    """Base class for cardiac cycle detection methods."""
    
    def __init__(self, cc_config: Optional[CardiacCycleConfig] = None,
                 vis_config: Optional[VisualizationConfig] = None,
                 proc_config: Optional[ProcessingConfig] = None):
        self.cc_config = cc_config or CardiacCycleConfig()
        self.vis_config = vis_config or VisualizationConfig()
        self.proc_config = proc_config or ProcessingConfig()
    
    @abstractmethod
    def detect(self, ds, **kwargs) -> Tuple[List, List]:
        """
        Detect cardiac cycle phases.
        
        Returns:
            Tuple of (sys_frames, dia_frames) where each is a list of [start, stop] frame pairs
        """
        pass
    
    def _should_recalculate(self, ds) -> bool:
        """Check if recalculation is needed."""
        return self.proc_config.recalculate or not ds.CARDIACCYCLE_CALCULATED
    
    def _plot_cardiac_cycle(self, ds, signal_data, signal_times, sys_intervals, dia_intervals,
                           xlabel: str, ylabel: str, title: str, filename_suffix: str):
        """Common plotting logic for cardiac cycle visualization."""
        if not (self.vis_config.save_cc_plot or self.vis_config.show_plot):
            return
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(signal_times, signal_data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        for start, stop in sys_intervals:
            ax.axvspan(signal_times[start] if isinstance(start, int) else start,
                      signal_times[stop] if isinstance(stop, int) else stop,
                      facecolor='0.8', alpha=0.5)
        for start, stop in dia_intervals:
            ax.axvspan(signal_times[start] if isinstance(start, int) else start,
                      signal_times[stop] if isinstance(stop, int) else stop,
                      facecolor='0.9', alpha=0.25)
        
        if self.vis_config.save_dir is not None and self.vis_config.save_cc_plot:
            safe_makedir(self.vis_config.save_dir)
            save_name = ds.filename + filename_suffix
            save_path = os.path.join(self.vis_config.save_dir, save_name)
            fig.savefig(save_path)
        elif self.vis_config.save_cc_plot:
            print('ERROR save_dir cannot be None if save_cc_plot flag is True!')
        
        if not self.vis_config.show_plot:
            plt.close(fig)
    
    def _update_dataset(self, ds, sys_frames: List, dia_frames: List):
        """Update dataset with detected frames."""
        ds.sys_frames = sys_frames
        ds.dia_frames = dia_frames
        ds.CARDIACCYCLE_CALCULATED = True


class AngleDetector(CardiacCycleDetector):
    """Detect cardiac cycle using optical flow angle analysis."""
    
    def detect(self, ds, param: str, label: str) -> Tuple[List, List]:
        """Detect systole/diastole using angle-based method."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by angle!')
            return ds.sys_frames, ds.dia_frames
        
        if self.proc_config.verbose:
            print('Calculating systolic and diastolic frame labels by angle...')
        
        arr = ds.get_masked_arr(param, label)
        ang_list = []
        for i in range(ds.nframes):
            flow = np.squeeze(arr[i, :, :, :])
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            ang_list.append(ang)
        
        ang_arr = np.stack(ang_list)
        mode_list = []
        for i in range(ds.nframes):
            ang_frame = np.round(ang_arr[i, :, :], decimals=2)
            flat = np.ravel(ang_frame)
            flat_nonzero = flat[flat != 0]
            res = mode(flat_nonzero)
            mode_list.append(res.mode)
        
        ang_mode_arr = np.asarray(mode_list)
        smoother_ang = SpectralSmoother(
            smooth_fraction=self.cc_config.smooth_fraction,
            pad_len=self.cc_config.pad_len
        )
        smoother_ang.smooth(ang_mode_arr)
        filt_ang_arr = smoother_ang.smooth_data[0]
        
        up_frames = np.squeeze(np.argwhere(filt_ang_arr < np.pi))
        down_frames = np.squeeze(np.argwhere(filt_ang_arr >= np.pi))
        sys_frames = find_start_stop(up_frames)
        dia_frames = find_start_stop(down_frames)
        
        if self.proc_config.verbose:
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        # Plot if requested
        if self.vis_config.save_cc_plot or self.vis_config.show_plot:
            frame_list = range(ds.nframes)
            self._plot_cardiac_cycle(
                ds, ang_mode_arr, frame_list, sys_frames, dia_frames,
                'Frame', 'Angle Mode', 'Angle-based Cardiac Cycle Detection',
                f'_{label}_{param}_sysdia_angle_diagnostic_plot.png'
            )
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


class AreaDetector(CardiacCycleDetector):
    """Detect cardiac cycle using mask area analysis."""
    
    def detect(self, ds, label: str) -> Tuple[List, List]:
        """Detect systole/diastole using area-based method."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by area!')
            return ds.sys_frames, ds.dia_frames
        
        if self.proc_config.verbose:
            print('Calculating systolic and diastolic frame labels by area...')
        
        mask_arr = ds.mask_ds_dict[label][()]
        area_list = []
        for i in range(ds.nframes):
            mask = mask_arr[i, :, :, 0]
            l = skimage.measure.label(mask)
            props = skimage.measure.regionprops(l, mask)
            if len(props) == 0:
                print('Error no mask detected!')
                if len(area_list) > 0:
                    area_list.append(area_list[-1])
                else:
                    area_list.append(0)
            else:
                area_list.append(props[0].area)
        
        smoother = SpectralSmoother(
            smooth_fraction=self.cc_config.smooth_fraction,
            pad_len=self.cc_config.pad_len
        )
        smoother.smooth(area_list)
        filt_area_list = smoother.smooth_data[0]
        filt_area_baseline = peakutils.baseline(filt_area_list)
        filt_area_list = np.asarray(filt_area_list) - np.asarray(filt_area_baseline)
        
        peak_i = sorted(list(peakutils.peak.indexes(
            filt_area_list, thres=self.cc_config.dia_thres, min_dist=5
        )))
        val_i = sorted(list(peakutils.peak.indexes(
            filt_area_list * -1, thres=self.cc_config.sys_thres, min_dist=5
        )))
        
        # Filter out double systolic peaks
        del_frame_list = []
        for i in range(len(val_i) - 1):
            v1 = val_i[i]
            v2 = val_i[i + 1]
            dia_frames = [p for p in peak_i if v1 < p < v2]
            if len(dia_frames) == 0:
                del_frame = np.argwhere(
                    filt_area_list == max(filt_area_list[v1], filt_area_list[v2])
                )
                if len(del_frame) > 0:
                    del_frame_list.append(val_i.index(del_frame[0][0]))
        
        for i in sorted(del_frame_list, reverse=True):
            if i < len(val_i):
                del val_i[i]
        
        # Calculate systolic and diastolic frame ranges
        val_i = sorted(val_i, reverse=True)
        peak_i = sorted(peak_i, reverse=True)
        sys_frames = []
        dia_frames = []
        
        for i in range(len(val_i)):
            end_sys = val_i[i]
            dia_distance = [(end_sys - j) for j in peak_i]
            end_dia_index = index_smallest_positive(dia_distance)
            if end_dia_index is None:
                break
            end_dia = peak_i[end_dia_index]
            sys_frames.append((end_dia, end_sys))
            if (i + 1) < len(val_i):
                dia_frames.append((val_i[i + 1], end_dia))
        
        if self.proc_config.verbose:
            print('peaks:', peak_i)
            print('vals:', val_i)
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        # Plot if requested
        if self.vis_config.save_cc_plot or self.vis_config.show_plot:
            frame_list = list(range(ds.nframes))
            self._plot_cardiac_cycle(
                ds, area_list, frame_list, sys_frames, dia_frames,
                'Frame', 'Area', 'Area-based Cardiac Cycle Detection',
                f'_{label}_area_plot.png'
            )
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


class RTimeDetector(CardiacCycleDetector):
    """Detect cardiac cycle using DICOM R-wave time metadata."""
    
    def detect(self, ds) -> Tuple[List, List]:
        """Detect systole/diastole using R-wave time metadata."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by DICOM metadata!')
            return ds.sys_frames, ds.dia_frames
        
        if not ds.RTimePresent:
            print('ERROR no R Wave Time Vector metadata present for automatic cardiac cycle calculation!')
            return [], []
        
        if ds.RWaveTimes.size < 2:
            print('ERROR not enough R waves recorded to determine at least 1 cardiac cycle!')
            return [], []
        
        frame_times = np.arange(ds.nframes) * (1000 / ds.frame_rate)
        sys_times = []
        dia_times = []
        
        for i in range(ds.RWaveTimes.size - 1):
            r1 = ds.RWaveTimes[i]
            r2 = ds.RWaveTimes[i + 1]
            rr_time = r2 - r1
            sys_end = r1 + rr_time * self.cc_config.rr_sys_ratio
            sys_times.append([r1, sys_end])
            dia_times.append([sys_end, r2])
        
        sys_frames = timeinterval2index(sys_times, frame_times)
        dia_frames = timeinterval2index(dia_times, frame_times)
        
        if self.proc_config.verbose:
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


class ECGLazyDetector(CardiacCycleDetector):
    """Detect cardiac cycle using ECG with lazy T-wave detection."""
    
    def detect(self, ds, ecg_arr: np.ndarray, sampling_rate: int = 500) -> Tuple[List, List]:
        """Detect systole/diastole using ECG lazy method."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by ecg_lazy!')
            return ds.sys_frames, ds.dia_frames
        
        sys_i = []
        dia_i = []
        ecg = nk.ecg_clean(ecg_arr, sampling_rate=sampling_rate, method='vg')
        smoother_ecg = SpectralSmoother(
            smooth_fraction=self.cc_config.smooth_fraction,
            pad_len=self.cc_config.pad_len
        )
        smoother_ecg.smooth(ecg)
        filt_ecg = np.squeeze(smoother_ecg.smooth_data[0])
        n_elem = filt_ecg.shape[0]
        
        # Get R-peaks location
        _, rpeaks = nk.ecg_peaks(
            filt_ecg, sampling_rate=sampling_rate,
            method='khamis2016', correct_artifacts=True, show=False
        )
        r_i = rpeaks['ECG_R_Peaks']
        
        for i in range(len(r_i) - 1):
            r1 = r_i[i].item()
            r2 = r_i[i + 1].item()
            rr_time = r2 - r1
            sys_end = r1 + rr_time * self.cc_config.rr_sys_ratio
            sys_i.append([r1, sys_end])
            dia_i.append([sys_end, r2])
        
        frame_times = np.arange(ds.nframes) * (1 / ds.frame_rate)
        sys_frames = timeinterval2index(frame2time(sys_i, sampling_rate), frame_times)
        dia_frames = timeinterval2index(frame2time(dia_i, sampling_rate), frame_times)
        
        sys_frames = [
            [s[0], np.min([s[1] + self.cc_config.sys_extension, ds.nframes - 1])]
            for s in sys_frames
        ]
        
        if self.proc_config.verbose:
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        # Plot if requested
        if self.vis_config.save_cc_plot or self.vis_config.show_plot:
            ecg_times = np.arange(n_elem) * (1000 / sampling_rate)
            self._plot_cardiac_cycle(
                ds, filt_ecg, ecg_times, sys_i, dia_i,
                'Time (msec)', 'Voltage (mV)', 'ECG Lazy Cardiac Cycle Detection',
                '_sysdia_ecg_diagnostic_plot.png'
            )
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


class ECGDetector(CardiacCycleDetector):
    """Detect cardiac cycle using ECG with T-wave peak detection."""
    
    def detect(self, ds, ecg_arr: np.ndarray, sampling_rate: int = 500) -> Tuple[List, List]:
        """Detect systole/diastole using ECG T-wave method."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by ecg!')
            return ds.sys_frames, ds.dia_frames
        
        ecg = nk.ecg_clean(ecg_arr, sampling_rate=sampling_rate, method='vg')
        smoother_ecg = SpectralSmoother(
            smooth_fraction=self.cc_config.smooth_fraction,
            pad_len=self.cc_config.pad_len
        )
        smoother_ecg.smooth(ecg)
        filt_ecg = np.squeeze(smoother_ecg.smooth_data[0])
        
        # Get R-peaks location
        _, rpeaks = nk.ecg_peaks(
            filt_ecg, sampling_rate=sampling_rate,
            method='khamis2016', correct_artifacts=True, show=False
        )
        r_i = rpeaks['ECG_R_Peaks']
        
        sys_i = []
        for idx in range(len(r_i) - 1):
            R_start = r_i[idx].item()
            R_stop = r_i[idx + 1].item()
            delta_R_i = R_stop - R_start
            search_start = int(np.round(
                delta_R_i * self.cc_config.rr_search_range[0] + R_start
            ))
            search_end = int(np.round(
                delta_R_i * self.cc_config.rr_search_range[1] + R_start
            ))
            segment = filt_ecg[search_start:search_end]
            candidate_i = peakutils.peak.indexes(
                segment, thres=self.cc_config.t_peak_thres,
                min_dist=self.cc_config.t_min_dist
            ) + search_start
            
            if len(candidate_i) > 0:
                candidate_y = [filt_ecg[i] for i in candidate_i]
                max_idx = np.argmax(candidate_y)
                sys_i.append([R_start, int(candidate_i[max_idx])])
        
        n_elem = filt_ecg.shape[0]
        dia_i = []
        if len(sys_i) > 0 and sys_i[-1][1] < r_i[-1]:
            dia_i.append([sys_i[-1][1], r_i[-1].item() - 1])
        for i in range(len(sys_i) - 1):
            start1, stop1 = sys_i[i]
            start2, stop2 = sys_i[i + 1]
            dia_i.append([stop1, start2])
        
        frame_times = np.arange(ds.nframes) * (1 / ds.frame_rate)
        sys_frames = timeinterval2index(frame2time(sys_i, sampling_rate), frame_times)
        dia_frames = timeinterval2index(frame2time(dia_i, sampling_rate), frame_times)
        
        if self.proc_config.verbose:
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        # Plot if requested
        if self.vis_config.save_cc_plot or self.vis_config.show_plot:
            ecg_times = np.arange(n_elem) * (1000 / sampling_rate)
            self._plot_cardiac_cycle(
                ds, filt_ecg, ecg_times, sys_i, dia_i,
                'Time (msec)', 'Voltage (mV)', 'ECG Cardiac Cycle Detection',
                '_sysdia_ecg_diagnostic_plot.png'
            )
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


class ArterialDetector(CardiacCycleDetector):
    """Detect cardiac cycle using arterial pressure waveform."""
    
    def detect(self, ds, art_arr: np.ndarray, sampling_rate: int = 125) -> Tuple[List, List]:
        """Detect systole/diastole using arterial pressure."""
        if not self._should_recalculate(ds):
            if self.proc_config.verbose:
                print('Cardiac cycle info calculated already! skipping calculation by art!')
            return ds.sys_frames, ds.dia_frames
        
        smoother_art = SpectralSmoother(
            smooth_fraction=self.cc_config.smooth_fraction,
            pad_len=self.cc_config.pad_len
        )
        smoother_art.smooth(art_arr)
        filt_art = np.squeeze(smoother_art.smooth_data[0])
        
        lows_i = peakutils.peak.indexes(
            filt_art * -1, thres=self.cc_config.low_peak_thres,
            min_dist=self.cc_config.low_min_dist
        ) - self.cc_config.sys_upstroke_offset
        lows_i[lows_i < 0] = 0
        
        highs_i = []
        sys_i = []
        for idx in range(len(lows_i) - 1):
            low_start = lows_i[idx].item()
            low_stop = lows_i[idx + 1].item()
            segment = filt_art[low_start:low_stop]
            candidate_i = peakutils.peak.indexes(
                segment, thres=self.cc_config.high_peak_thres,
                min_dist=self.cc_config.high_min_dist
            ) + low_start
            
            if len(candidate_i) > 0:
                candidate_y = [filt_art[i] for i in candidate_i]
                max_idx = np.argmax(candidate_y)
                highs_i.append(int(candidate_i[max_idx]))
                delta_lowhigh = int(candidate_i[max_idx]) - low_start
                sys_stop = low_start + int(np.round(
                    self.cc_config.sys_upstroke_multiplier * delta_lowhigh
                ))
                sys_i.append([low_start, sys_stop])
        
        n_elem = art_arr.size
        dia_i = []
        if len(sys_i) > 0 and sys_i[-1][1] < lows_i[-1]:
            dia_i.append([sys_i[-1][1], lows_i[-1].item() - 1])
        for i in range(len(sys_i) - 1):
            start1, stop1 = sys_i[i]
            start2, stop2 = sys_i[i + 1]
            dia_i.append([stop1, start2])
        
        frame_times = np.arange(ds.nframes) * (1 / ds.frame_rate)
        sys_frames = timeinterval2index(frame2time(sys_i, sampling_rate), frame_times)
        dia_frames = timeinterval2index(frame2time(dia_i, sampling_rate), frame_times)
        
        if self.proc_config.verbose:
            print('systole frames:', sys_frames)
            print('diastole frames:', dia_frames)
        
        # Plot if requested
        if self.vis_config.save_cc_plot or self.vis_config.show_plot:
            art_times = np.arange(n_elem) * (1000 / sampling_rate)
            self._plot_cardiac_cycle(
                ds, filt_art, art_times, sys_i, dia_i,
                'Time (msec)', 'Pressure (mmHg)', 'Arterial Pressure Cardiac Cycle Detection',
                '_sysdia_art_diagnostic_plot.png'
            )
        
        self._update_dataset(ds, sys_frames, dia_frames)
        return sys_frames, dia_frames


def create_detector(method: str, cc_config: Optional[CardiacCycleConfig] = None,
                   vis_config: Optional[VisualizationConfig] = None,
                   proc_config: Optional[ProcessingConfig] = None) -> CardiacCycleDetector:
    """
    Factory function to create appropriate detector based on method name.
    
    Args:
        method: Detection method ('angle', 'area', 'ecg', 'ecg_lazy', 'metadata', 'arterial')
        cc_config: Cardiac cycle configuration
        vis_config: Visualization configuration
        proc_config: Processing configuration
    
    Returns:
        Appropriate CardiacCycleDetector instance
    """
    method_map = {
        'angle': AngleDetector,
        'area': AreaDetector,
        'ecg': ECGDetector,
        'ecg_lazy': ECGLazyDetector,
        'metadata': RTimeDetector,
        'arterial': ArterialDetector,
    }
    
    detector_class = method_map.get(method)
    if detector_class is None:
        raise ValueError(f"Unknown detection method: {method}. "
                        f"Must be one of {list(method_map.keys())}")
    
    return detector_class(cc_config, vis_config, proc_config)


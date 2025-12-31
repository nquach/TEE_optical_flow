"""
Peak detection module for optical flow analysis.

This module provides classes and functions for detecting peaks in
radial/longitudinal component data, including systolic and diastolic peaks.
"""

import numpy as np
import peakutils
from tsmoothie.smoother import SpectralSmoother
from typing import List, Tuple, Optional

from optical_flow.config import PeakDetectionConfig, CardiacCycleConfig


class PeakDetector:
    """Detects peaks in radial/longitudinal component data."""
    
    def __init__(self, peak_config: Optional[PeakDetectionConfig] = None,
                 cc_config: Optional[CardiacCycleConfig] = None):
        self.peak_config = peak_config or PeakDetectionConfig()
        self.cc_config = cc_config or CardiacCycleConfig()
    
    def detect_systolic_peaks(self, filt_lo: np.ndarray, sys_frames: List[Tuple[int, int]],
                             lo_peaks_i: np.ndarray) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Detect systolic peaks in low component array.
        
        Args:
            filt_lo: Filtered low component array
            sys_frames: List of (start, stop) frame pairs for systole
            lo_peaks_i: Pre-computed peak indices in filt_lo
        
        Returns:
            Tuple of (sys_i, true_sys) where sys_i is list of peak indices
        """
        sys_i = []
        true_sys = []
        
        for start, stop in sys_frames:
            if self.peak_config.pick_peak_by_subset:
                candidate_i = peakutils.peak.indexes(
                    filt_lo[start:stop + 1] * -1,
                    thres=self.peak_config.peak_thres,
                    min_dist=self.peak_config.min_dist
                ) + start
            else:
                candidate_i = [k for k in lo_peaks_i if ((k >= start) and (k <= stop))]
            
            if len(candidate_i) > 0:
                candidate_y = [filt_lo[i] for i in candidate_i]
                index = np.argmin(candidate_y)
                sys_i.append(candidate_i[index])
                true_sys.append([start, stop])
            else:
                print('Warning no systolic peak found! Using max value')
                sys_i.append(np.argmin(filt_lo[start:stop]) + start)
        
        return sys_i, true_sys
    
    def detect_diastolic_peaks(self, filt_hi: np.ndarray, dia_frames: List[Tuple[int, int]],
                              hi_peaks_i: np.ndarray, nframes: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Detect diastolic peaks (e', l', a') in high component array.
        
        Args:
            filt_hi: Filtered high component array
            dia_frames: List of (start, stop) frame pairs for diastole
            hi_peaks_i: Pre-computed peak indices in filt_hi
            nframes: Total number of frames
        
        Returns:
            Tuple of (e_i, l_i, a_i) where each is a list of peak indices
        """
        e_i = []
        l_i = []
        a_i = []
        
        for start, stop in dia_frames:
            e_start = int(start)
            e_stop = int(start + np.floor((stop - start) / 3))
            l_start = int(e_stop + 1)
            l_stop = int(l_start + np.floor((stop - start) / 3))
            a_start = int(l_stop + 1)
            a_stop = int(stop + 1)
            
            # Find e' peak
            if self.peak_config.pick_peak_by_subset:
                e_candidate_i = peakutils.peak.indexes(
                    filt_hi[e_start:e_stop + 1],
                    thres=self.peak_config.peak_thres,
                    min_dist=self.peak_config.min_dist
                ) + e_start
                l_candidate_i = peakutils.peak.indexes(
                    filt_hi[l_start:l_stop + 1],
                    thres=self.peak_config.peak_thres,
                    min_dist=self.peak_config.min_dist
                ) + l_start
                a_candidate_i = peakutils.peak.indexes(
                    filt_hi[a_start:a_stop + 1],
                    thres=self.peak_config.peak_thres,
                    min_dist=self.peak_config.min_dist
                ) + a_start
            else:
                e_candidate_i = [k for k in hi_peaks_i if ((k >= e_start) and (k <= e_stop))]
                l_candidate_i = [k for k in hi_peaks_i if ((k >= l_start) and (k <= l_stop))]
                a_candidate_i = [k for k in hi_peaks_i if ((k >= a_start) and (k <= a_stop))]
            
            # Process e' peak
            if len(e_candidate_i) > 0:
                e_candidate_y = [filt_hi[i] for i in e_candidate_i]
                e_index = np.argmax(e_candidate_y)
                e_i.append(e_candidate_i[e_index])
            else:
                print('Warning no e\' peak found! Using max value')
                e_i.append(np.argmax(filt_hi[e_start:e_stop]) + e_start)
            
            # Process l' peak
            if len(l_candidate_i) > 0:
                l_candidate_y = [filt_hi[i] for i in l_candidate_i]
                l_index = np.argmax(l_candidate_y)
                l_i.append(l_candidate_i[l_index])
            else:
                print('Warning no l\' peak found! Using max value')
                l_i.append(np.argmax(filt_hi[l_start:l_stop]) + l_start)
            
            # Process a' peak
            if len(a_candidate_i) > 0:
                a_candidate_y = [filt_hi[i] for i in a_candidate_i]
                a_index = np.argmax(a_candidate_y)
                a_i.append(a_candidate_i[a_index])
            else:
                print('Warning no a\' peak found! Using max value')
                a_i.append(np.argmax(filt_hi[a_start:a_stop]) + a_start)
        
        return e_i, l_i, a_i


def calculate_radlong_peaks(hi_arr: np.ndarray, lo_arr: np.ndarray, frame_times: np.ndarray,
                           sys_frames: List[Tuple[int, int]], dia_frames: List[Tuple[int, int]],
                           nframes: int, cc_method: str = 'angle',
                           smooth_fraction: float = 0.3, pad_len: int = 20,
                           peak_thres: float = 0.5, min_dist: int = 5,
                           pick_peak_by_subset: bool = False) -> dict:
    """
    Calculate peaks for radial/longitudinal components.
    
    Args:
        hi_arr: High percentile array
        lo_arr: Low percentile array
        frame_times: Time array for frames
        sys_frames: Systole frame intervals
        dia_frames: Diastole frame intervals
        nframes: Total number of frames
        cc_method: Cardiac cycle detection method
        smooth_fraction: Smoothing fraction
        pad_len: Padding length
        peak_thres: Peak detection threshold
        min_dist: Minimum distance between peaks
        pick_peak_by_subset: Whether to pick peaks from subset
    
    Returns:
        Dictionary with filtered arrays, frame intervals, and peak coordinates
    """
    lo_smoother = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
    hi_smoother = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
    lo_smoother.smooth(lo_arr)
    hi_smoother.smooth(hi_arr)
    filt_lo = lo_smoother.smooth_data[0]
    filt_hi = hi_smoother.smooth_data[0]
    
    hi_peaks_i = peakutils.peak.indexes(filt_hi, thres=peak_thres, min_dist=min_dist)
    lo_peaks_i = peakutils.peak.indexes(filt_lo * -1, thres=peak_thres, min_dist=min_dist)
    
    # Determine true_sys and true_dia based on method
    if cc_method == 'angle':
        true_dia = []
        true_sys = sys_frames
        if len(true_sys) > 0:
            if true_sys[0][0] > 1:
                true_dia.append([0, true_sys[0][0] - 1])
            if true_sys[-1][1] < (nframes - 2):
                true_dia.append([true_sys[-1][1], nframes - 1])
            for i in range(len(true_sys) - 1):
                start1, stop1 = true_sys[i]
                start2, stop2 = true_sys[i + 1]
                true_dia.append([stop1, start2])
    else:
        true_dia = dia_frames
        true_sys = sys_frames
    
    # Detect peaks
    peak_config = PeakDetectionConfig(
        peak_thres=peak_thres,
        min_dist=min_dist,
        pick_peak_by_subset=pick_peak_by_subset
    )
    detector = PeakDetector(peak_config=peak_config)
    
    sys_i, true_sys_updated = detector.detect_systolic_peaks(filt_lo, true_sys, lo_peaks_i)
    e_i, l_i, a_i = detector.detect_diastolic_peaks(filt_hi, true_dia, hi_peaks_i, nframes)
    
    # Prepare return values
    sys_px = frame_times[sys_i]
    sys_py = filt_lo[sys_i]
    e_px = frame_times[e_i]
    e_py = filt_hi[e_i]
    l_px = frame_times[l_i]
    l_py = filt_hi[l_i]
    a_px = frame_times[a_i]
    a_py = filt_hi[a_i]
    
    return {
        'filt_hi': filt_hi,
        'filt_lo': filt_lo,
        'true_sys': true_sys_updated,
        'true_dia': true_dia,
        'sys_px': sys_px,
        'sys_py': sys_py,
        'e_px': e_px,
        'e_py': e_py,
        'l_px': l_px,
        'l_py': l_py,
        'a_px': a_px,
        'a_py': a_py
    }


def calculate_single_peaks(filt_arr: np.ndarray, frame_times: np.ndarray,
                           sys_frames: List[Tuple[int, int]], dia_frames: List[Tuple[int, int]],
                           nframes: int, cc_method: str = 'angle',
                           peak_thres: float = 0.2, min_dist: int = 5,
                           pick_peak_by_subset: bool = False,
                           show_all_peaks: bool = False) -> dict:
    """
    Calculate peaks for single component (non-radial/longitudinal).
    
    Args:
        filt_arr: Filtered/smoothed array
        frame_times: Time array for frames
        sys_frames: Systole frame intervals
        dia_frames: Diastole frame intervals
        nframes: Total number of frames
        cc_method: Cardiac cycle detection method
        peak_thres: Peak detection threshold
        min_dist: Minimum distance between peaks
        pick_peak_by_subset: Whether to pick peaks from subset
        show_all_peaks: Whether to return all peaks or just cardiac cycle peaks
    
    Returns:
        Dictionary with:
        - 'filt_arr': filtered array
        - 'true_sys': systole frame intervals
        - 'true_dia': diastole frame intervals
        - 'sys_px', 'sys_py': systolic peak coordinates
        - 'e_px', 'e_py': e' peak coordinates
        - 'l_px', 'l_py': l' peak coordinates
        - 'a_px', 'a_py': a' peak coordinates
        - 'all_px', 'all_py': all peaks (if show_all_peaks=True)
    """
    # Find all peaks
    peaks_i = peakutils.peak.indexes(filt_arr, thres=peak_thres, min_dist=min_dist)
    
    # Detect systolic peaks
    sys_i = []
    true_sys = []
    for start, stop in sys_frames:
        if pick_peak_by_subset:
            candidate_i = peakutils.peak.indexes(
                filt_arr[start:stop + 1], thres=peak_thres, min_dist=min_dist
            ) + start
        else:
            candidate_i = [k for k in peaks_i if ((k >= start) and (k <= stop))]
        
        if len(candidate_i) > 0:
            candidate_y = [filt_arr[i] for i in candidate_i]
            max_idx = np.argmax(candidate_y)
            sys_i.append(candidate_i[max_idx])
            true_sys.append([start, stop])
        else:
            print('Warning no sys peak found! Using max value')
            sys_i.append(np.argmax(filt_arr[start:stop]) + start)
    
    # Determine true_sys and true_dia based on method
    if cc_method == 'angle':
        true_dia = []
        if len(true_sys) > 0:
            if true_sys[0][0] > 1:
                true_dia.append([0, true_sys[0][0] - 1])
            if true_sys[-1][1] < (nframes - 2):
                true_dia.append([true_sys[-1][1], nframes - 1])
            for i in range(len(true_sys) - 1):
                start1, stop1 = true_sys[i]
                start2, stop2 = true_sys[i + 1]
                true_dia.append([stop1, start2])
    else:
        true_dia = dia_frames
        true_sys = sys_frames
    
    # Detect diastolic peaks (e', l', a')
    e_i = []
    l_i = []
    a_i = []
    
    for start, stop in true_dia:
        e_start = int(start)
        e_stop = int(start + np.floor((stop - start) / 3))
        l_start = int(e_stop + 1)
        l_stop = int(l_start + np.floor((stop - start) / 3))
        a_start = int(l_stop + 1)
        a_stop = int(stop + 1)
        
        if pick_peak_by_subset:
            e_candidate_i = peakutils.peak.indexes(
                filt_arr[e_start:e_stop + 1], thres=peak_thres, min_dist=min_dist
            ) + e_start
            l_candidate_i = peakutils.peak.indexes(
                filt_arr[l_start:l_stop + 1], thres=peak_thres, min_dist=min_dist
            ) + l_start
            a_candidate_i = peakutils.peak.indexes(
                filt_arr[a_start:a_stop + 1], thres=peak_thres, min_dist=min_dist
            ) + a_start
        else:
            e_candidate_i = [k for k in peaks_i if ((k >= e_start) and (k <= e_stop))]
            l_candidate_i = [k for k in peaks_i if ((k >= l_start) and (k <= l_stop))]
            a_candidate_i = [k for k in peaks_i if ((k >= a_start) and (k <= a_stop))]
        
        # Process e' peak
        if len(e_candidate_i) > 0:
            e_candidate_y = [filt_arr[i] for i in e_candidate_i]
            e_index = np.argmax(e_candidate_y)
            e_i.append(e_candidate_i[e_index])
        else:
            print('Warning no e\' peak found! Using max value')
            e_i.append(np.argmax(filt_arr[e_start:e_stop]) + e_start)
        
        # Process l' peak
        if len(l_candidate_i) > 0:
            l_candidate_y = [filt_arr[i] for i in l_candidate_i]
            l_index = np.argmax(l_candidate_y)
            l_i.append(l_candidate_i[l_index])
        else:
            print('Warning no l\' peak found! Using max value')
            l_i.append(np.argmax(filt_arr[l_start:l_stop]) + l_start)
        
        # Process a' peak
        if len(a_candidate_i) > 0:
            a_candidate_y = [filt_arr[i] for i in a_candidate_i]
            a_index = np.argmax(a_candidate_y)
            a_i.append(a_candidate_i[a_index])
        else:
            print('Warning no a\' peak found! Using max value')
            a_i.append(np.argmax(filt_arr[a_start:a_stop]) + a_start)
    
    # Prepare return values
    result = {
        'filt_arr': filt_arr,
        'true_sys': true_sys,
        'true_dia': true_dia,
        'sys_px': frame_times[sys_i],
        'sys_py': filt_arr[sys_i],
        'e_px': frame_times[e_i],
        'e_py': filt_arr[e_i],
        'l_px': frame_times[l_i],
        'l_py': filt_arr[l_i],
        'a_px': frame_times[a_i],
        'a_py': filt_arr[a_i]
    }
    
    # Add all peaks if requested
    if show_all_peaks:
        result['all_px'] = frame_times[peaks_i]
        result['all_py'] = filt_arr[peaks_i]
    
    return result


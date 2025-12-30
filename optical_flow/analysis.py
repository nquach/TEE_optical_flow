"""
Analysis module for optical flow data processing.

This module provides functions for histogram calculation, magnitude calculations,
and centroid calculations, decoupled from the dataset object.
"""

import numpy as np
import cv2
import skimage.measure
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import mode

from optical_flow.config import AnalysisConfig


def find_correct_centroid(props):
    """
    Find the centroid of the region with largest pixel area.
    
    Args:
        props: List of region properties from skimage.measure.regionprops
    
    Returns:
        Centroid tuple (row, col) of the largest region
    """
    num_regions = len(props)
    area_list = []
    centroid_list = []
    for i in range(num_regions):
        prop = props[i]
        area_list.append(prop.area)
        centroid_list.append(prop.centroid)
    index = np.argmax(area_list)
    return centroid_list[index]


def calc_AV_centroid(mask_arr: np.ndarray, nframes: int, filter: bool = True,
                    savgol_window: int = 10, savgol_poly: int = 4,
                    verbose: bool = False) -> list:
    """
    Calculate AV (atrioventricular) centroid for each frame.
    
    Args:
        mask_arr: Mask array of shape (N, H, W, C)
        nframes: Number of frames
        filter: Whether to apply Savitzky-Golay filter
        savgol_window: Window size for Savitzky-Golay filter
        savgol_poly: Polynomial order for Savitzky-Golay filter
        verbose: Whether to print progress
    
    Returns:
        List of centroid tuples (row, col) for each frame
    """
    if verbose:
        print('Calculating AV centroids...')
    centroid_list = []
    for i in tqdm(range(nframes), disable=(not verbose)):
        frame = np.squeeze(mask_arr[i, :, :, 0])
        label_img = skimage.measure.label(frame)
        props = skimage.measure.regionprops(label_img)
        if len(props) >= 1:
            centroid_list.append(find_correct_centroid(props))
            continue
        else:
            if len(centroid_list) > 0:
                centroid_list.append(centroid_list[i - 1])  # Copy previous if empty
            else:
                # First frame has no mask - use center of image as default
                centroid_list.append((mask_arr.shape[1] / 2, mask_arr.shape[2] / 2))
            print('WARNING: EMPTY MASK at Frame ', i)
            continue
    
    if filter:
        if verbose:
            print('Applying savgol filter...')
        if len(centroid_list) < savgol_window:
            print('ERROR: Cannot apply savgol filter! List smaller than window')
        else:
            centroid_list = savgol_filter(centroid_list, savgol_window, savgol_poly, axis=0)
    
    if verbose:
        print('Sample centroid coord and vel:', centroid_list[0])
        print('Done!')
    return centroid_list


def radial_vecgrid(H: int, W: int, centroid_list: list, nframes: int) -> np.ndarray:
    """
    Create radial unit vector grid for each frame.
    
    Args:
        H: Image height
        W: Image width
        centroid_list: List of centroid coordinates (row, col) for each frame
        nframes: Number of frames
    
    Returns:
        Unit vector array of shape (N, H, W, 2)
    """
    vec_list = []
    for i in range(nframes):
        center = centroid_list[i]
        center_H = center[0]
        center_W = center[1]
        H_arr = np.full((H, W), center_H)
        W_arr = np.full((H, W), center_W)
        end_arr = np.transpose(np.asarray([H_arr, W_arr]), (1, 2, 0))
        pos_H = np.arange(0, H, 1)
        pos_W = np.arange(0, W, 1)
        pos_arr_H, pos_arr_W = np.meshgrid(pos_H, pos_W)
        pos_arr = np.transpose(np.asarray([pos_arr_H, pos_arr_W]), (2, 1, 0))
        vec_arr = end_arr - pos_arr
        norm = np.linalg.norm(vec_arr, axis=2)
        norm2 = np.transpose(np.asarray([norm, norm]), (1, 2, 0))
        unitvec = np.nan_to_num(vec_arr / norm2, nan=0)  # Division by 0 at center
        vec_list.append(unitvec)
    return np.stack(vec_list)


def calc_proj_mag(OF_arr: np.ndarray, unitvec_arr: np.ndarray) -> np.ndarray:
    """
    Calculate magnitude of projection of optical flow onto unit vectors.
    
    Args:
        OF_arr: Optical flow array of shape (N, H, W, 2)
        unitvec_arr: Unit vector array of shape (N, H, W, 2)
    
    Returns:
        Projected magnitude array of shape (N, H, W)
    """
    mag_proj_arr = np.sum(OF_arr * unitvec_arr, axis=3)
    return mag_proj_arr


def calculate_comp_magnitude(OF_arr: np.ndarray, centroid_list: list,
                            verbose: bool = False) -> tuple:
    """
    Calculate radial and longitudinal component magnitudes.
    
    Args:
        OF_arr: Optical flow array of shape (N, H, W, 2)
        centroid_list: List of centroid coordinates for each frame
        verbose: Whether to print progress
    
    Returns:
        Tuple of (rad_arr, long_arr) where each is (N, H, W)
    """
    nframes = len(centroid_list)
    OF_arr = OF_arr[:nframes, ...]  # Truncate to correct number of frames
    H = OF_arr.shape[1]
    W = OF_arr.shape[2]
    if verbose:
        print("calculating unit vector grids...")
    unitvec_arr = radial_vecgrid(H, W, centroid_list, nframes)
    ortho_unitvec_arr = np.stack([unitvec_arr[:, :, :, 1], -1 * unitvec_arr[:, :, :, 0]], axis=-1)
    if verbose:
        print('Shape of unit vector grid:', unitvec_arr.shape)
        print("calculating OF magnitude of projections onto unit vectors...")
    rad_arr = calc_proj_mag(OF_arr, unitvec_arr)
    long_arr = calc_proj_mag(OF_arr, ortho_unitvec_arr)
    return (rad_arr, long_arr)


def calc_bidirectional_hist(mag_arr: np.ndarray, nframes: int, perc_lo: int = 1,
                           perc_hi: int = 99, nbins: int = 1000) -> tuple:
    """
    Calculate bidirectional histogram (percentiles and frequency).
    
    Args:
        mag_arr: Magnitude array of shape (N, H, W)
        nframes: Number of frames
        perc_lo: Lower percentile
        perc_hi: Upper percentile
        nbins: Number of histogram bins
    
    Returns:
        Tuple of (mag_freq_arr, mag_edges, hi_arr, low_arr)
    """
    mag_max = np.max(mag_arr)
    mag_min = np.min(mag_arr)
    mag_edges = []
    perc_hi_list = []
    perc_low_list = []
    mag_freq_list = []
    
    for i in range(nframes):
        mag_frame = mag_arr[i, :, :]
        flat = np.ravel(mag_frame)
        flat_nonzero = flat[flat != 0]
        if len(flat_nonzero) == 0:
            print(f'ERROR len(flat_nonzero) is 0 for frame {i}')
            if len(perc_hi_list) > 0:
                perc_hi_list.append(perc_hi_list[-1])
                perc_low_list.append(perc_low_list[-1])
                mag_freq_list.append(mag_freq_list[-1])
            else:
                # First frame has no data - use default values
                perc_hi_list.append(mag_max)
                perc_low_list.append(mag_min)
                mag_freq_list.append(np.ones(nbins))  # Default frequency array
        else:
            perc_hi_list.append(np.percentile(flat_nonzero, perc_hi))
            perc_low_list.append(np.percentile(flat_nonzero, perc_lo))
            freq, mag_edges = np.histogram(flat_nonzero, bins=nbins, range=(mag_min, mag_max))
            mag_freq_list.append(freq + 1)  # Prevent frequency of 0 for lognorm
    
    hi_arr = np.asarray(perc_hi_list)
    low_arr = np.asarray(perc_low_list)
    mag_freq_arr = np.stack(mag_freq_list)
    return mag_freq_arr, mag_edges, hi_arr, low_arr


def calculate_3dhist(masked_arr: np.ndarray, nframes: int, nbins: int = 1000,
                    percentile: int = 99) -> tuple:
    """
    Calculate 3D histogram (magnitude and angle) from masked optical flow array.
    
    Args:
        masked_arr: Masked optical flow array of shape (N, H, W, 2)
        nframes: Number of frames
        nbins: Number of histogram bins
        percentile: Percentile for magnitude thresholding
    
    Returns:
        Tuple of (mag, ang, mag_edges, ang_edges, perc_hi)
    """
    mag_list = []
    ang_list = []
    for i in range(nframes):
        flow = np.squeeze(masked_arr[i, :, :, :])
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_list.append(mag)
        ang_list.append(ang)
    
    mag_arr = np.stack(mag_list)
    ang_arr = np.stack(ang_list)
    
    mag_freq_list = []
    mag_max = np.max(mag_arr)
    mag_min = np.min(mag_arr)
    mag_edges = []
    perc_hi = []
    
    for i in range(nframes):
        mag_frame = mag_arr[i, :, :]
        flat = np.ravel(mag_frame)
        flat_nonzero = flat[flat != 0]
        if len(flat_nonzero) == 0:
            print(f'ERROR len(flat_nonzero) is 0 for frame {i}')
            if len(perc_hi) > 0:
                perc_hi.append(perc_hi[-1])
                mag_freq_list.append(mag_freq_list[-1])
            else:
                perc_hi.append(mag_max)
                freq, mag_edges = np.histogram([mag_max], bins=nbins, range=(mag_min, mag_max))
                mag_freq_list.append(freq + 1)
        else:
            perc_hi.append(np.percentile(flat_nonzero, percentile))
            freq, mag_edges = np.histogram(flat_nonzero, bins=nbins, range=(mag_min, mag_max))
            mag_freq_list.append(freq + 1)
    
    ang_freq_list = []
    ang_max = np.max(ang_arr)
    ang_min = np.min(ang_arr)
    ang_edges = []
    
    for i in range(nframes):
        ang_frame = ang_arr[i, :, :]
        flat = np.ravel(ang_frame)
        flat_nonzero = flat[flat != 0]
        if len(flat_nonzero) == 0:
            print(f'ERROR len(flat_nonzero) is 0 for frame {i}')
            if len(ang_freq_list) > 0:
                ang_freq_list.append(ang_freq_list[-1])
            else:
                freq, ang_edges = np.histogram([ang_max], bins=nbins, range=(ang_min, ang_max))
                ang_freq_list.append(freq + 1)
        else:
            freq, ang_edges = np.histogram(flat_nonzero, bins=nbins, range=(ang_min, ang_max))
            ang_freq_list.append(freq + 1)
    
    mag = np.stack(mag_freq_list)
    ang = np.stack(ang_freq_list)
    return mag, ang, mag_edges, ang_edges, np.asarray(perc_hi)


def calculate_3dhist_radlong(param_arr: np.ndarray, av_masks: np.ndarray, nframes: int,
                            nbins: int = 1000, perc_lo: int = 1, perc_hi: int = 99,
                            av_filter_flag: bool = True, av_savgol_window: int = 10,
                            av_savgol_poly: int = 4, verbose: bool = False) -> dict:
    """
    Calculate 3D histogram for radial and longitudinal components.
    
    Args:
        param_arr: Parameter array (velocity, acceleration, or PWR) of shape (N, H, W, 2)
        av_masks: AV mask array of shape (N, H, W, C)
        nframes: Number of frames
        nbins: Number of histogram bins
        perc_lo: Lower percentile
        perc_hi: Upper percentile
        av_filter_flag: Whether to filter AV centroids
        av_savgol_window: Savitzky-Golay window size
        av_savgol_poly: Savitzky-Golay polynomial order
        verbose: Whether to print progress
    
    Returns:
        Dictionary with 'radial' and 'longitudinal' keys, each containing
        (mag_freq_arr, mag_edges, hi_arr, low_arr)
    """
    centroid_list = calc_AV_centroid(av_masks, nframes, filter=av_filter_flag,
                                    savgol_window=av_savgol_window,
                                    savgol_poly=av_savgol_poly, verbose=verbose)
    rad_arr, long_arr = calculate_comp_magnitude(param_arr, centroid_list, verbose=False)
    
    rad_mag_freq_arr, rad_mag_edges, rad_hi_arr, rad_low_arr = calc_bidirectional_hist(
        rad_arr, nframes, perc_lo=perc_lo, perc_hi=perc_hi, nbins=nbins
    )
    long_mag_freq_arr, long_mag_edges, long_hi_arr, long_low_arr = calc_bidirectional_hist(
        long_arr, nframes, perc_lo=perc_lo, perc_hi=perc_hi, nbins=nbins
    )
    
    data_dict = {}
    data_dict['radial'] = (rad_mag_freq_arr, rad_mag_edges[:-1], rad_hi_arr, rad_low_arr)
    data_dict['longitudinal'] = (long_mag_freq_arr, long_mag_edges[:-1], long_hi_arr, long_low_arr)
    return data_dict


"""
Plotting utilities for optical flow visualization.

This module provides helper functions for common plot elements like
systole/diastole shading, waveform overlays, and figure layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def add_systole_diastole_shading(ax, frame_times: np.ndarray, sys_frames: List[Tuple[int, int]],
                                 dia_frames: List[Tuple[int, int]], nframes: int,
                                 sys_color: str = '0.8', dia_color: str = '0.8',
                                 sys_alpha: float = 0.5, dia_alpha: float = 0.25,
                                 add_labels: bool = True):
    """
    Add systole and diastole shading to an axis.
    
    Args:
        ax: Matplotlib axis to add shading to
        frame_times: Array of time values for each frame
        sys_frames: List of (start, stop) frame pairs for systole
        dia_frames: List of (start, stop) frame pairs for diastole
        nframes: Total number of frames
        sys_color: Color for systole shading
        dia_color: Color for diastole shading
        sys_alpha: Alpha for systole shading
        dia_alpha: Alpha for diastole shading
        add_labels: Whether to add legend labels
    """
    if len(sys_frames) > 0:
        counter = 0
        for start, stop in sys_frames:
            if stop >= nframes:
                stop = nframes - 1
            if counter == 0 and add_labels:
                ax.axvspan(frame_times[start], frame_times[stop],
                          facecolor=sys_color, alpha=sys_alpha, label='Systole')
            else:
                ax.axvspan(frame_times[start], frame_times[stop],
                          facecolor=sys_color, alpha=sys_alpha)
            counter += 1
    
    if len(dia_frames) > 0:
        counter = 0
        for start, stop in dia_frames:
            if stop >= nframes:
                stop = nframes - 1
            if counter == 0 and add_labels:
                ax.axvspan(frame_times[start], frame_times[stop],
                          facecolor=dia_color, alpha=dia_alpha, label='Diastole')
            else:
                ax.axvspan(frame_times[start], frame_times[stop],
                          facecolor=dia_color, alpha=dia_alpha)
            counter += 1


def plot_waveform_with_shading(ax, waveform_data: np.ndarray, waveform_times: np.ndarray,
                               frame_times: np.ndarray, sys_frames: List[Tuple[int, int]],
                               dia_frames: List[Tuple[int, int]], nframes: int,
                               xlabel: str = 'Time (ms)', ylabel: str = 'Amplitude'):
    """
    Plot waveform with systole/diastole shading.
    
    Args:
        ax: Matplotlib axis to plot on
        waveform_data: Waveform data array
        waveform_times: Time array for waveform
        frame_times: Time array for frames
        sys_frames: List of (start, stop) frame pairs for systole
        dia_frames: List of (start, stop) frame pairs for diastole
        nframes: Total number of frames
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    ax.plot(waveform_times, waveform_data)
    add_systole_diastole_shading(ax, frame_times, sys_frames, dia_frames, nframes)
    ax.legend(loc='lower right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def create_heatmap_figure(show_waveform: bool = False, show_sysdia: bool = False,
                          nrows: int = 2, figsize: Tuple[int, int] = (8, 6)) -> Tuple[plt.Figure, List]:
    """
    Create figure with appropriate subplot layout for heatmaps.
    
    Args:
        show_waveform: Whether to include waveform subplot
        show_sysdia: Whether to show systole/diastole shading
        nrows: Number of rows (excluding waveform)
        figsize: Figure size tuple
    
    Returns:
        Tuple of (figure, list of axes)
    """
    if show_waveform and show_sysdia:
        fig, axes = plt.subplots(nrows=nrows + 1, ncols=1, sharex=True, sharey=False,
                                 figsize=(figsize[0], figsize[1] + 1),
                                 layout='constrained',
                                 gridspec_kw={'height_ratios': [4] * nrows + [1]})
        return fig, axes
    elif show_sysdia and not show_waveform:
        fig, axes = plt.subplots(nrows=nrows + 1, ncols=1, sharex=True, sharey=False,
                                 figsize=(figsize[0], figsize[1] + 0.5),
                                 layout='constrained',
                                 gridspec_kw={'height_ratios': [4] * nrows + [0.5]})
        return fig, axes
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, sharey=False,
                                 figsize=figsize)
        if nrows == 1:
            axes = [axes]
        return fig, axes


def setup_colorbar(im, ax, label: str = 'log(freq)'):
    """
    Setup colorbar for a plot.
    
    Args:
        im: Image/pcolormesh object
        ax: Axis to attach colorbar to
        label: Colorbar label
    """
    plt.colorbar(im, ax=ax, label=label)


def get_colormap(name: str):
    """
    Get matplotlib colormap by name.
    
    Args:
        name: Colormap name
    
    Returns:
        Matplotlib colormap
    """
    return plt.cm.get_cmap(name)


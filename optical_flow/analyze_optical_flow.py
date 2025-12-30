import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v3 as iio
import matplotlib.colors
import matplotlib.cm as cm
from tqdm import tqdm
from skimage import feature
from skimage.color import rgb2gray, gray2rgb
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.transform import warp_polar
import skimage.measure
from matplotlib.colors import LogNorm, Normalize
import pickle as pkl
import multiprocessing
import gc
from scipy.signal import savgol_filter
import peakutils
import h5py
import polars as pl
from scipy.stats import mode
from tsmoothie.smoother import SpectralSmoother
import neurokit2 as nk
import traceback
from optical_flow_utils import *
from optical_flow_dataset import OpticalFlowDataset
from optical_flow.config import (
    CardiacCycleConfig, VisualizationConfig, ProcessingConfig, AnalysisConfig,
    CardiacCycleMethodConfig, default_cardiac_cycle_config, default_visualization_config,
    default_processing_config, default_analysis_config, ecg_gated_config, arterial_gated_config
)
from optical_flow.cardiac_cycle_detection import create_detector
from shutil import copy
import argparse

#AUTOMATIC CARDIAC CYCLE DETECTION

def _build_cardiac_cycle_config_from_params(smooth_fraction=0.2, pad_len=20, sys_thres=0.9, 
                                           dia_thres=0.5, rr_sys_ratio=0.333, sys_extension=2,
                                           t_peak_thres=0.5, t_min_dist=20, rr_search_range=[0.2, 0.75],
                                           low_peak_thres=0.9, low_min_dist=50, high_peak_thres=0.9,
                                           high_min_dist=50, sys_upstroke_multiplier=2, sys_upstroke_offset=5,
                                           **kwargs) -> CardiacCycleConfig:
    """Helper to build CardiacCycleConfig from individual parameters."""
    config = CardiacCycleConfig(
        smooth_fraction=smooth_fraction,
        pad_len=pad_len,
        sys_thres=sys_thres,
        dia_thres=dia_thres,
        rr_sys_ratio=rr_sys_ratio,
        sys_extension=sys_extension,
        t_peak_thres=t_peak_thres,
        t_min_dist=t_min_dist,
        rr_search_range=rr_search_range,
        low_peak_thres=low_peak_thres,
        low_min_dist=low_min_dist,
        high_peak_thres=high_peak_thres,
        high_min_dist=high_min_dist,
        sys_upstroke_multiplier=sys_upstroke_multiplier,
        sys_upstroke_offset=sys_upstroke_offset
    )
    return config

def _build_visualization_config_from_params(save_dir=None, show_plot=False, show_img=False, 
                                           save_cc_plot=False, nbins=1000, invert_rad_yaxis=False,
                                           invert_long_yaxis=False, **kwargs) -> VisualizationConfig:
    """Helper to build VisualizationConfig from individual parameters."""
    return VisualizationConfig(
        save_dir=save_dir,
        show_plot=show_plot,
        show_img=show_img,
        save_cc_plot=save_cc_plot,
        nbins=nbins,
        invert_rad_yaxis=invert_rad_yaxis,
        invert_long_yaxis=invert_long_yaxis
    )

def _build_processing_config_from_params(recalculate=True, verbose=False, sampling_rate=None, **kwargs) -> ProcessingConfig:
    """Helper to build ProcessingConfig from individual parameters."""
    config = ProcessingConfig(
        recalculate=recalculate,
        verbose=verbose,
        sampling_rate=sampling_rate
    )
    if sampling_rate is not None:
        config.sampling_rate = sampling_rate
    return config

def _build_analysis_config_from_params(av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4,
                                      perc_lo=1, perc_hi=99, **kwargs) -> AnalysisConfig:
    """Helper to build AnalysisConfig from individual parameters."""
    return AnalysisConfig(
        av_filter_flag=av_filter_flag,
        av_savgol_window=av_savgol_window,
        av_savgol_poly=av_savgol_poly,
        perc_lo=perc_lo,
        perc_hi=perc_hi
    )

def sysdia_frames_by_angle(ds, param, label, smooth_fraction=0.2, pad_len=20, recalculate=False, verbose=False,
												save_angle_plot=False, save_dir=None, show_plot=False,
												cc_config=None, vis_config=None, proc_config=None):
	"""Detect cardiac cycle using angle-based method. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	if cc_config is None:
		cc_config = _build_cardiac_cycle_config_from_params(smooth_fraction=smooth_fraction, pad_len=pad_len)
	if vis_config is None:
		vis_config = _build_visualization_config_from_params(save_dir=save_dir, show_plot=show_plot, save_cc_plot=save_angle_plot)
	if proc_config is None:
		proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('angle', cc_config=cc_config, vis_config=vis_config, proc_config=proc_config)
	return detector.detect(ds, param=param, label=label)

def sysdia_frames_by_area(ds, label, smooth_fraction=0.3, pad_len=20, recalculate=False, verbose=False,
												save_area_plot=False, save_dir=None, show_plot=False, sys_thres = 0.9, dia_thres=0.5):
	"""Detect cardiac cycle using area-based method. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	cc_config = _build_cardiac_cycle_config_from_params(
		smooth_fraction=smooth_fraction, pad_len=pad_len,
		sys_thres=sys_thres, dia_thres=dia_thres
	)
	vis_config = _build_visualization_config_from_params(
		save_dir=save_dir, show_plot=show_plot, save_cc_plot=save_area_plot
	)
	proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('area', cc_config=cc_config, vis_config=vis_config, proc_config=proc_config)
	return detector.detect(ds, label=label)

def sysdia_frames_by_RTime(ds, recalculate=False, verbose=False, rr_sys_ratio=0.333):
	"""Detect cardiac cycle using DICOM R-wave time metadata. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	cc_config = _build_cardiac_cycle_config_from_params(rr_sys_ratio=rr_sys_ratio)
	proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('metadata', cc_config=cc_config, proc_config=proc_config)
	return detector.detect(ds)

def sysdia_frames_by_ecg_lazy(ds, ecg_arr, sampling_rate=500, smooth_fraction=0.2, pad_len=20, recalculate=False, verbose=False,
												save_ecg_plot=False, save_dir=None, show_plot=False, rr_sys_ratio=0.333, sys_extension=2):
	"""Detect cardiac cycle using ECG lazy method. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	cc_config = _build_cardiac_cycle_config_from_params(
		smooth_fraction=smooth_fraction, pad_len=pad_len,
		rr_sys_ratio=rr_sys_ratio, sys_extension=sys_extension
	)
	vis_config = _build_visualization_config_from_params(
		save_dir=save_dir, show_plot=show_plot, save_cc_plot=save_ecg_plot
	)
	proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('ecg_lazy', cc_config=cc_config, vis_config=vis_config, proc_config=proc_config)
	return detector.detect(ds, ecg_arr=ecg_arr, sampling_rate=sampling_rate)

def sysdia_frames_by_ecg(ds, ecg_arr, sampling_rate=500, smooth_fraction=0.2, pad_len=20, recalculate=False, verbose=False,
												save_ecg_plot=False, save_dir=None, show_plot=False, t_peak_thres=0.5,
												 t_min_dist = 20, rr_search_range=[0.2,0.75]):
	"""Detect cardiac cycle using ECG T-wave method. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	cc_config = _build_cardiac_cycle_config_from_params(
		smooth_fraction=smooth_fraction, pad_len=pad_len,
		t_peak_thres=t_peak_thres, t_min_dist=t_min_dist,
		rr_search_range=rr_search_range
	)
	vis_config = _build_visualization_config_from_params(
		save_dir=save_dir, show_plot=show_plot, save_cc_plot=save_ecg_plot
	)
	proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('ecg', cc_config=cc_config, vis_config=vis_config, proc_config=proc_config)
	return detector.detect(ds, ecg_arr=ecg_arr, sampling_rate=sampling_rate)

def sysdia_frames_by_art(ds, art_arr, sampling_rate=125, smooth_fraction=0.2, pad_len=20, recalculate=False, verbose=False,
												save_art_plot=False, save_dir=None, show_plot=False, low_peak_thres=0.9, low_min_dist = 50,
												 high_peak_thres=0.9, high_min_dist=50, sys_upstroke_multiplier=2, sys_upstroke_offset=5):
	"""Detect cardiac cycle using arterial pressure. Wrapper for backward compatibility."""
	# Build config objects from parameters if not provided
	cc_config = _build_cardiac_cycle_config_from_params(
		smooth_fraction=smooth_fraction, pad_len=pad_len,
		low_peak_thres=low_peak_thres, low_min_dist=low_min_dist,
		high_peak_thres=high_peak_thres, high_min_dist=high_min_dist,
		sys_upstroke_multiplier=sys_upstroke_multiplier, sys_upstroke_offset=sys_upstroke_offset
	)
	vis_config = _build_visualization_config_from_params(
		save_dir=save_dir, show_plot=show_plot, save_cc_plot=save_art_plot
	)
	proc_config = _build_processing_config_from_params(recalculate=recalculate, verbose=verbose)
	
	# Use new detector class
	detector = create_detector('arterial', cc_config=cc_config, vis_config=vis_config, proc_config=proc_config)
	return detector.detect(ds, art_arr=art_arr, sampling_rate=sampling_rate)

#AV Centroid Finding
def find_correct_centroid(props):
	num_regions = len(props)
	area_list = []
	centroid_list = []
	for i in range(num_regions):
		prop = props[i]
		area_list.append(prop.area)
		centroid_list.append(prop.centroid)
	index = np.argmax(area_list)
	return centroid_list[index] #return centroid of region with largest pixel area

def calc_AV_centroid(mask_arr, nframes, filter=True, savgol_window=10, savgol_poly=4, verbose=False):
	if verbose:
		print('Calculating AV centroids...')
	centroid_list = []
	for i in tqdm(range(nframes), disable=(not verbose)):
		frame = np.squeeze(mask_arr[i,:,:,0])
		label_img = skimage.measure.label(frame)
		props = skimage.measure.regionprops(label_img)
		if len(props) >= 1:
			centroid_list.append(find_correct_centroid(props))
			continue
		else:
			if len(centroid_list) > 0:
				centroid_list.append(centroid_list[i-1]) #copy the same if image mask is empty
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

def radial_vecgrid(H, W, centroid_list, nframes):
	vec_list = []
	for i in range(nframes):
		center = centroid_list[i]
		center_H = center[0]
		center_W = center[1]
		H_arr = np.full((H, W), center_H)
		W_arr = np.full((H,W), center_W)
		end_arr = np.transpose(np.asarray([H_arr, W_arr]), (1, 2, 0))
		pos_H = np.arange(0, H, 1)
		pos_W = np.arange(0, W, 1)
		pos_arr_H, pos_arr_W = np.meshgrid(pos_H, pos_W,)
		pos_arr = np.transpose(np.asarray([pos_arr_H, pos_arr_W]), (2, 1, 0))
		vec_arr = end_arr - pos_arr
		norm = np.linalg.norm(vec_arr, axis=2)
		norm2 = np.transpose(np.asarray([norm,norm]), (1,2,0))
		unitvec = np.nan_to_num(vec_arr/norm2, nan=0) #there is a division by 0 at position (center_x, center_y)
		vec_list.append(unitvec)
	return np.stack(vec_list)

def calc_proj_mag(OF_arr, unitvec_arr): #must be (N, H, W, 2)
	mag_proj_arr = np.sum(OF_arr*unitvec_arr, axis=3)
	return mag_proj_arr

def calculate_comp_magnitude(OF_arr, centroid_list, verbose=False): #(N, H, W, 2)
	nframes = len(centroid_list)
	OF_arr = OF_arr[:nframes,...] #truncate to the correct number of frames
	H = OF_arr.shape[1] #600
	W = OF_arr.shape[2] #800
	if verbose:
		print("calculating unit vector grids...")
	unitvec_arr = radial_vecgrid(H, W, centroid_list, nframes)
	ortho_unitvec_arr = np.stack([unitvec_arr[:,:,:,1], -1*unitvec_arr[:,:,:,0]], axis=-1)
	if verbose:
		print('Shape of unit vector grid:', unitvec_arr.shape)
		print("calculating OF magnitude of projections onto unit vectors...")
	rad_arr = calc_proj_mag(OF_arr, unitvec_arr)
	long_arr = calc_proj_mag(OF_arr, ortho_unitvec_arr)
	return (rad_arr, long_arr)

#RADIAL LONGITUDINAL PLOTS
def calc_bidirectional_hist(mag_arr, nframes, perc_lo=1, perc_hi=99, nbins=1000):
	mag_max = np.max(mag_arr)
	mag_min = np.min(mag_arr)
	mag_edges = []
	perc_hi_list = []
	perc_low_list = []
	mag_freq_list = []
	for i in range(nframes):
		mag_frame = mag_arr[i,:,:]
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
			mag_freq_list.append(freq+1) #prevent frequency of 0 for lognorm

	hi_arr = np.asarray(perc_hi_list)
	low_arr = np.asarray(perc_low_list)
	mag_freq_arr = np.stack(mag_freq_list)
	return mag_freq_arr, mag_edges, hi_arr, low_arr

def calculate_3dhist_radlong(ds, param, nbins=1000, perc_lo=1, perc_hi=99,
												av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4,
												verbose=False):
	if not ds._validate_param(param):
		print(f'ERROR! {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if 'RVIO' not in ds.mode:
		print(f'ERROR only mode=RVIO_2class is supported for radlong functions! got mode={ds.mode}')
		return

	param_arr = ds.get_masked_arr(param, 'rv')
	av_masks = ds.get_mask('av')

	centroid_list = calc_AV_centroid(av_masks, ds.nframes, filter=av_filter_flag, savgol_window=av_savgol_window,
																	 savgol_poly=av_savgol_poly, verbose=verbose)
	rad_arr, long_arr = calculate_comp_magnitude(param_arr, centroid_list, verbose=False)
	rad_mag_freq_arr, rad_mag_edges, rad_hi_arr, rad_low_arr = calc_bidirectional_hist(rad_arr, ds.nframes, perc_lo=perc_lo,
																																										 perc_hi=perc_hi, nbins=nbins)
	long_mag_freq_arr, long_mag_edges, long_hi_arr, long_low_arr = calc_bidirectional_hist(long_arr, ds.nframes, perc_lo=perc_lo,
																																										 perc_hi=perc_hi, nbins=nbins)
	data_dict = {}
	data_dict['radial'] = (rad_mag_freq_arr, rad_mag_edges[:-1], rad_hi_arr, rad_low_arr)
	data_dict['longitudinal'] = (long_mag_freq_arr, long_mag_edges[:-1], long_hi_arr, long_low_arr)
	return data_dict

def plot_radlong_heatmap(ds, param, label, save_dir, nbins=1000,
												 av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4,
												 show_img=False, recalculate=True, verbose=False, invert_rad_yaxis=False, invert_long_yaxis=False,
												 waveform_data=None, sampling_rate=None,
												 show_sysdia=False, cc_method='angle', cc_label='rv_inner',
												 cc_smooth_fraction=0.2, save_cc_plot=False, area_sys_thres=0.9,
												 area_dia_thres=0.5, rr_sys_ratio=0.333):
	if not ds._validate_param(param):
		print(f'ERROR! {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return

	safe_makedir(save_dir)
	if show_sysdia:
		save_name = ds.filename + '_' + label + '_' + param + '_' + cc_method +'_radlong_3dhist.png'
	else:
		save_name = ds.filename + '_' + label + '_' + param +'_radlong_3dhist.png'
	save_path = os.path.join(save_dir, save_name)
	if os.path.exists(save_path) and not recalculate:
		print(f'{save_path} already exists, skipping!')
		return

	if show_sysdia:
		if not ds.CARDIACCYCLE_CALCULATED or recalculate:
			if cc_method == 'area':
				sysdia_frames_by_area(ds, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_area_plot=save_cc_plot, save_dir=save_dir, show_plot=False, sys_thres=area_sys_thres, dia_thres=area_dia_thres)
			elif cc_method == 'angle':
				sysdia_frames_by_angle(ds, param, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_angle_plot=save_cc_plot, save_dir=save_dir, show_plot=False)
			elif cc_method == 'metadata':
				sysdia_frames_by_RTime(ds, recalculate=recalculate, verbose=verbose, rr_sys_ratio=rr_sys_ratio)
			elif cc_method == 'ecg':
				sysdia_frames_by_ecg(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			elif cc_method == 'ecg_lazy':
				sysdia_frames_by_ecg_lazy(ds, waveform_data, sampling_rate=sampling_rate, smooth_fraction=0.2, pad_len=20, recalculate=recalculate,
																	verbose=verbose, rr_sys_ratio=0.333)
			elif cc_method == 'arterial':
				sysdia_frames_by_art(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			else:
				print(f'ERROR! cc_method must be [area, angle, ecg, metadata, arterial] not {cc_method}')
				return

		true_dia = ds.dia_frames
		true_sys = ds.sys_frames

	if waveform_data is None:
		waveform_exists = False
	else:
		waveform_exists = True

	if waveform_exists and sampling_rate is None:
		print('ERROR, must specify sampling rate if waveform data is provided!')
		return

	data_dict = calculate_3dhist_radlong(ds, param, nbins=nbins, perc_lo=1, perc_hi=99,
												av_filter_flag=av_filter_flag, av_savgol_window=av_savgol_window, av_savgol_poly=av_savgol_poly,
												verbose=verbose)
	rad_mag_freq_arr, rad_mag_edges, _, _ = data_dict['radial']
	long_mag_freq_arr, long_mag_edges, _, _ = data_dict['longitudinal']

	rad_norm = LogNorm(vmin=np.min(rad_mag_freq_arr), vmax=np.max(rad_mag_freq_arr))
	long_norm = LogNorm(vmin=np.min(long_mag_freq_arr), vmax=np.max(long_mag_freq_arr))

	if waveform_exists and show_sysdia:
		if 'ecg' in cc_method:
			waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
		fig, (ax1, ax2, ax_t) = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=False, figsize=(8,9), layout='constrained',
																 gridspec_kw={'height_ratios': [4, 4, 1]})
		wave_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax_t.plot(wave_times, waveform_data)
		x = np.arange(ds.nframes)*(1000/ds.frame_rate)
		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1
		ax_t.legend(loc='lower right')
		ax_t.set_xlabel('Time (ms)')
	elif show_sysdia and not waveform_exists:
		fig, (ax1, ax2, ax_t) = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=False, figsize=(8,9), layout='constrained',
																 gridspec_kw={'height_ratios': [4, 4, 0.5]})
		x = np.arange(ds.nframes)*(1000/ds.frame_rate)
		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1
		ax_t.legend(loc='lower right')
		ax_t.set_xlabel('Time (ms)')
	else:
		fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, figsize=(8,6))
		ax2.set_xlabel('Time (ms)')
	xgrid = np.arange(ds.nframes)*(1000/ds.frame_rate)
	ygrid_rad = rad_mag_edges
	ygrid_long = long_mag_edges
	plt1 = ax1.pcolormesh(xgrid, ygrid_rad, rad_mag_freq_arr.T, norm=rad_norm, cmap='hot')
	ax1.set_ylabel(param.capitalize() + ' (' + ds._param_unit(param) + ')')
	ax1.set_title('Radial ' + param.capitalize() + ' vs Time (ms)')
	plt2 = ax2.pcolormesh(xgrid, ygrid_long, long_mag_freq_arr.T, norm=long_norm, cmap='hot')
	ax2.set_ylabel(param.capitalize() + ' (' + ds._param_unit(param) + ')')
	ax2.set_title('Longitudinal ' + param.capitalize() + ' vs Time (ms)')
	if invert_rad_yaxis:
		ax1.invert_yaxis()
	if invert_long_yaxis:
		ax2.invert_yaxis()
	fig.colorbar(plt1, ax=ax1, label='log(freq)')
	fig.colorbar(plt2, ax=ax2, label='log(freq)')
	fig.savefig(save_path)
	if not show_img:
		plt.close(fig)

def overlay3(dcm_arr, rad_arr, long_arr, verbose=False):
	x1 = np.concatenate([dcm_arr, dcm_arr], axis=2)
	x2 = np.concatenate([rad_arr, long_arr], axis=2)
	if verbose:
		print('Shapes on concat:', rad_arr.shape, '->', x2.shape)
	x = (0.5 * (x1/ np.max(x1)) + 0.5 * (x2/ np.max(x2))) * 255
	return x.astype(np.uint8)

def visualize_radlong(ds, param, save_dir, fps=30, verbose=False, av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4):
	if not ds._validate_param(param):
		print(f'ERROR! {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if 'RVIO' not in ds.mode:
		print(f'ERROR only RVIO modes are supported for radlong visualization, got mode={ds.mode}')
		return
	if verbose:
		print('Retrieving masked param arr and echo pixel arr...')
	param_arr = ds.get_masked_arr(param, 'rv')
	av_masks = ds.get_mask('av')

	centroid_list = calc_AV_centroid(av_masks, ds.nframes, filter=av_filter_flag, savgol_window=av_savgol_window,
																	 savgol_poly=av_savgol_poly, verbose=verbose)
	rad_arr, long_arr = calculate_comp_magnitude(param_arr, centroid_list, verbose=False)

	pixel_arr = gray2rgb(ds.get_echo())

	rad_list = np.split(rad_arr, rad_arr.shape[0])
	long_list = np.split(long_arr, long_arr.shape[0])
	rad_arr_sqz= [np.squeeze(rad_arr) for rad_arr in rad_list]
	long_arr_sqz = [np.squeeze(long_arr) for long_arr in long_list]
	norm = matplotlib.colors.CenteredNorm()

	rad_norm_list = []
	long_norm_list = []
	if verbose:
		print('Converting to colormap...')
	for rad_arr,long_arr in zip(rad_arr_sqz, long_arr_sqz):
		rad_norm = norm(rad_arr)
		long_norm = norm(long_arr)
		rad_rgb = plt.cm.bwr(rad_norm)
		long_rgb = plt.cm.BrBG(long_norm)
		rad_norm_list.append(rad_rgb[:,:,0:3])
		long_norm_list.append(long_rgb[:,:,0:3])

	rad_rgb_arr = np.stack(rad_norm_list)
	long_rgb_arr = np.stack(long_norm_list)
	del rad_norm_list
	del long_norm_list
	del rad_list
	del long_list
	del rad_arr
	del long_arr
	del rad_arr_sqz
	del long_arr_sqz
	del norm
	gc.collect()

	if verbose:
		print('Overlaying colormaps and greyscale imgs...')
	overlay_arr = overlay3(pixel_arr[0:ds.nframes,...], rad_rgb_arr, long_rgb_arr, verbose=verbose)
	del pixel_arr
	del rad_rgb_arr
	del long_rgb_arr
	gc.collect()

	safe_makedir(save_dir)
	save_name = ds.filename + '_' + param + '_radlong_overlay.mp4'
	save_path = os.path.join(save_dir, save_name)
	writer = iio.get_writer(save_path, fps=fps)

	for i in tqdm(range(ds.nframes), disable=(not verbose)):
		writer.append_data(overlay_arr[i,...])
	writer.close()

#by default looks for systolic peaks in lo_arr and diastolic peaks in hi_arr
def calculate_radlong_peaks(ds, hi_arr, lo_arr, frame_times, smooth_fraction=0.3, pad_len=20,	peak_thres=0.5, min_dist=5,
										pick_peak_by_subset=False, cc_method='angle'):
	lo_smoother = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
	hi_smoother = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
	lo_smoother.smooth(lo_arr)
	hi_smoother.smooth(hi_arr)
	filt_lo = lo_smoother.smooth_data[0]
	filt_hi = hi_smoother.smooth_data[0]

	hi_peaks_i = peakutils.peak.indexes(filt_hi, thres=peak_thres, min_dist=min_dist)
	lo_peaks_i = peakutils.peak.indexes(filt_lo * -1, thres=peak_thres, min_dist=min_dist)

	sys_i = []
	e_i = []
	a_i = []
	l_i = []
	true_sys = []
	#pick lowest peak for systole in filt_lo * -1
	for start, stop in ds.sys_frames:
		if pick_peak_by_subset:
			candidate_i = peakutils.peak.indexes(filt_lo[start:stop+1] * -1, thres=peak_thres, min_dist=min_dist) + start
		else:
			candidate_i = [k for k in lo_peaks_i if ((k >= start) and (k <= stop))]
		if len(candidate_i) > 0:
			candidate_y = [filt_lo[i] for i in candidate_i]
			index = np.argmin(candidate_y)
			sys_i.append(candidate_i[index])
			true_sys.append([start, stop])
		else:
			print('Warning no systolic peak found! Using max value')
			sys_i.append(np.argmin(filt_lo[start:stop])+start)

	if cc_method == 'angle':
		true_dia = []
		if true_sys[0][0] > 1:
			true_dia.append([0, true_sys[0][0] - 1])
		if true_sys[-1][1] < (ds.nframes - 2):
			true_dia.append([true_sys[-1][1], ds.nframes - 1])
		for i in range(len(true_sys)-1):
			start1, stop1 = true_sys[i]
			start2, stop2 = true_sys[i+1]
			true_dia.append([stop1, start2])
	else:
		true_dia = ds.dia_frames
		true_sys = ds.sys_frames

	#pick highest peak for diastole in filt_hi
	for start, stop in true_dia:
		e_start = int(start)
		e_stop = int(start + np.floor((stop-start)/3))
		l_start = int(e_stop+1)
		l_stop = int(l_start + np.floor((stop-start)/3))
		a_start = int(l_stop + 1)
		a_stop = int(stop+1)
		#find e' first
		if pick_peak_by_subset:
			e_candidate_i = peakutils.peak.indexes(filt_hi[e_start:e_stop+1], thres=peak_thres, min_dist=min_dist) + e_start
			l_candidate_i = peakutils.peak.indexes(filt_hi[l_start:l_stop+1], thres=peak_thres, min_dist=min_dist) + l_start
			a_candidate_i = peakutils.peak.indexes(filt_hi[a_start:a_stop+1], thres=peak_thres, min_dist=min_dist) + a_start
		else:
			e_candidate_i = [k for k in hi_peaks_i if ((k >= e_start) and (k <= (e_stop)))]
			l_candidate_i = [k for k in hi_peaks_i if ((k >= l_start) and (k <= (l_stop)))]
			a_candidate_i = [k for k in hi_peaks_i if ((k >= a_start) and (k <= (a_stop)))]
		if len(e_candidate_i) > 0:
			e_candidate_y = [filt_hi[i] for i in e_candidate_i]
			e_index = np.argmax(e_candidate_y)
			e_i.append(e_candidate_i[e_index])
		else:
			print('Warning no e\' peak found! Using max value')
			e_i.append(np.argmin(filt_hi[e_start:e_stop])+e_start)
		if len(l_candidate_i) > 0:
			l_candidate_y = [filt_hi[i] for i in l_candidate_i]
			l_index = np.argmax(l_candidate_y)
			l_i.append(l_candidate_i[l_index])
		else:
			print('Warning no l\' peak found! Using max value')
			l_i.append(np.argmin(filt_hi[l_start:l_stop])+l_start)
		if len(a_candidate_i) > 0:
			a_candidate_y = [filt_hi[i] for i in a_candidate_i]
			a_index = np.argmax(a_candidate_y)
			a_i.append(a_candidate_i[a_index])
		else:
			print('Warning no a\' peak found! Using max value')
			a_i.append(np.argmin(filt_hi[a_start:a_stop])+a_start)

	sys_px = frame_times[sys_i]
	sys_py = filt_lo[sys_i]
	e_px = frame_times[e_i]
	e_py = filt_hi[e_i]
	l_px = frame_times[l_i]
	l_py = filt_hi[l_i]
	a_px = frame_times[a_i]
	a_py = filt_hi[a_i]

	return (filt_hi, filt_lo, true_sys, true_dia, sys_px, sys_py, e_px, e_py, l_px, l_py, a_px, a_py)

#AUTOMATIC OPTICAL FLOW PARAM CALCULATION AND PLOTTING
def percentile_plot_radlong(ds, param, save_dir, cc_method='angle', cc_label='rv_inner', true_sysdia_mode='radial',
										cc_smooth_fraction=0.2, cc_pad_len=20, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4, perc_lo=1, perc_hi=99,
										waveform_data=None, sampling_rate=500, pick_peak_by_subset=True,
										nbins=1000, smooth_fraction=0.3, pad_len=20,
										peak_thres=0.2, min_dist=5, show_all_peaks=False, show_img=False,
										print_report=False, return_value=True, recalculate=True, verbose=False, rr_sys_ratio=0.333):
	if not ds._validate_param(param):
		print(f'ERROR! param input {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if cc_method == 'area' or cc_method == 'angle':
		if not ds._validate_label(cc_label):
			print(f'ERROR cc_label input {cc_label} not a valid key. Choose from {ds.accepted_labels}')
			return
	if ds.mode != 'otsu':
		if not ds.CARDIACCYCLE_CALCULATED or recalculate:
			if cc_method == 'area':
				sysdia_frames_by_area(ds, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=cc_pad_len, recalculate=recalculate, verbose=verbose,
												save_area_plot=save_cc_plot, save_dir=save_dir, show_plot=False, sys_thres=area_sys_thres, dia_thres=area_dia_thres)
			elif cc_method == 'angle':
				sysdia_frames_by_angle(ds, param, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=cc_pad_len, recalculate=recalculate, verbose=verbose,
												save_angle_plot=save_cc_plot, save_dir=save_dir, show_plot=False)
			elif cc_method == 'metadata':
				sysdia_frames_by_RTime(ds, recalculate=recalculate, verbose=verbose, rr_sys_ratio=rr_sys_ratio)
			elif cc_method == 'ecg':
				sysdia_frames_by_ecg(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			elif cc_method == 'ecg_lazy':
				sysdia_frames_by_ecg_lazy(ds, waveform_data, sampling_rate=sampling_rate, smooth_fraction=0.2, pad_len=20, recalculate=recalculate,
																	verbose=verbose, rr_sys_ratio=0.333)
			elif cc_method == 'arterial':
				sysdia_frames_by_art(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			else:
				print(f'ERROR! cc_method must be [area, angle, ecg, metadata, arterial] not {cc_method}')
				return

	safe_makedir(save_dir)
	save_name = ds.filename + '_' + param + '_' + cc_method +'_perc_line_radlong.png'
	save_path = os.path.join(save_dir, save_name)

	data_dict = calculate_3dhist_radlong(ds, param, nbins=nbins, perc_lo=perc_lo, perc_hi=perc_hi,
												av_filter_flag=av_filter_flag, av_savgol_window=av_savgol_window, av_savgol_poly=av_savgol_poly,
												verbose=verbose)

	_, _, rad_hi_arr, rad_lo_arr = data_dict['radial']
	_, _, long_hi_arr, long_lo_arr = data_dict['longitudinal']

	x = np.arange(ds.nframes)*(1000/ds.frame_rate)

	if 'ecg' in cc_method or cc_method == 'arterial':
		fig, (ax,ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, figsize=(8,6))
	else:
		fig, ax = plt.subplots(nrows=1,ncols=1)

	rad_result = calculate_radlong_peaks(ds, rad_hi_arr, rad_lo_arr, frame_times=x, smooth_fraction=smooth_fraction, pad_len=pad_len,
																			 peak_thres=peak_thres, min_dist=min_dist, pick_peak_by_subset=pick_peak_by_subset, cc_method=cc_method)
	long_result = calculate_radlong_peaks(ds, long_hi_arr, long_lo_arr, frame_times=x, smooth_fraction=smooth_fraction, pad_len=pad_len,
																			 peak_thres=peak_thres, min_dist=min_dist, pick_peak_by_subset=pick_peak_by_subset, cc_method=cc_method)

	rad_filt_hi, rad_filt_lo, rad_true_sys, rad_true_dia, rad_sys_px, rad_sys_py, rad_e_px, rad_e_py, rad_l_px, rad_l_py, rad_a_px, rad_a_py = rad_result
	long_filt_hi, long_filt_lo, long_true_sys, long_true_dia, long_sys_px, long_sys_py, long_e_px, long_e_py, long_l_px, long_l_py, long_a_px, long_a_py = long_result

	radline, = ax.plot(x, rad_filt_hi, 'r:')
	ax.plot(x, rad_filt_lo, 'r:')
	longline, = ax.plot(x, long_filt_hi, 'c:')
	ax.plot(x, long_filt_lo, 'c:')

	ax.plot(rad_sys_px, rad_sys_py, 'r+')
	ax.plot(rad_e_px, rad_e_py, 'r+')
	ax.plot(rad_l_px, rad_l_py, 'r+')
	ax.plot(rad_a_px, rad_a_py, 'r+')
	ax.plot(long_sys_px, long_sys_py, 'b+')
	ax.plot(long_e_px, long_e_py, 'b+')
	ax.plot(long_l_px, long_l_py, 'b+')
	ax.plot(long_a_px, long_a_py, 'b+')

	for xy in zip(rad_e_px, rad_e_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='r')
	for xy in zip(rad_l_px, rad_l_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='r')
	for xy in zip(rad_a_px, rad_a_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='r')
	for xy in zip(rad_sys_px, rad_sys_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, -1.5),
								 textcoords='offset points', fontsize=8, color='r')
	for xy in zip(long_e_px, long_e_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='b')
	for xy in zip(long_l_px, long_l_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='b')
	for xy in zip(long_a_px, long_a_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, 1.5),
								 textcoords='offset points', fontsize=8, color='b')
	for xy in zip(long_sys_px, long_sys_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), xycoords='data', xytext=(1.5, -1.5),
								 textcoords='offset points', fontsize=8, color='b')

	ax.set_title(param.capitalize() + ' vs Time')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel(param.capitalize() + ' (' + ds._param_unit(param) + ')')

	if 'ecg' in cc_method:
		ecg_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax2.plot(ecg_times, fix_ecg(waveform_data, sampling_rate=sampling_rate))
		ax2.set_ylabel('Voltage (mV)')
	if cc_method == 'arterial':
		art_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax2.plot(art_times, waveform_data)
		ax2.set_ylabel('Pressure (mmHg)')

	if ds.mode != 'otsu':
		if true_sysdia_mode == 'radial':
			true_sys = rad_true_sys
			true_dia = rad_true_dia
		else:
			true_sys = long_true_sys
			true_dia = long_true_dia

		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					#ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
					sys_label = ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				else:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					#ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
					dia_label = ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				else:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1

		#ax.legend(loc='lower right')
		ax.legend([radline, longline, sys_label, dia_label], ['Radial Component', 'Longitudinal Component', 'Systole', 'Diastole'], loc='lower right')
		fig.tight_layout()
		fig.savefig(save_path)
		if not show_img:
			plt.close(fig)

		if print_report:
			label='rv'
			print('=====================')
			print('RADIAL COMPONENT:')
			print('----------------')
			if len(rad_sys_py) > 0:
				print(f'Global peak systolic {label.upper()} {param}: {np.max(np.abs(rad_sys_py))}')
				print(f'Global mean systolic {label.upper()} {param}: {np.mean(np.abs(rad_sys_py))}')
				print(f'Number of cardiac cycles: {len(rad_sys_py)}')
				print('---------------------')
			if len(rad_e_py) > 0:
				print(f'Global early peak diastolic {label.upper()} {param}: {np.max(rad_e_py)}')
				print(f'Global early mean diastolic {label.upper()} {param}: {np.mean(rad_e_py)}')
				print('---------------------')
			if len(rad_l_py) > 0:
				print(f'Global diastasis peak diastolic {label.upper()} {param}: {np.max(rad_l_py)}')
				print(f'Global diastasis mean diastolic {label.upper()} {param}: {np.mean(rad_l_py)}')
				print('---------------------')
			if len(rad_a_py) > 0:
				print(f'Global late peak diastolic {label.upper()} {param}: {np.max(rad_a_py)}')
				print(f'Global late mean diastolic {label.upper()} {param}: {np.mean(rad_a_py)}')
			print('----------------')
			print('LONGITUDINAL COMPONENT:')
			print('----------------')
			if len(long_sys_py) > 0:
				print(f'Global peak systolic {label.upper()} {param}: {np.max(np.abs(long_sys_py))}')
				print(f'Global mean systolic {label.upper()} {param}: {np.mean(np.abs(long_sys_py))}')
				print(f'Number of cardiac cycles: {len(rad_sys_py)}')
				print('---------------------')
			if len(long_e_py) > 0:
				print(f'Global early peak diastolic {label.upper()} {param}: {np.max(long_e_py)}')
				print(f'Global early mean diastolic {label.upper()} {param}: {np.mean(long_e_py)}')
				print('---------------------')
			if len(long_l_py) > 0:
				print(f'Global diastasis peak diastolic {label.upper()} {param}: {np.max(long_l_py)}')
				print(f'Global diastasis mean diastolic {label.upper()} {param}: {np.mean(long_l_py)}')
				print('---------------------')
			if len(long_a_py) > 0:
				print(f'Global late peak diastolic {label.upper()} {param}: {np.max(long_a_py)}')
				print(f'Global late mean diastolic {label.upper()} {param}: {np.mean(long_a_py)}')
			print('=====================')

		if len(rad_sys_py) > 0:
			rad_peak_sys = np.max(np.abs(rad_sys_py))
			rad_mean_sys = np.mean(np.abs(rad_sys_py))
			rad_n_cycles = len(rad_sys_py)
		else:
			rad_peak_sys, rad_mean_sys = 0,0
			rad_n_cycles = 0
		if len(long_sys_py) > 0:
			long_peak_sys = np.max(np.abs(long_sys_py))
			long_mean_sys = np.mean(np.abs(long_sys_py))
			long_n_cycles = len(long_sys_py)
		else:
			long_peak_sys, long_mean_sys = 0,0
			long_n_cycles = 0

		if len(rad_e_py) > 0:
			rad_peak_e = np.max(np.abs(rad_e_py))
			rad_mean_e = np.mean(np.abs(rad_e_py))
		else:
			rad_peak_e, rad_mean_e = 0,0
		if len(rad_l_py) > 0:
			rad_peak_l = np.max(np.abs(rad_l_py))
			rad_mean_l = np.mean(np.abs(rad_l_py))
		else:
			rad_peak_l, rad_mean_l = 0,0
		if len(rad_a_py) > 0:
			rad_peak_a = np.max(np.abs(rad_a_py))
			rad_mean_a = np.mean(np.abs(rad_a_py))
		else:
			rad_peak_a, rad_mean_a = 0,0

		if len(long_e_py) > 0:
			long_peak_e = np.max(np.abs(long_e_py))
			long_mean_e = np.mean(np.abs(long_e_py))
		else:
			long_peak_e, long_mean_e = 0,0
		if len(long_l_py) > 0:
			long_peak_l = np.max(np.abs(long_l_py))
			long_mean_l = np.mean(np.abs(long_l_py))
		else:
			long_peak_l, long_mean_l = 0,0
		if len(long_a_py) > 0:
			long_peak_a = np.max(np.abs(long_a_py))
			long_mean_a = np.mean(np.abs(long_a_py))
		else:
			long_peak_a, long_mean_a = 0,0

		if return_value:
			result = (rad_peak_sys, rad_mean_sys, rad_peak_e, rad_mean_e, rad_peak_l, rad_mean_l, rad_peak_a, rad_mean_a,
								long_peak_sys, long_mean_sys,
								long_peak_e, long_mean_e, long_peak_l, long_mean_l, long_peak_a, long_mean_a,
								rad_n_cycles, long_n_cycles)
			return result

#OPTICAL FLOW GRAPH VISUALIZATIONS
def calculate_3dhist(ds, param, label, nbins=1000, percentile=99): #assumes already converted
	if not ds._validate_param(param):
		print(f'ERROR! {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if not ds._validate_label(label):
		print(f'ERROR {label} not a valid key. Choose from {ds.accepted_labels}')
		return
	arr = ds.get_masked_arr(param, label) #retrieve masked array of parameter
	mag_list = []
	ang_list = []
	for i in range(ds.nframes):
		flow = np.squeeze(arr[i,:,:,:])
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
	for i in range(ds.nframes):
		mag_frame = mag_arr[i,:,:]
		flat = np.ravel(mag_frame)
		flat_nonzero = flat[flat != 0]
		if len(flat_nonzero) == 0:
			print(f'ERROR len(flat_nonzero) is 0 for frame {i}')
			perc_hi.append(perc_hi[-1])
			mag_freq_list.append(mag_freq_list[-1])
		else:
			perc_hi.append(np.percentile(flat_nonzero, percentile))
			freq, mag_edges = np.histogram(flat_nonzero, bins=nbins, range=(mag_min, mag_max))
			mag_freq_list.append(freq+1)


	ang_freq_list = []
	ang_max = np.max(ang_arr)
	ang_min = np.min(ang_arr)
	ang_edges = []
	for i in range(ds.nframes):
		ang_frame = ang_arr[i,:,:]
		flat = np.ravel(ang_frame)
		flat_nonzero = flat[flat != 0]
		if len(flat_nonzero) == 0:
			print(f'ERROR len(flat_nonzero) is 0 for frame {i}')
			ang_freq_list.append(mag_freq_list[-1])
		else:
			freq, ang_edges = np.histogram(flat_nonzero, bins=nbins, range=(ang_min, ang_max))
			ang_freq_list.append(freq+1)

	hi_arr = np.asarray(perc_hi)
	mag_freq_arr = np.stack(mag_freq_list)
	ang_freq_arr = np.stack(ang_freq_list)
	return mag_freq_arr, ang_freq_arr, mag_edges[:-1], ang_edges[:-1], hi_arr

def plot_heatmap(ds, param, label, save_dir, nbins=1000,
								 verbose=False, waveform_data=None, sampling_rate=None,
								 show_sysdia=False, cc_method='angle', cc_label='rv_inner',
								 cc_smooth_fraction=0.2, save_cc_plot=False, area_sys_thres=0.9,
								 area_dia_thres=0.5, rr_sys_ratio=0.333,
								 show_img=False, recalculate=True):
	if not ds._validate_param(param):
		print(f'ERROR! {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if not ds._validate_label(label):
		print(f'ERROR {label} not a valid key. Choose from {ds.accepted_labels}')
		return

	safe_makedir(save_dir)
	if show_sysdia:
		save_name = ds.filename + '_' + label + '_' + param + '_' + cc_method +'_3dhist.png'
	else:
		save_name = ds.filename + '_' + label + '_' + param +'_3dhist.png'
	save_path = os.path.join(save_dir, save_name)
	if os.path.exists(save_path) and not recalculate:
		print(f'{save_path} already exists, skipping!')
		return

	if show_sysdia:
		if not ds.CARDIACCYCLE_CALCULATED or recalculate:
			if cc_method == 'area':
				sysdia_frames_by_area(ds, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_area_plot=save_cc_plot, save_dir=save_dir, show_plot=False, sys_thres=area_sys_thres, dia_thres=area_dia_thres)
			elif cc_method == 'angle':
				sysdia_frames_by_angle(ds, param, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_angle_plot=save_cc_plot, save_dir=save_dir, show_plot=False)
			elif cc_method == 'metadata':
				sysdia_frames_by_RTime(ds, recalculate=recalculate, verbose=verbose, rr_sys_ratio=rr_sys_ratio)
			elif cc_method == 'ecg':
				sysdia_frames_by_ecg(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			elif cc_method == 'ecg_lazy':
				sysdia_frames_by_ecg_lazy(ds, waveform_data, sampling_rate=sampling_rate, smooth_fraction=0.2, pad_len=20, recalculate=recalculate,
																	verbose=verbose, rr_sys_ratio=0.333)
			elif cc_method == 'arterial':
				sysdia_frames_by_art(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			else:
				print(f'ERROR! cc_method must be [area, angle, ecg, metadata, arterial] not {cc_method}')
				return

		if cc_method == 'angle':
			true_dia = []
			true_sys = ds.sys_frames
			if true_sys[0][0] > 1:
				true_dia.append([0, true_sys[0][0] - 1])
			if true_sys[-1][1] < (ds.nframes - 2):
				true_dia.append([true_sys[-1][1], ds.nframes - 1])
			for i in range(len(true_sys)-1):
				start1, stop1 = true_sys[i]
				start2, stop2 = true_sys[i+1]
				true_dia.append([stop1, start2])
		else:
			true_dia = ds.dia_frames
			true_sys = ds.sys_frames

	if waveform_data is None:
		waveform_exists = False
	else:
		waveform_exists = True

	if waveform_exists and sampling_rate is None:
		print('ERROR, must specify sampling rate if waveform data is provided!')
		return

	mag, ang, mag_edges, ang_edges, _ = calculate_3dhist(ds, param, label, nbins=nbins)

	mag_norm = LogNorm(vmin=np.min(mag), vmax=np.max(mag))
	ang_norm = LogNorm(vmin=np.min(ang), vmax=np.max(ang))

	if waveform_exists and show_sysdia:
		if 'ecg' in cc_method:
			waveform_data = fix_ecg(waveform_data, sampling_rate=sampling_rate)
		fig, (ax1, ax2, ax_t) = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=False, figsize=(8,9), layout='constrained',
																 gridspec_kw={'height_ratios': [4, 4, 1]})
		wave_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax_t.plot(wave_times, waveform_data)
		x = np.arange(ds.nframes)*(1000/ds.frame_rate)
		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1
		ax_t.legend(loc='lower right')
	elif show_sysdia and not waveform_exists:
		fig, (ax1, ax2, ax_t) = plt.subplots(nrows=3,ncols=1, sharex=True, sharey=False, figsize=(8,9), layout='constrained',
																 gridspec_kw={'height_ratios': [4, 4, 0.5]})
		x = np.arange(ds.nframes)*(1000/ds.frame_rate)
		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
				else:
					ax_t.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1
		ax_t.legend(loc='lower right')
	else:
		fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, figsize=(8,6))
	xgrid = np.arange(ds.nframes)*(1000/ds.frame_rate)
	ygrid_mag = mag_edges
	ygrid_ang = ang_edges*180/np.pi
	plt1 = ax1.pcolormesh(xgrid, ygrid_mag, mag.T, norm=mag_norm, cmap='hot')
	ax1.set_ylabel(param.capitalize() + ' (' + ds._param_unit(param) + ')')
	ax1.set_title('Magnitude of ' + param.capitalize() + ' vs Time (ms)')
	plt2 = ax2.pcolormesh(xgrid, ygrid_ang, ang.T, norm=ang_norm, cmap='viridis')
	ax2.set_ylabel('Angle (deg)')

	if not waveform_exists and not show_sysdia:
		ax2.set_xlabel('Time (ms)')
	else:
		ax_t.set_xlabel('Time (ms)')

	fig.colorbar(plt1, ax=ax1, label='log(freq)')
	fig.colorbar(plt2, ax=ax2, label='log(freq)')
	fig.savefig(save_path)
	if not show_img:
		plt.close(fig)

#AUTOMATIC OPTICAL FLOW PARAM CALCULATION AND PLOTTING
def percentile_plot(ds, param, label, save_dir, cc_method='angle', cc_label='rv_inner',
										cc_smooth_fraction=0.2, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										waveform_data=None, sampling_rate=500, pick_peak_by_subset=True,
										nbins=1000, percentile=99, smooth_fraction=0.5, pad_len=20,
										peak_thres=0.2, min_dist=5, show_all_peaks=False, show_img=False,
										print_report=False, return_value=True, recalculate=True, verbose=False, rr_sys_ratio=0.333):
	if not ds._validate_param(param):
		print(f'ERROR! param input {param} is not a valid optical flow parameter, choose from {ds.accepted_params}')
		return
	if not ds._validate_label(label):
		print(f'ERROR label input {label} not a valid key. Choose from {ds.accepted_labels}')
		return
	if cc_method == 'area' or cc_method == 'angle':
		if not ds._validate_label(cc_label):
			print(f'ERROR cc_label input {cc_label} not a valid key. Choose from {ds.accepted_labels}')
			return
	if ds.mode != 'otsu':
		if not ds.CARDIACCYCLE_CALCULATED or recalculate:
			if cc_method == 'area':
				sysdia_frames_by_area(ds, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_area_plot=save_cc_plot, save_dir=save_dir, show_plot=False, sys_thres=area_sys_thres, dia_thres=area_dia_thres)
			elif cc_method == 'angle':
				sysdia_frames_by_angle(ds, param, cc_label, smooth_fraction=cc_smooth_fraction, pad_len=20, recalculate=recalculate, verbose=verbose,
												save_angle_plot=save_cc_plot, save_dir=save_dir, show_plot=False)
			elif cc_method == 'metadata':
				sysdia_frames_by_RTime(ds, recalculate=recalculate, verbose=verbose, rr_sys_ratio=rr_sys_ratio)
			elif cc_method == 'ecg':
				sysdia_frames_by_ecg(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			elif cc_method == 'ecg_lazy':
				sysdia_frames_by_ecg_lazy(ds, waveform_data, sampling_rate=sampling_rate, smooth_fraction=0.2, pad_len=20, recalculate=recalculate,
																	verbose=verbose, rr_sys_ratio=0.333)
			elif cc_method == 'arterial':
				sysdia_frames_by_art(ds, waveform_data, sampling_rate=sampling_rate, recalculate=recalculate)
			else:
				print(f'ERROR! cc_method must be [area, angle, ecg, metadata, arterial] not {cc_method}')
				return

	safe_makedir(save_dir)
	save_name = ds.filename + '_' + label + '_' + param + '_' + str(percentile) + '_' + cc_method + '_' + 'perc_line.png'
	save_path = os.path.join(save_dir, save_name)

	_, _, _, _, hi_arr = calculate_3dhist(ds, param, label, nbins=nbins, percentile=percentile)
	x = np.arange(ds.nframes)*(1000/ds.frame_rate)
	if 'ecg' in cc_method or cc_method == 'arterial':
		fig, (ax,ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, figsize=(8,6))
	else:
		fig, ax = plt.subplots(nrows=1,ncols=1)

	smoother_mag = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
	smoother_mag.smooth(hi_arr)
	filt_arr = smoother_mag.smooth_data[0]

	peaks_i = peakutils.peak.indexes(filt_arr, thres=peak_thres, min_dist=min_dist)

	sys_i = []
	e_i = []
	l_i = []
	a_i = []
	true_sys = []
	for start, stop in ds.sys_frames:
		if pick_peak_by_subset:
			candidate_i = peakutils.peak.indexes(filt_arr[start:stop+1], thres=peak_thres, min_dist=min_dist) + start
		else:
			candidate_i = [k for k in peaks_i if ((k >= start) and (k <= stop))]
		if len(candidate_i) > 0:
			candidate_y = [filt_arr[i] for i in candidate_i]
			max_idx = np.argmax(candidate_y)
			sys_i.append(candidate_i[max_idx])
			true_sys.append([start, stop])
		else:
			print('Warning no sys peak found! Using max value')
			sys_i.append(np.argmax(filt_arr[start:stop])+start)

	if cc_method == 'angle':
		true_dia = []
		if true_sys[0][0] > 1:
			true_dia.append([0, true_sys[0][0] - 1])
		if true_sys[-1][1] < (ds.nframes - 2):
			true_dia.append([true_sys[-1][1], ds.nframes - 1])
		for i in range(len(true_sys)-1):
			start1, stop1 = true_sys[i]
			start2, stop2 = true_sys[i+1]
			true_dia.append([stop1, start2])
	else:
		true_dia = ds.dia_frames
		true_sys = ds.sys_frames

	for start, stop in true_dia:
		e_start = int(start)
		e_stop = int(start + np.floor((stop-start)/3))
		l_start = int(e_stop + 1)
		l_stop = int(l_start + np.floor((stop-start)/3))
		a_start = int(l_stop + 1)
		a_stop = int(stop+1)
		if pick_peak_by_subset:
			e_candidate_i = peakutils.peak.indexes(filt_arr[e_start:e_stop+1], thres=peak_thres, min_dist=min_dist) + e_start
			l_candidate_i = peakutils.peak.indexes(filt_arr[l_start:l_stop+1], thres=peak_thres, min_dist=min_dist) + l_start
			a_candidate_i = peakutils.peak.indexes(filt_arr[a_start:a_stop+1], thres=peak_thres, min_dist=min_dist) + a_start
		else:
			e_candidate_i = [k for k in peaks_i if ((k >= e_start) and (k <= (e_stop)))]
			l_candidate_i = [k for k in peaks_i if ((k >= l_start) and (k <= (l_stop)))]
			a_candidate_i = [k for k in peaks_i if ((k >= a_start) and (k <= (a_stop)))]
		if len(e_candidate_i) > 0:
			e_candidate_y = [filt_arr[i] for i in e_candidate_i]
			e_index = np.argmax(e_candidate_y)
			e_i.append(e_candidate_i[e_index])
		else:
			print('Warning no e\' peak found! Using max value')
			e_i.append(np.argmin(filt_arr[e_start:e_stop])+e_start)
		if len(l_candidate_i) > 0:
			l_candidate_y = [filt_arr[i] for i in l_candidate_i]
			l_index = np.argmax(l_candidate_y)
			l_i.append(l_candidate_i[l_index])
		else:
			print('Warning no l\' peak found! Using max value')
			l_i.append(np.argmin(filt_arr[l_start:l_stop])+l_start)
		if len(a_candidate_i) > 0:
			a_candidate_y = [filt_arr[i] for i in a_candidate_i]
			a_index = np.argmax(a_candidate_y)
			a_i.append(a_candidate_i[a_index])
		else:
			print('Warning no a\' peak found! Using max value')
			a_i.append(np.argmin(filt_arr[a_start:a_stop])+a_start)

	if show_all_peaks:
		px = x[peaks_i]
		py = filt_arr[peaks_i]
	else:
		sys_px = x[sys_i]
		sys_py = filt_arr[sys_i]
		e_px = x[e_i]
		e_py = filt_arr[e_i]
		l_px = x[l_i]
		l_py = filt_arr[l_i]
		a_px = x[a_i]
		a_py = filt_arr[a_i]

	ax.plot(x, filt_arr)
	ax.plot(sys_px, sys_py, 'r+')
	ax.plot(e_px, e_py, 'b+')
	ax.plot(l_px, l_py, 'b+')
	ax.plot(a_px, a_py, 'b+')

	if 'ecg' in cc_method:
		ecg_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax2.plot(ecg_times, waveform_data)
		ax2.set_ylabel('Voltage (mV)')
	if cc_method == 'arterial':
		art_times = np.arange(waveform_data.size)*(1000/sampling_rate)
		ax2.plot(art_times, waveform_data)
		ax2.set_ylabel('Pressure (mmHg)')

	for xy in zip(sys_px, sys_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), fontsize=8)
	for xy in zip(e_px, e_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), fontsize=8)
	for xy in zip(l_px, l_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), fontsize=8)
	for xy in zip(a_px, a_py):
		ax.annotate('%.2f' % xy[1], xy=(xy[0], xy[1]), fontsize=8)
	ax.set_title(label.upper() + ' ' + param.capitalize() + ' vs Time')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel(param.capitalize() + ' (' + ds._param_unit(param) + ')')

	if ds.mode != 'otsu':
		if len(true_sys) > 0:
			counter = 0
			for start, stop in true_sys:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5, label='Systole')
				else:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.5)
				counter += 1

		if len(true_dia) > 0:
			counter = 0
			for start, stop in true_dia:
				if stop >= ds.nframes:
					stop = ds.nframes - 1 #sanity check
				if counter == 0:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25, label='Diastole')
				else:
					ax.axvspan(x[start], x[stop], facecolor='0.8', alpha=0.25)
				counter += 1

		ax.legend(loc='lower right')
		fig.tight_layout()
		fig.savefig(save_path)
		if not show_img:
			plt.close(fig)

		if print_report:
			print('=====================')
			if len(sys_py) > 0:
				print(f'Global peak systolic {label.upper()} {param}: {np.max(sys_py)}')
				print(f'Global mean systolic {label.upper()} {param}: {np.mean(sys_py)}')
				print(f'Number of cardiac cycles: {len(sys_py)}')
				print('---------------------')
			if len(e_py) > 0:
				print(f'Global peak early diastolic {label.upper()} {param}: {np.max(e_py)}')
				print(f'Global mean early diastolic {label.upper()} {param}: {np.mean(e_py)}')
				print('---------------------')
			if len(l_py) > 0:
				print(f'Global peak diastasis diastolic {label.upper()} {param}: {np.max(e_py)}')
				print(f'Global mean diastasis diastolic {label.upper()} {param}: {np.mean(e_py)}')
				print('---------------------')
			if len(a_py) > 0:
				print(f'Global peak late diastolic {label.upper()} {param}: {np.max(a_py)}')
				print(f'Global mean late diastolic {label.upper()} {param}: {np.mean(a_py)}')
			print('=====================')
		if len(sys_py) > 0:
			peak_sys = np.max(sys_py)
			mean_sys = np.mean(sys_py)
			n_cycles = len(sys_py)
		else:
			peak_sys, mean_sys = 0,0
			n_cycles = 0

		if len(e_py) > 0:
			peak_e = np.max(e_py)
			mean_e = np.mean(e_py)
		else:
			peak_e, mean_e = 0,0
		if len(l_py) > 0:
			peak_l = np.max(l_py)
			mean_l = np.mean(l_py)
		else:
			peak_l, mean_l = 0,0
		if len(a_py) > 0:
			peak_a = np.max(a_py)
			mean_a = np.mean(a_py)
		else:
			peak_a, mean_a = 0,0

		if return_value:
			if len(sys_py) == 0:
				print(f'ERROR not complete cardiac cycle: systolic cycles={len(sys_py)}')
			return peak_sys, mean_sys, peak_e, mean_e, peak_l, mean_l, peak_a, mean_a, n_cycles

def analyze_hdf5_folder(hdf5_folder, save_dir, param_list, label_list, nchunks=10, chunk_index=0,
												cc_label='rv_inner', recalculate=False, save_mp4=False, verbose=True, produce_auxiliary_plots=False):
	file_list = os.listdir(hdf5_folder)
	total_files = len(file_list)
	split_size = total_files // nchunks

	error_list = []
	for i in tqdm(range(chunk_index*split_size, (chunk_index+1)*split_size), disable=verbose):
		filename = file_list[i]
		try:
			if verbose:
				print(f'Processing file {i}/{(chunk_index+1)*split_size}:')
			if filename[-4:] == 'hdf5':
				filepath = os.path.join(hdf5_folder, filename)
				dataset_loaded = False
				for param in param_list:
					for label in label_list:
						save_subdir = os.path.join(save_dir, param + '_' + label)
						safe_makedir(save_subdir)
						pkl_dir = os.path.join(save_subdir, 'pkl_files')
						plot_dir = os.path.join(save_subdir, 'plots')
						safe_makedir(pkl_dir)
						safe_makedir(plot_dir)
						savename = filename[:-5] + '_' + label + '_' + param + '_data.pkl'
						save_path = os.path.join(pkl_dir, savename)
						if save_mp4:
							mp4_dir = os.path.join(save_subdir, 'mp4')
							safe_makedir(mp4_dir)

						if os.path.exists(save_path) and not recalculate:
							print(f'{save_path} already exists, skipping!')
							continue
						else:
							if not dataset_loaded:
								if verbose:
									print(f'Analyzing file: {filepath}')
								ds = OpticalFlowDataset(filepath)
								dataset_loaded = True
							if save_mp4:
								if verbose:
									print('Producing mp4 visualization...')
								visualize(ds, param, label, mp4_dir, resize_factor=1, speed_factor=0.25, verbose=verbose, recalculate=False)

							if verbose:
								print('Calculating peak info and producing plots...')
							if ds.waveforms_present:
								try:
									if verbose:
										print(f'Calculating ECG gated total results...')
									ecg_total_result = percentile_plot(ds, param, label, plot_dir, cc_method='ecg_lazy', cc_label='rv_inner',
										cc_smooth_fraction=0.2, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										waveform_data=ds.ecg, sampling_rate=ds.ecg_sampling_rate, pick_peak_by_subset=True,
										nbins=1000, percentile=99, smooth_fraction=0.5, pad_len=20,
										peak_thres=0.05, min_dist=3, show_all_peaks=False, show_img=False,
										print_report=True, return_value=True, recalculate=True, verbose=verbose, rr_sys_ratio=0.333)

								except Exception:
									if verbose:
										traceback.print_exc()
									ecg_total_result = [0,0,0,0,0,0,0,0,0]
									print(f'An error with ECG processing of {param} {label}, skipping!')

								try:
									if verbose:
										print(f'Calculating ECG gated radlong results...')
									ecg_radlong_result = percentile_plot_radlong(ds, param, plot_dir, cc_method='ecg_lazy', cc_label='rv_inner', true_sysdia_mode='radial',
										cc_smooth_fraction=0.2, cc_pad_len=20, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4, perc_lo=1, perc_hi=99,
										waveform_data=ds.ecg, sampling_rate=ds.ecg_sampling_rate, pick_peak_by_subset=True,
										nbins=1000, smooth_fraction=0.5, pad_len=20,
										peak_thres=0.05, min_dist=3, show_all_peaks=False, show_img=False,
										print_report=True, return_value=True, recalculate=False, verbose=verbose, rr_sys_ratio=0.333)

								except Exception:
									if verbose:
										traceback.print_exc()
									ecg_radlong_result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
									print(f'An error with ECG processing of component analysis of {param} {label}, skipping!')

								try:
									if verbose:
										print(f'Calculating ART gated total results...')
									art_total_result = percentile_plot(ds, param, label, plot_dir, cc_method='arterial', cc_label='rv_inner',
										cc_smooth_fraction=0.2, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										waveform_data=ds.art, sampling_rate=ds.art_sampling_rate, pick_peak_by_subset=True,
										nbins=1000, percentile=99, smooth_fraction=0.5, pad_len=20,
										peak_thres=0.05, min_dist=3, show_all_peaks=False, show_img=False,
										print_report=True, return_value=True, recalculate=True, verbose=verbose, rr_sys_ratio=0.333)

								except Exception:
									if verbose:
										traceback.print_exc()
									art_total_result = [0,0,0,0,0,0,0,0,0]
									print(f'An error with ART processing of {param} {label}, skipping!')

								try:
									if verbose:
										print(f'Calculating ART gated radlong results...')
									art_radlong_result = percentile_plot_radlong(ds, param, plot_dir, cc_method='arterial', cc_label='rv_inner', true_sysdia_mode='radial',
										cc_smooth_fraction=0.2, cc_pad_len=20, save_cc_plot=False, area_sys_thres=0.9, area_dia_thres=0.5,
										av_filter_flag=True, av_savgol_window=10, av_savgol_poly=4, perc_lo=1, perc_hi=99,
										waveform_data=ds.art, sampling_rate=ds.art_sampling_rate, pick_peak_by_subset=True,
										nbins=1000, smooth_fraction=0.5, pad_len=20,
										peak_thres=0.05, min_dist=3, show_all_peaks=False, show_img=False,
										print_report=True, return_value=True, recalculate=True, verbose=verbose, rr_sys_ratio=0.333)
								except Exception:
									if verbose:
										traceback.print_exc()
									art_radlong_result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
									print(f'An error with ART processing of component analysis of {param} {label}, skipping!')

								mrn = ds.ID
								frame_rate = ds.frame_rate
								pixel_spacing = ds.pixel_spacing
								hr = ds.ds_OF.attrs['HR']
								nframes = ds.nframes
								mean_art = np.mean(ds.art)
								peak_art = np.max(ds.art)
								min_art = np.min(ds.art)
								if ds.cvp_exists:
									mean_cvp = np.mean(ds.cvp)
									peak_cvp = np.max(ds.cvp)
									min_cvp = np.min(ds.cvp)
								else:
									mean_cvp, peak_cvp, min_cvp = 0, 0, 0
								if ds.pap_exists:
									mean_pap = np.mean(ds.pap)
									peak_pap = np.max(ds.pap)
									min_pap = np.min(ds.pap)
								else:
									mean_pap, peak_pap, min_pap = 0, 0, 0

								metadata = [filename, mrn, frame_rate, pixel_spacing, hr, nframes, mean_art, peak_art, min_art, mean_cvp, peak_cvp, min_cvp, mean_pap, peak_pap, min_pap]
								aggregated_results = metadata + list(ecg_total_result) + list(art_total_result) + list(ecg_radlong_result) + list(art_radlong_result)
								if verbose:
									print(f'metadata: {len(metadata)} | ecg_total_result: {len(ecg_total_result)} | art_total_result: {len(art_total_result)}')
									print(f'ecg_radlong_result: {len(ecg_radlong_result)} | art_radlong_result: {len(art_radlong_result)}')
									print(f'Total length of data list: {len(aggregated_results)}')
							pkl.dump(aggregated_results, open(os.path.join(pkl_dir, savename), 'wb'))
							if verbose:
								print(f'Saved data to file: {os.path.join(pkl_dir, savename)}')

		except Exception:
			if verbose:
				traceback.print_exc()
			print(f'An error occurs processing this hdf5 file: {filepath}')
			if filepath not in error_list:
				error_list.append(filepath)
			continue


	error_dir = os.path.join(save_dir, 'errors')
	safe_makedir(error_dir)
	pkl.dump(error_list, open(os.path.join(error_dir, 'error_filelist.pkl'), 'wb'))
	print(f'Total files unable to be processed: {len(error_list)}')
	print(f'Files unable to be processed: {error_list}')

def aggregate_pkl(param_list, label_list, save_dir):
	for param in param_list:
		for label in label_list:
			save_subdir = os.path.join(save_dir, param + '_' + label)
			pkl_dir = os.path.join(save_subdir, 'pkl_files')
			csv_dir = os.path.join(save_dir, 'csv')
			safe_makedir(csv_dir)
			file_list = os.listdir(pkl_dir)
			data_list = []
			print(f'Analyzing {pkl_dir}')
			for filename in tqdm(file_list):
				if filename[-3:] == 'pkl':
					pkl_path = os.path.join(pkl_dir, filename)
					data = pkl.load(open(pkl_path, 'rb'))
					#print(filename, len(data))
					data_list.append(data)

			header = ['Filename', 'MRN', 'FrameRate', 'PixelSpacing',	'HR', 'Frames', 'MeanART', 'MaxART','MinART',
								'MeanCVP', 'MaxCVP', 'MinCVP', 'MeanPAP', 'MaxPAP', 'MinPAP',
								f'ECGTotalPeakSystolic{param.capitalize()}', f'ECGTotalMeanSystolic{param.capitalize()}',
								f'ECGTotalPeakE{param.capitalize()}', f'ECGTotalMeanE{param.capitalize()}',
								f'ECGTotalPeakL{param.capitalize()}', f'ECGTotalMeanL{param.capitalize()}',
								f'ECGTotalPeakA{param.capitalize()}', f'ECGTotalMeanA{param.capitalize()}', f'ECGCardiacCycles{param.capitalize()}',
								f'ARTTotalPeakSystolic{param.capitalize()}', f'ARTTotalMeanSystolic{param.capitalize()}',
								f'ARTTotalPeakE{param.capitalize()}', f'ARTTotalMeanE{param.capitalize()}',
								f'ARTTotalPeakL{param.capitalize()}', f'ARTTotalMeanL{param.capitalize()}',
								f'ARTTotalPeakA{param.capitalize()}', f'ARTTotalMeanA{param.capitalize()}',f'ARTCardiacCycles{param.capitalize()}',
								f'ECGRadialPeakSystolic{param.capitalize()}', f'ECGRadialMeanSystolic{param.capitalize()}',
								f'ECGRadialPeakE{param.capitalize()}', f'ECGRadialMeanE{param.capitalize()}',
								f'ECGRadialPeakL{param.capitalize()}', f'ECGRadialMeanL{param.capitalize()}',
								f'ECGRadialPeakA{param.capitalize()}', f'ECGRadialMeanA{param.capitalize()}',
								f'ECGLongPeakSystolic{param.capitalize()}', f'ECGLongMeanSystolic{param.capitalize()}',
								f'ECGLongPeakE{param.capitalize()}', f'ECGLongMeanE{param.capitalize()}',
								f'ECGLongPeakL{param.capitalize()}', f'ECGLongMeanL{param.capitalize()}',
								f'ECGLongPeakA{param.capitalize()}', f'ECGLongMeanA{param.capitalize()}',
								f'ECGRadialCardiacCycles{param.capitalize()}', f'ECGLongCardiacCycles{param.capitalize()}',
								f'ARTRadialPeakSystolic{param.capitalize()}', f'ARTRadialMeanSystolic{param.capitalize()}',
								f'ARTRadialPeakE{param.capitalize()}', f'ARTRadialMeanE{param.capitalize()}',
								f'ARTRadialPeakL{param.capitalize()}', f'ARTRadialMeanL{param.capitalize()}',
								f'ARTRadialPeakA{param.capitalize()}', f'ARTRadialMeanA{param.capitalize()}',
								f'ARTLongPeakSystolic{param.capitalize()}', f'ARTLongMeanSystolic{param.capitalize()}',
								f'ARTLongPeakE{param.capitalize()}', f'ARTLongMeanE{param.capitalize()}',
								f'ARTLongPeakL{param.capitalize()}', f'ARTLongMeanL{param.capitalize()}',
								f'ARTLongPeakA{param.capitalize()}', f'ARTLongMeanA{param.capitalize()}',
								f'ARTRadialCardiacCycles{param.capitalize()}', f'ARTLongCardiacCycles{param.capitalize()}']

			df = pl.DataFrame(data_list, schema=header, orient='row')
			csv_name = label + '_' + param + '_data.csv'
			csv_path = os.path.join(csv_dir, csv_name)
			print(f'Saving csv file as {csv_path}')
			df.write_csv(csv_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nchunks', type=int)
	parser.add_argument('--hdf5_folder', type=str)
	parser.add_argument('--save_folder', type=str)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--recalculate', action='store_true')
	args = parser.parse_args()
	for i in range(args.nchunks):
		hdf5_dir = os.path.join(args.hdf5_folder, f'chunk{i}')
		save_dir = os.path.join(args.save_folder, f'chunk{i}')
		param_list = ['velocity']
		label_list = ['rv']
		analyze_hdf5_folder(hdf5_dir, save_dir, param_list, label_list, nchunks=1, chunk_index=0,
											cc_label='rv_inner', recalculate=args.recalculate, save_mp4=False,
											verbose=args.verbose, produce_auxiliary_plots=False)

	if args.verbose:
		print('Merging results of chunks...')
	merged_dir = os.path.join(args.save_folder, 'merged')
	safe_makedir(merged_dir)
	param_list = ['velocity_rv']
	subdir_list = ['pkl_files']
	total_error_list = []
	for i in range(args.nchunks):
		print(f'Processing chunk {i}')
		error_path = os.path.join(os.path.join(os.path.join(args.save_folder, f'chunk{i}'), 'errors'), 'error_filelist.pkl')
		error_list = pkl.load(open(error_path, 'rb'))
		print(f'Loaded error pickle file: {error_path}')
		total_error_list += error_list

		for param in param_list:
			for subdir in subdir_list:
				old_dir = os.path.join(os.path.join(os.path.join(args.save_folder, f'chunk{i}'), param), subdir)
				new_dir = os.path.join(os.path.join(merged_dir, param), subdir)
				safe_makedir(new_dir)
				file_list = os.listdir(old_dir)
				for filename in tqdm(file_list):
					old_path = os.path.join(old_dir, filename)
					new_path = os.path.join(new_dir, filename)
					if os.path.exists(new_path):
						print(f'Skipping! File exists: {new_path}')
						continue
					else:
						copy(old_path, new_path)

				print(f'Number of files in old_dir: {len(os.listdir(old_dir))}')
				print(f'Number of files in new_dir: {len(os.listdir(new_dir))}')

	pkl_savepath = os.path.join(merged_dir, 'total_error_filelist.pkl')
	pkl.dump(total_error_list, open(pkl_savepath, 'wb'))
	
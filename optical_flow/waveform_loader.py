"""
Waveform loading and validation utilities.

This module provides functions for loading and validating physiological waveform data
(ECG, arterial pressure, CVP, PAP) from numpy files.
"""

import os
import numpy as np
from typing import Optional, Dict, Tuple
from optical_flow.config import OpticalFlowCalculationConfig, default_optical_flow_config


def _load_waveform_file(path: str) -> Optional[np.ndarray]:
	"""
	Load waveform data from numpy file.
	
	Args:
		path: Path to .npy file
	
	Returns:
		Waveform array or None if file doesn't exist
	"""
	if not os.path.exists(path):
		return None
	try:
		return np.load(path)
	except (IOError, ValueError) as e:
		print(f'Error loading waveform from {path}: {e}')
		return None


def _validate_waveform_flatness(waveform: np.ndarray, threshold: float) -> bool:
	"""
	Check if waveform is flat (has no variation).
	
	Args:
		waveform: Waveform array
		threshold: Maximum gradient threshold
	
	Returns:
		True if waveform is flat, False otherwise
	"""
	return np.max(np.gradient(waveform)) < threshold


def _validate_waveform_range(waveform: np.ndarray, min_val: float, max_val: float,
                            name: str) -> Tuple[bool, str]:
	"""
	Validate waveform is within expected range.
	
	Args:
		waveform: Waveform array
		min_val: Minimum allowed mean value
		max_val: Maximum allowed mean value
		name: Waveform name for error messages
	
	Returns:
		Tuple of (is_valid, error_message)
	"""
	mean_val = np.mean(waveform)
	if mean_val > max_val:
		return False, f'{name} waveform is too high, mean > {max_val}mmHg!'
	if mean_val < min_val:
		return False, f'{name} waveform is too negative, mean < {min_val}mmHg!'
	return True, ''


def load_all_waveforms(dcm_path: str, waveform_folder: str,
                      config: Optional[OpticalFlowCalculationConfig] = None,
                      verbose: bool = False) -> Dict[str, Tuple[bool, Optional[np.ndarray]]]:
	"""
	Load and validate all waveform files for a DICOM file.
	
	Args:
		dcm_path: Path to DICOM file
		waveform_folder: Folder containing waveform files
		config: Optional configuration object
		verbose: Whether to print progress
	
	Returns:
		Dictionary with keys: 'ecg', 'art', 'cvp', 'pap'
		Values are tuples of (exists_and_valid, waveform_data)
	"""
	if config is None:
		config = default_optical_flow_config()
	
	base_name = os.path.basename(dcm_path)[:-4]  # Remove .dcm extension
	
	# Construct paths
	ecg_path = os.path.join(waveform_folder, base_name + '_II.npy')
	art_path = os.path.join(waveform_folder, base_name + '_ART.npy')
	abp_path = os.path.join(waveform_folder, base_name + '_ABP.npy')
	pap_path = os.path.join(waveform_folder, base_name + '_PAP.npy')
	cvp_path = os.path.join(waveform_folder, base_name + '_CVP.npy')
	
	results = {
		'ecg': (False, None),
		'art': (False, None),
		'cvp': (False, None),
		'pap': (False, None)
	}
	
	# Load PAP
	if os.path.exists(pap_path):
		pap = _load_waveform_file(pap_path)
		if pap is not None:
			# Check flatness
			if _validate_waveform_flatness(pap, config.waveform_flatness_threshold):
				if verbose:
					print('ERROR PAP waveform is flat!')
			# Check range
			elif np.mean(pap) > config.pap_max_mean:
				if verbose:
					print(f'ERROR PAP waveform is too high, mean > {config.pap_max_mean}mmHg!')
			elif np.mean(pap) < 0:
				if verbose:
					print('ERROR PAP waveform is negative, mean < 0mmHg!')
			else:
				results['pap'] = (True, pap)
	
	# Load CVP
	if os.path.exists(cvp_path):
		cvp = _load_waveform_file(cvp_path)
		if cvp is not None:
			# Check range
			is_valid, error_msg = _validate_waveform_range(cvp, config.cvp_min_mean,
			                                               config.cvp_max_mean, 'CVP')
			if not is_valid:
				if verbose:
					print(f'ERROR {error_msg}')
			else:
				results['cvp'] = (True, cvp)
	
	# Load ECG
	if os.path.exists(ecg_path):
		ecg = _load_waveform_file(ecg_path)
		if ecg is not None:
			results['ecg'] = (True, ecg)
			if verbose:
				print('Loaded ECG!')
		else:
			if verbose:
				print(f'ERROR {ecg_path} doesnt exist! No ECG waveform detected')
	else:
		if verbose:
			print(f'ERROR {ecg_path} doesnt exist! No ECG waveform detected')
	
	# Load ART (with fallback to ABP)
	art = None
	if os.path.exists(art_path):
		art = _load_waveform_file(art_path)
		if art is not None:
			if _validate_waveform_flatness(art, config.waveform_flatness_threshold):
				# Try ABP as fallback
				if os.path.exists(abp_path):
					art = _load_waveform_file(abp_path)
					if art is not None:
						if _validate_waveform_flatness(art, config.waveform_flatness_threshold):
							if verbose:
								print('ERROR ART and ABP waveforms given are flat!')
							art = None
						else:
							results['art'] = (True, art)
				else:
					if verbose:
						print('ERROR ART is flat and ABP doesnt exist!')
			else:
				results['art'] = (True, art)
	else:
		# Try ABP directly
		if os.path.exists(abp_path):
			art = _load_waveform_file(abp_path)
			if art is not None:
				if _validate_waveform_flatness(art, config.waveform_flatness_threshold):
					if verbose:
						print('ERROR ART and ABP waveforms given are flat!')
				else:
					results['art'] = (True, art)
		else:
			if verbose:
				print('ERROR ART and ABP path doesnt exist!')
	
	return results


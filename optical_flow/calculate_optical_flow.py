import cv2
import numpy as np
import os
import pydicom as dcm
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import h5py
import traceback
from typing import Optional, Tuple, Dict, List, Any
import json
import argparse
from argparse import Namespace

# PyTorch packages
import torch
from torchvision import transforms
from PIL import Image

# SAM model
from models.sam import sam_model_registry

# Utilities
from optical_flow_utils import *
from optical_flow.config import OpticalFlowCalculationConfig, default_optical_flow_config
from optical_flow.waveform_loader import load_all_waveforms
from optical_flow.exceptions import DICOMReadError, OpticalFlowCalculationError, ConfigurationError
import logging

# Setup logging
logger = logging.getLogger(__name__)

#UTILITY FUNCTIONS

# Function to evaluate a single image slice
def evaluate_1_slice(nparr: np.ndarray, model: torch.nn.Module) -> np.ndarray:
	# Load the image
	img = Image.fromarray(nparr).convert('RGB')
	orig_size = img.size

	# Resize the image to 1024x1024
	img = transforms.Resize((1024, 1024))(img)

	# Transform the image to a tensor and normalize
	transform_img = transforms.Compose([
			transforms.ToTensor(),
	])
	img = transform_img(img)
	imgs = torch.unsqueeze(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img), 0).cuda()

	# Perform model inference without gradient calculation
	with torch.no_grad():
		# Get image embeddings from the image encoder
		img_emb = model.image_encoder(imgs)

		# Get sparse and dense embeddings from the prompt encoder
		sparse_emb, dense_emb = model.prompt_encoder(
				points=None,
				boxes=None,
				masks=None,
		)

		# Get the prediction from the mask decoder
		pred, _ = model.mask_decoder(
				image_embeddings=img_emb,
				image_pe=model.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_emb,
				dense_prompt_embeddings=dense_emb,
				multimask_output=True,
		)

		# Get the most likely prediction
		pred = pred.argmax(dim=1)
		mask_pred = ((pred).cpu()).float() # this is of shape (N, C, H, W)
		pil_mask = Image.fromarray(np.array(mask_pred[0], dtype=np.uint8), 'L').resize(orig_size, resample=Image.NEAREST)
		np_mask = np.asarray(pil_mask, dtype=np.uint8)
	return np_mask #(H, W) but with multiclass

def moving_avg_mask(arr: np.ndarray, n: int = 4, threshold: float = 0.49,
                   config: Optional[OpticalFlowCalculationConfig] = None) -> np.ndarray:
	"""
	Apply moving average filter to mask array.
	
	Args:
		arr: Input array
		n: Window size for moving average
		threshold: Threshold for binarization
		config: Optional configuration object (overrides n and threshold if provided)
	
	Returns:
		Binarized array after moving average
	"""
	if config is not None:
		n = config.moving_avg_window
		threshold = config.moving_avg_threshold
	arr2 = np.vstack((np.expand_dims(arr[0,:,:], axis=0), arr, np.expand_dims(arr[-1,:,:], axis=0), np.expand_dims(arr[-1,:,:], axis=0)))
	sum = np.cumsum(arr2.astype(float), axis=0)
	sum[n:,:,:] = sum[n:,:,:] - sum[:-n,:,:]
	avg = sum[n-1:,:,:]/n
	return avg > threshold

def clean_mask(arr: np.ndarray, mode: str = 'A4C', verbose: bool = False,
              config: Optional[OpticalFlowCalculationConfig] = None) -> Optional[Dict[str, np.ndarray]]:
	"""
	Clean and process mask array.
	
	Args:
		arr: Mask array of shape (N, H, W)
		mode: Processing mode ('A4C', 'RVIO_2class', 'MouseRV_A4C')
		verbose: Whether to print progress
		config: Optional configuration object
	
	Returns:
		Dictionary of cleaned masks or None if mode is invalid
	"""
	if config is None:
		config = default_optical_flow_config()
	if verbose:
		logger.info('====Cleaning masks====')
		logger.info('Turning into 1-hot encodings...')
	if mode == 'A4C':
		mask_dict = {
			'lv_inner': arr == 1,
			'lv': arr == 2,
			'la_inner': arr == 3,
			'la': arr == 4,
			'rv_inner': arr == 5,
			'ra_inner': arr == 6,
			'rv': arr == 7,
			'ra': arr == 8
		}
	elif mode == 'RVIO_2class':
		mask_dict = {
				'rv': arr == 1,
				'av': arr == 2
		}
	elif mode == 'MouseRV_A4C':
		mask_dict = {
				'rv': arr == 1,
				'rv_inner': arr == 2
		}
	else:
		error_msg = f'mode={mode} not supported, must be [A4C, RVIO_2class, MouseRV_A4C]!'
		if verbose:
			logger.error(error_msg)
		return None
	aggregate_mask = np.zeros_like(arr)
	for k in mask_dict.keys():
		mask_arr = mask_dict[k]
		mask_arr = moving_avg_mask(np.squeeze(mask_arr))
		nframes = mask_arr.shape[0]
		frame_list = []
		for i in tqdm(range(nframes), disable=(not verbose)):
			clean = remove_small_objects(binary_fill_holes(mask_arr[i, :, :]), min_size=config.min_mask_size)
			frame_list.append(clean)

		clean_arr = np.stack(frame_list)
		aggregate_mask = np.logical_or(clean_arr, aggregate_mask)
		clean_arr_2c = np.repeat(clean_arr[:,:,:,np.newaxis], 2, axis=3)
		if verbose:
			logger.debug(f'For mask {k}, produced cleaned mask arr of shape {clean_arr_2c.shape}')
		mask_dict[k] = clean_arr_2c

	if verbose:
		logger.info('calculating bkgd...')
	bkgd_1c = np.logical_not(aggregate_mask)
	bkgd_2c = np.repeat(bkgd_1c[:, :, :, np.newaxis], 2, axis=3)
	if verbose:
		logger.debug(f'For bkgd mask, produced cleaned mask arr of shape {bkgd_2c.shape}')
	mask_dict['bkgd'] = bkgd_2c
	return mask_dict

def predict_movie_thres(nparr: np.ndarray, verbose: bool = False,
                       config: Optional[OpticalFlowCalculationConfig] = None) -> Dict[str, np.ndarray]:
	"""
	Predict masks using Otsu thresholding.
	
	Args:
		nparr: Input image array
		verbose: Whether to print progress
		config: Optional configuration object
	
	Returns:
		Dictionary with 'otsu' key containing thresholded masks
	"""
	if config is None:
		config = default_optical_flow_config()
	nframes = nparr.shape[0]
	mask_list = []
	if verbose:
		logger.info('Predicting frames...')
	for i in tqdm(range(nframes), disable=(not verbose)):
		img_arr = rgb2gray(np.squeeze(nparr[i, :, :, :]))
		mask = img_arr > threshold_otsu(img_arr)
		mask = remove_small_objects(binary_fill_holes(mask), min_size=config.min_mask_size)
		mask_list.append(mask)
	mask_arr = np.stack(mask_list)
	mask_arr = moving_avg_mask(np.squeeze(mask_arr))
	mask_arr_2c = np.repeat(mask_arr[:,:,:,np.newaxis], 2, axis=3)
	if verbose:
		logger.info(f'Produced thresholded mask of shape {mask_arr_2c.shape}')
	return {'otsu':mask_arr_2c}

def predict_movie(nparr: np.ndarray, model: torch.nn.Module, mode: str = 'A4C',
                 verbose: bool = False, config: Optional[OpticalFlowCalculationConfig] = None) -> Dict[str, np.ndarray]:
	"""
	Predict masks using SAM model.
	
	Args:
		nparr: Input image array
		model: SAM model for segmentation
		mode: Processing mode
		verbose: Whether to print progress
		config: Optional configuration object
	
	Returns:
		Dictionary of cleaned masks
	"""
	if config is None:
		config = default_optical_flow_config()
	nframes = nparr.shape[0]
	pred_list = []
	if verbose:
		logger.info('Predicting frames...')
	for i in tqdm(range(nframes), disable=(not verbose)):
		pred = evaluate_1_slice(nparr[i,:,:,:], model)
		pred_list.append(pred)

	mask_arr = np.stack(pred_list)
	return clean_mask(mask_arr, mode, verbose, config=config)

def process_folder(dcm_folder: str, save_folder: str, segmentor_model: torch.nn.Module,
                  nchunks: int = 10, chunk_index: int = 0,
                  mode: str = 'RVIO_2class', bkgd_comp: str = 'none', flipLR: bool = False,
                  verbose: bool = True, recalculate: bool = False, no_saliency: bool = True,
                  OF_algo: str = 'TVL1', save_mask_subset: Optional[List[str]] = None,
                  include_waveforms: bool = False, waveform_folder: Optional[str] = None,
                  pixel_spacing: Optional[float] = None, frame_rate: Optional[float] = None,
                  process_subset: bool = False, file_subset_list: List[str] = []) -> None:
	safe_makedir(save_folder)
	file_list = os.listdir(dcm_folder)

	if process_subset:
		if len(file_subset_list) == 0:
			print('ERROR! File subset list is empty!')
			return
		if verbose:
			print(f'Processing subset! Found {len(file_subset_list)} files in subset list provided!')
		file_list = [f for f in file_list if f in file_subset_list]

	if include_waveforms and waveform_folder is None:
		print('ERROR if include_waveform is selected, must define waveform_folder!')
		return

	total_files = len(file_list)
	split_size = total_files // nchunks

	for i in tqdm(range(chunk_index*split_size, (chunk_index+1)*split_size), disable=verbose):
		filename = file_list[i]
		save_path = os.path.join(save_folder, filename[:-3] + 'hdf5')
		if not os.path.exists(save_path) or recalculate:
			if verbose:
				logger.info(f'Processing file: {filename}...')
			if filename[-3:] == 'dcm':
				try:
					process_video(os.path.join(dcm_folder, filename), save_path, segmentor_model,
					              verbose=verbose, mode=mode, bkgd_comp=bkgd_comp, flipLR=flipLR,
					              no_saliency=no_saliency, OF_algo=OF_algo, save_mask_subset=save_mask_subset,
					              include_waveforms=include_waveforms, waveform_folder=waveform_folder)
				except Exception as e:
					logger.error(f'Error processing {filename}: {e}')
					if verbose:
						traceback.print_exc()
			else:
				logger.warning(f'File extension must be dcm, found {filename[-3:]}, skipping')
				continue
		else:
			if verbose:
				logger.debug(f'File {save_path} exists! Skipping file {filename}')

def _read_dicom_file(dcm_path: str, verbose: bool = False) -> Tuple[Optional[Any], Optional[np.ndarray]]:
	"""
	Read DICOM file and extract pixel array.
	
	Args:
		dcm_path: Path to DICOM file
		verbose: Whether to print progress
	
	Returns:
		Tuple of (dataset, pixel_array) or (None, None) on error
	"""
	if verbose:
		logger.info('Detected DICOM as input type!')
		logger.info(f'Opening file {dcm_path}')
	try:
		ds = dcm.dcmread(dcm_path)
		nparr = ds.pixel_array
		return ds, nparr
	except (IOError, OSError, KeyError, AttributeError) as error:
		logger.error(f'Unable to read DICOM: {error}')
		return None, None


def _extract_dicom_metadata(ds: Any, verbose: bool = False) -> Dict[str, Any]:
	"""
	Extract metadata from DICOM dataset.
	
	Args:
		ds: DICOM dataset
		verbose: Whether to print progress
	
	Returns:
		Dictionary with metadata keys: pixel_spacing, frame_rate, R_times, R_wave_data_present
	"""
	metadata = {
		'pixel_spacing': None,
		'frame_rate': None,
		'R_times': None,
		'R_wave_data_present': False
	}
	
	# Extract pixel spacing
	try:
		metadata['pixel_spacing'] = ds[0x0018, 0x6011][0]['PhysicalDeltaX'].value
	except (KeyError, AttributeError, IndexError, TypeError) as e:
		if verbose:
			logger.warning(f'No pixel spacing metadata: {e}. Flagging as no conversion factor.')
	
	# Extract R-wave times
	try:
		if type(ds.RWaveTimeVector) != float and ds.RWaveTimeVector is not None:
			metadata['R_times'] = np.asarray(ds.RWaveTimeVector)
			if verbose:
				logger.info('RWaveTimeVector tag present! Saving data...')
			metadata['R_wave_data_present'] = True
		else:
			if verbose:
				logger.debug('No RWaveTimeVector tag present!')
	except (AttributeError, KeyError, TypeError) as e:
		if verbose:
			logger.debug(f'No RWaveTimeVector tag present: {e}')
	
	# Extract frame rate
	try:
		metadata['frame_rate'] = ds.CineRate
	except (AttributeError, KeyError):
		try:
			metadata['frame_rate'] = np.round(1000 / float(ds.FrameTime))
		except (AttributeError, KeyError, ValueError, ZeroDivisionError):
			try:
				metadata['frame_rate'] = np.round(1000 / float(ds.FrameTimeVector[1]))
			except (AttributeError, KeyError, IndexError, ValueError, ZeroDivisionError) as e:
				if verbose:
					logger.warning(f'No frame rate information: {e}. Flagging as no conversion factor.')
	
	return metadata


def _save_optical_flow_to_hdf5(save_path: str, flow_arr: np.ndarray, nparr: np.ndarray,
                               mask_dict: Dict[str, np.ndarray], metadata: Dict[str, Any],
                               waveforms: Dict[str, Tuple[bool, Optional[np.ndarray]]],
                               ds: Any, config: OpticalFlowCalculationConfig,
                               mode: str, no_saliency: bool, include_waveforms: bool,
                               save_mask_subset: Optional[List[str]], verbose: bool) -> None:
	"""
	Save optical flow data to HDF5 file.
	
	Args:
		save_path: Path to save HDF5 file
		flow_arr: Optical flow array
		nparr: Original pixel array
		mask_dict: Dictionary of masks
		metadata: Metadata dictionary
		waveforms: Dictionary of waveform results
		ds: DICOM dataset
		config: Configuration object
		mode: Processing mode
		no_saliency: Whether saliency was skipped
		include_waveforms: Whether waveforms are included
		save_mask_subset: Optional list of mask keys to save
		verbose: Whether to print progress
	"""
	if verbose:
		logger.info('Saving as hdf5 file...')
	if os.path.exists(save_path):
		os.remove(save_path)
	
	with h5py.File(save_path, 'w') as f:
		gray_arr = rgb2gray(nparr)
		pxl_dset = f.create_dataset('echo', data=gray_arr.astype(np.float16),
		                           compression='gzip', compression_opts=9)
		flow_dset = f.create_dataset('flow', data=flow_arr.astype(np.float16),
		                            compression='gzip', compression_opts=9)
		flow_dset.attrs['frame_rate'] = metadata['frame_rate']
		flow_dset.attrs['nframes'] = nparr.shape[0]
		flow_dset.attrs['pixel_spacing'] = metadata['pixel_spacing']
		flow_dset.attrs['ID'] = ds.PatientID
		try:
			flow_dset.attrs['HR'] = ds.HeartRate
		except (AttributeError, KeyError):
			flow_dset.attrs['HR'] = 0
		flow_dset.attrs['no_saliency'] = no_saliency
		flow_dset.attrs['mode'] = mode
		flow_dset.attrs['units_converted'] = (metadata['pixel_spacing'] is not None and
		                                     metadata['frame_rate'] is not None)
		flow_dset.attrs['waveforms_present'] = include_waveforms
		
		if include_waveforms:
			ecg_exists, _ = waveforms.get('ecg', (False, None))
			art_exists, _ = waveforms.get('art', (False, None))
			CVP_exists, _ = waveforms.get('cvp', (False, None))
			PAP_exists, _ = waveforms.get('pap', (False, None))
			
			flow_dset.attrs['CVP_exists'] = CVP_exists
			flow_dset.attrs['PAP_exists'] = PAP_exists
			flow_dset.attrs['R_wave_data_present'] = metadata['R_wave_data_present']
			
			if art_exists:
				art = waveforms['art'][1]
				art_dset = f.create_dataset('art', data=art.astype(np.float16),
				                           compression='gzip', compression_opts=9)
				art_dset.attrs['sampling_rate'] = config.art_sampling_rate
			if ecg_exists:
				ecg = waveforms['ecg'][1]
				ecg_dset = f.create_dataset('ecg', data=ecg.astype(np.float16),
				                           compression='gzip', compression_opts=9)
				ecg_dset.attrs['sampling_rate'] = config.ecg_sampling_rate
			if CVP_exists:
				cvp = waveforms['cvp'][1]
				cvp_dset = f.create_dataset('cvp', data=cvp.astype(np.float16),
				                           compression='gzip', compression_opts=9)
				cvp_dset.attrs['sampling_rate'] = config.cvp_sampling_rate
			if PAP_exists:
				pap = waveforms['pap'][1]
				pap_dset = f.create_dataset('pap', data=pap.astype(np.float16),
				                           compression='gzip', compression_opts=9)
				pap_dset.attrs['sampling_rate'] = config.pap_sampling_rate
		
		if metadata['R_wave_data_present']:
			if verbose:
				logger.debug(f'R-wave times: {metadata["R_times"]}')
			RWaveTime_dset = f.create_dataset('RWaveTime', data=metadata['R_times'],
			                                 compression='gzip', compression_opts=9)
		
		saved_keys = []
		for k in mask_dict.keys():
			if save_mask_subset is not None:
				if k in save_mask_subset:
					if verbose:
						print(f'Saving binary mask: {k}')
					mask_dset = f.create_dataset(k, data=mask_dict[k],
					                           compression="gzip", compression_opts=9)
					saved_keys.append(k)
			else:
				if verbose:
					logger.debug(f'Saving binary mask: {k}')
				mask_dset = f.create_dataset(k, data=mask_dict[k],
				                           compression="gzip", compression_opts=9)
				saved_keys.append(k)
		
		flow_dset.attrs['labels'] = saved_keys
	
	if verbose:
		logger.info(f'Saved optical flow array of shape {flow_arr.shape} to {save_path}!')


def process_video(dcm_path: str, save_path: str, segmentor_model: torch.nn.Module,
                 verbose: bool = True, mode: str = 'A4C', bkgd_comp: str = 'none',
                 flipLR: bool = False, no_saliency: bool = False, OF_algo: str = 'TVL1',
                 save_mask_subset: Optional[List[str]] = None, include_waveforms: bool = False,
                 waveform_folder: Optional[str] = None,
                 config: Optional[OpticalFlowCalculationConfig] = None) -> Optional[None]:
	"""
	Process DICOM video file and calculate optical flow.
	
	Args:
		dcm_path: Path to DICOM file
		save_path: Path to save HDF5 file
		segmentor_model: SAM model for segmentation
		verbose: Whether to print progress
		mode: Processing mode ('A4C', 'RVIO_2class', 'otsu')
		bkgd_comp: Background compensation method ('WASE', 'none')
		flipLR: Whether to flip left-right
		no_saliency: Whether to skip saliency calculation
		OF_algo: Optical flow algorithm ('TVL1', 'deepflow')
		save_mask_subset: Optional list of mask keys to save
		include_waveforms: Whether to include waveform data
		waveform_folder: Folder containing waveform files
		config: Optional configuration object
	
	Returns:
		None on success, None on error (should be changed to raise exceptions)
	"""
	if config is None:
		config = default_optical_flow_config()
	
	# Validate configuration
	if mode == 'otsu':
		if bkgd_comp != 'none':
			error_msg = f'bkgd_comp {bkgd_comp} is not supported in mode=otsu, can only support bkgd_comp=none'
			logger.error(error_msg)
			raise ConfigurationError(error_msg)
		if save_mask_subset is not None:
			error_msg = 'In mode=otsu, save_mask_subset must be None'
			logger.error(error_msg)
			raise ConfigurationError(error_msg)

	# Read DICOM file
	ds, nparr = _read_dicom_file(dcm_path, verbose)
	if ds is None or nparr is None:
		raise DICOMReadError(f'Failed to read DICOM file: {dcm_path}')
	
	# Convert color space if needed
	if dcm.pixel_data_handlers.numpy_handler.should_change_PhotometricInterpretation_to_RGB(ds):
		nparr = dcm.pixel_data_handlers.convert_color_space(nparr, ds.PhotometricInterpretation, 'RGB')
	
	# Extract metadata
	metadata = _extract_dicom_metadata(ds, verbose)
	pixel_spacing = metadata['pixel_spacing']
	frame_rate = metadata['frame_rate']

	if len(nparr.shape) == 3 and nparr.shape[0] > 1:
		if verbose:
			logger.warning(f'Pixel data is of shape {nparr.shape}, likely greyscale data. Converting to RGB...')
		nparr = gray2rgb(nparr)

	if pixel_spacing is None or frame_rate is None:
		conversion_factor = 1.0  # No conversion if metadata missing
	else:
		conversion_factor = pixel_spacing * frame_rate

	if verbose:
		logger.info(f'Pixel data obtained, of shape: {nparr.shape}')
	if flipLR:
		nparr = np.flip(nparr, axis=2)
		if verbose:
			logger.debug(f'Flipping pixel array LR, shape is now: {nparr.shape}')
	if mode == 'A4C' or mode == 'RVIO_2class':
		mask_dict = predict_movie(nparr, segmentor_model, mode=mode, verbose=verbose, config=config)
	elif mode == 'otsu':
		mask_dict = predict_movie_thres(nparr, verbose=verbose, config=config)
	else:
		error_msg = f'Input for mode must be [A4C, otsu, RVIO_2class], not {mode}.'
		logger.error(error_msg)
		raise ConfigurationError(error_msg)
	nframes = nparr.shape[0]

	if not no_saliency:
		saliency_obj = cv2.saliency.StaticSaliencyFineGrained_create()
	saliency_1 = 0
	saliency_2 = 0

	OF_model = None
	if OF_algo == 'deepflow':
		if verbose:
			logger.info('Using deepflow as OF algo')
		OF_model = cv2.optflow.createOptFlow_DeepFlow()
	if OF_algo == 'TVL1':
		if verbose:
			logger.info('Using DualTVL1 as OF algo')
		if cv2.cuda.getCudaEnabledDeviceCount() > 0:
			if verbose:
				logger.info('CUDA detected, using GPU acceleration!')
			OF_model = cv2.cuda.OpticalFlowDual_TVL1.create()
		else:
			OF_model = cv2.optflow.createOptFlow_DualTVL1()
			OF_model.setLambda(config.lambda_value)

	flow_list = []
	if verbose:
		logger.info('Calculating saliency and bkgd compensated optical flow...')
		logger.debug(f'Settings for optical flow calc: bkgd_comp = {bkgd_comp}, no_saliency = {no_saliency}')
	for i in tqdm(range(nframes), disable=(not verbose)):
		if not no_saliency:
			success, saliency_2 = saliency_obj.computeSaliency(nparr[i,:,:,:])
		else:
			saliency_2 = img2uint8(rgb2gray(nparr[i,:,:,:]))

		if i == 0:
			saliency_1 = saliency_2
			continue
		else:
			flow_frame = calculate_optical_flow(saliency_1, saliency_2, mask_dict,
																					OF_model, bkgd_comp=bkgd_comp, OF_algo=OF_algo)
			flow_list.append(flow_frame)
			saliency_1 = saliency_2

	flow_list.append(flow_list[-1]) #copy last optical flow	to make it the same length
	flow_arr = np.stack(flow_list) * conversion_factor #convert to cm/s

	# Load waveforms if requested
	waveform_results = {}
	PAP_exists = False
	CVP_exists = False
	ecg_exists = False
	art_exists = False
	
	if include_waveforms:
		if verbose:
			logger.info('Loading waveform data:')
		waveform_results = load_all_waveforms(dcm_path, waveform_folder, config, verbose)
		
		ecg_exists, ecg = waveform_results.get('ecg', (False, None))
		art_exists, art = waveform_results.get('art', (False, None))
		CVP_exists, cvp = waveform_results.get('cvp', (False, None))
		PAP_exists, pap = waveform_results.get('pap', (False, None))
		
		if not ecg_exists and not art_exists:
			include_waveforms = False

	# Save to HDF5
	_save_optical_flow_to_hdf5(save_path, flow_arr, nparr, mask_dict, metadata,
	                          waveform_results, ds, config, mode, no_saliency,
	                          include_waveforms, save_mask_subset, verbose)

def calculate_optical_flow(saliency_1: np.ndarray, saliency_2: np.ndarray,
                          mask_dict: Dict[str, np.ndarray], OF_model: Any,
                          bkgd_comp: str = 'none', OF_algo: str = 'TVL1') -> Optional[np.ndarray]:
	if OF_algo == 'deepflow':
		flow = OF_model.calc(saliency_1, saliency_2, None)
	elif OF_algo == 'TVL1':
		if cv2.cuda.getCudaEnabledDeviceCount() > 0:
			gpu_frame1 = cv2.cuda_GpuMat()
			gpu_frame2 = cv2.cuda_GpuMat()
			gpu_frame1.upload(saliency_1)
			gpu_frame2.upload(saliency_2)
			gpu_flow = OF_model.calc(gpu_frame1, gpu_frame2, None)
			flow = gpu_flow.download()
		else:
			logger.debug("CUDA not available, using CPU instead.")
			flow = OF_model.calc(saliency_1, saliency_2, None)
	else:
		error_msg = 'OF_algo only supports deepflow or TVL1'
		logger.error(error_msg)
		raise OpticalFlowCalculationError(error_msg)
	background = 0

	if bkgd_comp == 'WASE':
		mask = mask_dict['bkgd']
		masked_flow = flow * mask
		background = np.mean(masked_flow[masked_flow != 0]) #mean flow vector of the bkgd
	elif bkgd_comp == 'none':
		background = 0
	else:
		error_msg = f'bkgd_comp value must be [WASE, none], got {bkgd_comp}!'
		logger.error(error_msg)
		raise OpticalFlowCalculationError(error_msg)
	flow_bkgd_comp = flow - background
	return flow_bkgd_comp

def _load_segmentor_model(checkpoint_dir: str, arch: str = 'vit_t') -> torch.nn.Module:
	"""
	Load segmentor model from checkpoint.
	
	Args:
		checkpoint_dir: Directory containing checkpoint and args.json
		arch: Model architecture (default: 'vit_t')
	
	Returns:
		Loaded and configured model
	"""
	args_path = os.path.join(checkpoint_dir, 'args.json')
	
	if not os.path.exists(args_path):
		raise FileNotFoundError(f'Model args file not found: {args_path}')
	
	# Reading the args from the json file
	with open(args_path, 'r') as f:
		model_args_dict = json.load(f)
	
	# Converting dictionary to Namespace
	model_args = Namespace(**model_args_dict)
	
	# Override arch if provided
	if arch:
		model_args.arch = arch
	
	segmentor = sam_model_registry[model_args.arch](
		model_args,
		checkpoint=os.path.join(model_args.dir_checkpoint, 'checkpoint_best.pth'),
		num_classes=model_args.num_cls
	)
	segmentor = segmentor.to('cuda').eval()
	
	return segmentor


if __name__ == '__main__':
	# Setup logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	
	parser = argparse.ArgumentParser(description='Process DICOM files and calculate optical flow')
	parser.add_argument('--nchunks', type=int, required=True, help='Number of chunks to split processing')
	parser.add_argument('--dcm_folder', type=str, required=True, help='Folder containing DICOM files')
	parser.add_argument('--save_folder', type=str, required=True, help='Folder to save HDF5 files')
	parser.add_argument('--waveform_folder', type=str, help='Folder containing waveform files')
	parser.add_argument('--checkpoint_dir', type=str,
	                   default='2D-SAM_vitT_encoderdecoder_vanilla_noprompt_RVIO201_multiclass_fold0',
	                   help='Directory containing model checkpoint and args.json')
	parser.add_argument('--arch', type=str, default='vit_t', help='Model architecture')
	parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
	parser.add_argument('--recalculate', action='store_true', help='Recalculate even if files exist')
	parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device ID')
	
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
	
	# Load model
	try:
		segmentor = _load_segmentor_model(args.checkpoint_dir, args.arch)
		logger.info(f'Loaded model from {args.checkpoint_dir}')
	except Exception as e:
		logger.error(f'Failed to load model: {e}')
		raise
	
	chunk_index_list = range(args.nchunks)
	total_chunks = args.nchunks

	for chunk_index in chunk_index_list:
		dcm_folder = args.dcm_folder
		save_folder = os.path.join(args.save_folder, f'chunk{chunk_index}')
		waveform_folder = args.waveform_folder

		process_folder(dcm_folder, save_folder, segmentor, nchunks=total_chunks, chunk_index=chunk_index,
		               mode='RVIO_2class', bkgd_comp='none', flipLR=False, verbose=args.verbose,
		               recalculate=args.recalculate, no_saliency=True,
		               OF_algo='deepflow', save_mask_subset=None, include_waveforms=True,
		               waveform_folder=waveform_folder, pixel_spacing=None, frame_rate=None)
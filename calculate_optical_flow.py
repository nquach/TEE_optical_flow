import cv2
import numpy as np
import os
import pydicom as dcm
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
import imageio.v3 as iio
import matplotlib.colors
from tqdm import tqdm
from skimage import feature
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy.ndimage import binary_fill_holes
from matplotlib.colors import LogNorm, Normalize
import pickle as pkl
import multiprocessing
import gc
from scipy.signal import savgol_filter
import peakutils
import h5py
import gdcm
import traceback

#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
#Pytorch packages
import torch
from torch import nn
import torchvision
from torchvision import datasets
#Visulization
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from argparse import Namespace
import cfg
import PIL
import torchio as tio
import json
import argparse
from optical_flow_utils import *

#smoothing factor for DualTVL1 algo, smaller is smoother, algo defaults to 0.15
LAMBDA = 0.15

np.float = np.float32
np.int = np.int_

#UTILITY FUNCTIONS

# Function to evaluate a single image slice
def evaluate_1_slice(nparr, model):
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

def moving_avg_mask(arr, n=4, threshold=0.49):
	arr2 = np.vstack((np.expand_dims(arr[0,:,:], axis=0), arr, np.expand_dims(arr[-1,:,:], axis=0), np.expand_dims(arr[-1,:,:], axis=0)))
	sum = np.cumsum(arr2.astype(float), axis=0)
	sum[n:,:,:] = sum[n:,:,:] - sum[:-n,:,:]
	avg = sum[n-1:,:,:]/n
	return avg > threshold

def clean_mask(arr, mode='A4C', verbose=False): #expects (N, H, W)
	if verbose:
		print('====Cleaning masks====')
		print('Turning into 1-hot encodings...')
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
		if verbose:
			print(f'ERROR mode={mode} not supported, must be [A4C, RVIO_2class, MouseRV_A4C]!')
			return None
	aggregate_mask = np.zeros_like(arr)
	for k in mask_dict.keys():
		arr = mask_dict[k]
		arr = moving_avg_mask(np.squeeze(arr))
		nframes = arr.shape[0]
		frame_list = []
		for i in tqdm(range(nframes), disable=(not verbose)):
			clean = remove_small_objects(binary_fill_holes(arr[i,:,:]), min_size=500)
			frame_list.append(clean)

		clean_arr = np.stack(frame_list)
		aggregate_mask = np.logical_or(clean_arr, aggregate_mask)
		clean_arr_2c = np.repeat(clean_arr[:,:,:,np.newaxis], 2, axis=3)
		if verbose:
			print(f'For mask {k}, produced cleaned mask arr of shape ', clean_arr_2c.shape)
		mask_dict[k] = clean_arr_2c

	if verbose:
		print('calculating bkgd...')
	bkgd_1c = np.logical_not(aggregate_mask)
	bkgd_2c = np.repeat(bkgd_1c[:,:,:,np.newaxis], 2, axis=3)
	if verbose:
			print(f'For bkgd mask, produced cleaned mask arr of shape ', bkgd_2c.shape)
	mask_dict['bkgd'] = bkgd_2c
	return mask_dict

def predict_movie_thres(nparr, verbose=False):
	nframes = nparr.shape[0]
	mask_list = []
	if verbose:
		print('Predicting frames...')
	for i in tqdm(range(nframes), disable=(not verbose)):
		img_arr = rgb2gray(np.squeeze(nparr[i,:,:,:]))
		mask =	img_arr > threshold_otsu(img_arr)
		mask = remove_small_objects(binary_fill_holes(mask), min_size=500)
		mask_list.append(mask)
	mask_arr = np.stack(mask_list)
	mask_arr = moving_avg_mask(np.squeeze(mask_arr))
	mask_arr_2c = np.repeat(mask_arr[:,:,:,np.newaxis], 2, axis=3)
	if verbose:
		print(f'Produced thresholded mask of shape {mask_arr_2c.shape}')
	return {'otsu':mask_arr_2c}

def predict_movie(nparr, model, mode='A4C', verbose=False):
	nframes = nparr.shape[0]
	pred_list = []
	if verbose:
		print('Predicting frames...')
	for i in tqdm(range(nframes), disable=(not verbose)):
		pred = evaluate_1_slice(nparr[i,:,:,:], model)
		pred_list.append(pred)

	mask_arr = np.stack(pred_list)
	return clean_mask(mask_arr, mode, verbose) #spits out mask_dict

def process_folder(dcm_folder, save_folder, segmentor_model, nchunks=10, chunk_index=0,
									 mode='RVIO_2class', bkgd_comp='none', flipLR=False, verbose=True,
									recalculate=False, no_saliency=True,
									 OF_algo='TVL1', save_mask_subset=None, include_waveforms=False, waveform_folder=None,
									 pixel_spacing=None, frame_rate=None, process_subset=False, file_subset_list=[]):
	safe_makedir(save_folder)
	file_list = os.listdir(dcm_folder)

	if process_subset:
		if len(file_subset_list) == 0:
			print('ERROR! File subset list is empty!')
			return
		if verbose:
			print(f'Processing subset! Found {len(file_subset_list)} files in subset list provided!')
		file_list = [f for f in file_list if f in file_subset_list]

	if include_waveforms and waveform_folder == None:
		print('ERROR if include_waveform is selected, must define waveform_folder!')
		return

	total_files = len(file_list)
	split_size = total_files // nchunks

	for i in tqdm(range(chunk_index*split_size, (chunk_index+1)*split_size), disable=verbose):
		filename = file_list[i]
		save_path = os.path.join(save_folder, filename[:-3] + 'hdf5')
		if not os.path.exists(save_path) or recalculate:
			if verbose:
				print(f'Processing file: {filename}...')
			if filename[-3:] == 'dcm':
				try:
					process_video(os.path.join(dcm_folder, filename), save_path, segmentor_model,
													verbose=verbose, mode=mode, bkgd_comp=bkgd_comp, flipLR=flipLR,
											no_saliency=no_saliency, OF_algo=OF_algo, save_mask_subset=save_mask_subset,
											include_waveforms=include_waveforms, waveform_folder=waveform_folder)
				except Exception:
					if verbose:
						traceback.print_exc()
					print(f'An error occurred skipping!')
			else:
				print(f'ERROR: file extension must be dcm, found {filename[-3:]}')
				continue
		else:
			if verbose:
				print(f'File {save_path} exists! Skipping file {filename}')

def process_video(dcm_path, save_path, segmentor_model, verbose=True,
									mode='A4C', bkgd_comp='none', flipLR=False,
									no_saliency=False, OF_algo='TVL1', save_mask_subset=None,
									include_waveforms=False, waveform_folder=None):
	if mode == 'otsu':
		if bkgd_comp != 'none':
			print(f'ERROR bkgd_comp {bkgd_comp} is not supported in mode=otsu, can only support bkgd_comp=none')
			return None
		if save_mask_subset != None:
			print(f'ERROR in mode=otsu, save_mask_subset must be None')
			return None

	if verbose:
		print('Detected DICOM as input type!')
		print('Opening file ' + dcm_path)
	try:
		ds = dcm.dcmread(dcm_path)
		nparr = ds.pixel_array
		#print(f'nparr dtype is: {nparr.dtype}')
	except Exception as error:
		print('ERROR UNABLE TO READ DICOM:\n', error)
		return
	if dcm.pixel_data_handlers.numpy_handler.should_change_PhotometricInterpretation_to_RGB(ds):
			nparr = dcm.pixel_data_handlers.convert_color_space(nparr, ds.PhotometricIntepretation, 'RGB')
	try:
		pixel_spacing = ds[0x0018,0x6011][0]['PhysicalDeltaX'].value
	except:
		print('ERROR no pixel spacing metadata! Flagging as no conversion factor!!')
		pixel_spacing = None

	R_wave_data_present = False
	R_times = None
	try:
		if type(ds.RWaveTimeVector) != float and not np.any(ds.RWaveTimeVector == None):
			R_times = np.asarray(ds.RWaveTimeVector)
			if verbose:
				print('RWaveTimeVector tag present! Saving data...')
			R_wave_data_present = True
		else:
			if verbose:
				print('No RWaveTimeVector tag present!')

	except:
		if verbose:
			print('No RWaveTimeVector tag present!')
			R_wave_data_present = False
	try:
		frame_rate = ds.CineRate
	except:
		try:
			frame_rate = np.round(1000/float(ds.FrameTime))
		except:
			try:
				frame_rate = np.round(1000/float(ds.FrameTimeVector[1]))
			except:
				print('ERROR: No frame rate information! Flagging as no conversion factor!')
				frame_rate = None

	if len(nparr.shape) == 3 and nparr.shape[0] > 1:
		if verbose:
			print(f'WARNING: Pixel data is of shape {nparr.shape}, likely greyscale data. Converting to RGB...')
		nparr = gray2rgb(nparr)

	if pixel_spacing == None or frame_rate == None:
		units_converted_flag = False
	else:
		units_converted_flag = True
		conversion_factor = pixel_spacing * frame_rate

	if verbose:
		print(f'Pixel data obtained, of shape: {nparr.shape}')
	if flipLR:
		nparr = np.flip(nparr, axis=2)
		if verbose:
			print(f'Flipping pixel array LR, shape is now: {nparr.shape}')
	if mode == 'A4C' or mode == 'RVIO_2class':
		mask_dict = predict_movie(nparr, segmentor_model, mode=mode, verbose=verbose) #this needs to	be (N, H, W, 2)
	elif mode == 'otsu':
		mask_dict = predict_movie_thres(nparr, verbose=verbose)
	else:
		print(f'ERROR input for mode must be [A4C, otsu, RVIO_2class], not {mode}.')
		return None
	nframes = nparr.shape[0]

	if not no_saliency:
		saliency_obj = cv2.saliency.StaticSaliencyFineGrained_create()
	saliency_1 = 0
	saliency_2 = 0

	OF_model = None
	if OF_algo == 'deepflow':
		if verbose:
			print('Using deepflow as OF algo')
		OF_model = cv2.optflow.createOptFlow_DeepFlow()
	if OF_algo == 'TVL1':
		if verbose:
			print('Using DualTVL1 as OF algo')
		if cv2.cuda.getCudaEnabledDeviceCount() > 0:
			if verbose:
				print('CUDA detected, using GPU acceleration!')
			OF_model = cv2.cuda.OpticalFlowDual_TVL1.create()
		else:
			OF_model = cv2.optflow.createOptFlow_DualTVL1()
			OF_model.setLambda(LAMBDA)
			#OF_model.setWarpingsNumber(10)
			#OF_model.setScalesNumber(10)


	flow_list = []
	if verbose:
		print('Calculating saliency and bkgd compensated optical flow...')
		print(f'Settings for optical flow calc: bkgd_comp = {bkgd_comp}, no_saliency = {no_saliency}')
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

	PAP_exists = False
	CVP_exists = False
	if include_waveforms:
		if verbose:
			print('Loading waveform data:')
		ecg_path = os.path.join(waveform_folder, os.path.basename(dcm_path)[:-4] + '_II.npy')
		art_path = os.path.join(waveform_folder, os.path.basename(dcm_path)[:-4] + '_ART.npy')
		abp_path = os.path.join(waveform_folder, os.path.basename(dcm_path)[:-4] + '_ABP.npy')
		pap_path = os.path.join(waveform_folder, os.path.basename(dcm_path)[:-4] + '_PAP.npy')
		cvp_path = os.path.join(waveform_folder, os.path.basename(dcm_path)[:-4] + '_CVP.npy')
		if os.path.exists(pap_path):
			PAP_exists = True
			pap = np.load(pap_path)
			if np.max(np.gradient(pap)) < 0.05:
				print('ERROR PAP waveform is flat!')
				PAP_exists = False #Don't include PAP if its flat
			if np.mean(pap) > 100:
				print('ERROR PAP waveform is too high, mean > 100mmHg!')
				PAP_exists = False #Don't include CVP if its mean is unreasonably high
			if np.mean(pap) < 0:
				print('ERROR PAP waveform is negative, mean < 0mmHg!')
				PAP_exists = False #Don't include CVP if its mean is unreasonably low
		if os.path.exists(cvp_path):
			CVP_exists = True
			cvp = np.load(cvp_path)
			if np.mean(cvp) > 50:
				print('ERROR CVP waveform is too high, mean > 50mmHg!')
				CVP_exists = False #Don't include CVP if its mean is unreasonably high
			if np.mean(cvp) < -10:
				print('ERROR CVP waveform is too negative, mean < -10mmHg!')
				CVP_exists = False #Don't include CVP if its mean is unreasonably low
		ecg_exists = False
		if os.path.exists(ecg_path):
			ecg = np.load(ecg_path)
			ecg_exists = True
			if verbose:
				print('Loaded ECG!')
		else:
			print(f'ERROR {ecg_path} doesnt exist! No ECG waveform detected')

		art_exists = False
		if os.path.exists(art_path):
			art = np.load(art_path)
			art_exists = True
			if verbose:
				print('Loaded ART!')
			if np.max(np.gradient(art)) < 0.05: #check if its flat
				if os.path.exists(abp_path):
					art = np.load(abp_path)
					if np.max(np.gradient(art)) < 0.05:
						art_exists = False
						print('ERROR ART and ABP waveforms given are flat!')
				else:
					art_exists = False
					print('ERROR ART is flat and ABP doesnt exist!')
		else:
			if os.path.exists(abp_path):
				art = np.load(abp_path)
				art_exists = True
				if np.max(np.gradient(art)) < 0.05:
					print('ERROR ART and ABP waveforms given are flat!')
					art_exists = False
			else:
				print('ERROR ART and ABP path doesnt exist!')
				art_exists = False

		if not ecg_exists and not art_exists:
			include_waveforms = False

	if verbose:
		print('Saving as hdf5 file...')
	if os.path.exists(save_path):
		os.remove(save_path)
	with h5py.File(save_path, 'w') as f:
		gray_arr = rgb2gray(nparr)
		pxl_dset = f.create_dataset('echo', data=gray_arr.astype(np.float16), compression='gzip', compression_opts=9)
		flow_dset = f.create_dataset('flow', data=flow_arr.astype(np.float16), compression='gzip', compression_opts=9)
		flow_dset.attrs['frame_rate'] = frame_rate
		flow_dset.attrs['nframes'] = nframes
		flow_dset.attrs['pixel_spacing'] = pixel_spacing
		flow_dset.attrs['ID'] = ds.PatientID
		try:
			flow_dset.attrs['HR'] = ds.HeartRate
		except:
			flow_dset.attrs['HR'] = 0
		flow_dset.attrs['no_saliency'] = no_saliency
		flow_dset.attrs['mode'] = mode
		flow_dset.attrs['units_converted'] = units_converted_flag
		flow_dset.attrs['waveforms_present'] = include_waveforms
		if include_waveforms:
			flow_dset.attrs['CVP_exists'] = CVP_exists
			flow_dset.attrs['PAP_exists'] = PAP_exists
			flow_dset.attrs['R_wave_data_present'] = R_wave_data_present
			if include_waveforms:
				if art_exists:
					art_dset = f.create_dataset('art', data=art.astype(np.float16), compression='gzip', compression_opts=9)
					art_dset.attrs['sampling_rate'] = 125
				if ecg_exists:
					ecg_dset = f.create_dataset('ecg', data=ecg.astype(np.float16), compression='gzip', compression_opts=9)
					ecg_dset.attrs['sampling_rate'] = 500
				if CVP_exists:
					cvp_dset = f.create_dataset('cvp', data=cvp.astype(np.float16), compression='gzip', compression_opts=9)
					cvp_dset.attrs['sampling_rate'] = 125
				if PAP_exists:
					pap_dset = f.create_dataset('pap', data=pap.astype(np.float16), compression='gzip', compression_opts=9)
					pap_dset.attrs['sampling_rate'] = 125

		if R_wave_data_present:
			print(R_times)
			RWaveTime_dset = f.create_dataset('RWaveTime', data=R_times, compression='gzip', compression_opts=9)

		saved_keys = []
		for k in mask_dict.keys():
			if save_mask_subset != None:
				if k in save_mask_subset:
					if verbose:
						print(f'Saving binary mask: {k}')
					mask_dset = f.create_dataset(k, data=mask_dict[k], compression="gzip", compression_opts=9)
					saved_keys.append(k)
			else:
				if verbose:
					print(f'Saving binary mask: {k}')
				mask_dset = f.create_dataset(k, data=mask_dict[k], compression="gzip", compression_opts=9)
				saved_keys.append(k)

		flow_dset.attrs['labels'] = saved_keys

	if verbose:
		print(f'Saved optical flow array of shape {flow_arr.shape} to {save_path}!')

def calculate_optical_flow(saliency_1, saliency_2, mask_dict, OF_model, bkgd_comp = 'none', OF_algo='TVL1'):
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
			print("CUDA not available, using CPU instead.")
			flow = OF_model.calc(saliency_1, saliency_2, None)
	else:
		print('ERROR OF_algo only supports deepflow or TVL1')
		return
	background = 0

	if bkgd_comp == 'WASE':
		mask = mask_dict['bkgd']
		masked_flow = flow * mask
		background = np.mean(masked_flow[masked_flow != 0]) #mean flow vector of the bkgd
	elif bkgd_comp == 'none':
		background = 0
	else:
		print('ERROR: bkgd_comp value must be [WASE, none]!')
		return None
	flow_bkgd_comp = flow - background
	return flow_bkgd_comp

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nchunks', type=int)
	parser.add_argument('--dcm_folder', type=str)
	parser.add_argument('--save_folder', type=str)
	parser.add_argument('--waveform_folder', type=str)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--recalculate', action='store_true')
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	arch="vit_t"	# Change this value as needed
	finetune_type="vanilla"
	dataset_name="WASE150multiclass"	# Assuming you set this if it's dynamic

	# Construct the checkpoint directory argument
	checkpoint_dir= "2D-SAM_vitT_encoderdecoder_vanilla_noprompt_RVIO201_multiclass_fold0"

	args_path = f"{checkpoint_dir}/args.json"

	# Reading the args from the json file
	with open(args_path, 'r') as f:
		args_dict = json.load(f)

	# Converting dictionary to Namespace
	args = Namespace(**args_dict)

	segmentor = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
	segmentor = segmentor.to('cuda').eval()

	chunk_index_list = range(args.nchunks)
	total_chunks = args.nchunks

	for chunk_index in chunk_index_list:
		dcm_folder = args.dcm_folder
		save_folder = os.path.join(args.save_folder, f'chunk{chunk_index}')
		waveform_folder = args.waveform_folder

		process_folder(dcm_folder, save_folder, segmentor, nchunks=total_chunks, chunk_index=chunk_index,
		mode='RVIO_2class', bkgd_comp='none', flipLR=False, verbose=args.verbose,
		recalculate=args.recalculate, no_saliency=True,
		OF_algo='deepflow', save_mask_subset=None, include_waveforms=True, waveform_folder=waveform_folder,
		pixel_spacing=None, frame_rate=None)
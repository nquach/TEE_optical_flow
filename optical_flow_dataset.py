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

class OpticalFlowDataset:
	def __init__(self, hdf5_filepath):
		self.GRAPH_CALCULATED = False
		self.CARDIACCYCLE_CALCULATED = False
		f = h5py.File(hdf5_filepath)
		self.filename = os.path.basename(hdf5_filepath)[:-4]
		self.ds_OF = f['flow']
		self.ds_echo = f['echo'] #this will be grayscale (N,W,H)
		self.vel_array = self.ds_OF[()].astype(np.float32) #deep copy the value of the optical flow array (N, H, W, 2)
		self.nframes = self.ds_OF.attrs['nframes'] - 2
		self.mode = self.ds_OF.attrs['mode']
		if 'RWaveTime' in f:
			self.RTimePresent = True
			self.RWaveTimes = f['RWaveTime'][()]
		else:
			self.RTimePresent = False

		self.waveforms_present = self.ds_OF.attrs['waveforms_present']
		self.units_converted_flag = self.ds_OF.attrs['units_converted']
		if self.units_converted_flag:
			self.frame_rate = self.ds_OF.attrs['frame_rate']
			self.pixel_spacing = self.ds_OF.attrs['pixel_spacing']
			self.ID = self.ds_OF.attrs['ID']
		else:
			self.frame_rate = 1
			self.pixel_spacing = 1
		if self.waveforms_present:
			if 'art' in f:
				self.art = f['art'][()]
				self.art_sampling_rate = f['art'].attrs['sampling_rate']
			else:
				print('ERROR no ART waveform!')

			if 'ecg' in f:
				self.ecg = f['ecg'][()]
				self.ecg_sampling_rate = f['ecg'].attrs['sampling_rate']
			else:
				print('ERROR no ECG waveform')
			if 'cvp' in f:
				self.cvp_exists = True
				self.cvp = f['cvp'][()]
				self.cvp_sampling_rate = f['cvp'].attrs['sampling_rate']
			else:
				self.cvp_exists = False
			if 'pap' in f:
				self.pap = f['pap'][()]
				self.pap_exists = True
				self.pap_sampling_rate = f['pap'].attrs['sampling_rate']
			else:
				self.pap_exists = False

		self.accel_array = np.gradient(self.vel_array, 1/self.frame_rate, axis=0)
		self.pwr_array = self.vel_array * self.accel_array
		self.accepted_labels = self.ds_OF.attrs['labels']
		self.accepted_params = ['velocity', 'acceleration', 'PWR']
		self.mask_ds_dict = {}
		for label in self.accepted_labels:
			ds_label = f[label]
			self.mask_ds_dict[label] = ds_label

	def _validate_label(self, label):
		if label in self.accepted_labels:
			return True
		else:
			return False

	def _validate_param(self, param):
		if param in self.accepted_params:
			return True
		else:
			return False

	def _param_unit(self, param):
		if self.units_converted_flag:
			if param == 'velocity':
				return 'cm/s'
			elif param == 'acceleration':
				return 'cm/s2'
			elif param == 'PWR':
				return 'cm2/s3'
			else:
				print(f'ERROR! {param} is not a valid optical flow parameter, choose from {self.accepted_params}')
				return None
		else:
			if param == 'velocity':
				return 'pixel/frame'
			elif param == 'acceleration':
				return 'pixel/frame2'
			elif param == 'PWR':
				return 'pixel2/frame3'
			else:
				print(f'ERROR! {param} is not a valid optical flow parameter, choose from {self.accepted_params}')
				return None

	def get_echo(self):
		return self.ds_echo[()]

	def get_mask(self, label):
		if self._validate_label(label):
			return self.mask_ds_dict[label][()]
		else:
			print(f'ERROR {label} not a valid key. Choose from {self.accepted_labels}')
			return None

	def get_velocity(self, label):
		if self._validate_label(label):
			mask = self.mask_ds_dict[label][()] #get deep copy of mask
			return self.vel_array * mask
		else:
			print(f'ERROR {label} not a valid key. Choose from {self.accepted_labels}')
			return None

	def get_accel(self, label):
		if self._validate_label(label):
			mask = self.mask_ds_dict[label][()] #get deep copy of mask
			return self.accel_array * mask
		else:
			print(f'ERROR {label} not a valid key. Choose from {self.accepted_labels}')
			return None

	def get_pwr(self, label):
		if self._validate_label(label):
			mask = self.mask_ds_dict[label][()]
			return self.pwr_array * mask
		else:
			print(f'ERROR {label} not a valid key. Choose from {self.accepted_labels}')

	def get_masked_arr(self, param, label):
		if param == 'velocity':
			return self.get_velocity(label)
		elif param == 'acceleration':
			return self.get_accel(label)
		elif param == 'PWR':
			return self.get_pwr(label)
		else:
			print(f'ERROR! {param} is not a valid optical flow parameter, choose from {self.accepted_params}')
			return


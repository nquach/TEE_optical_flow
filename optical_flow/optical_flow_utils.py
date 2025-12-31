import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as iio
import matplotlib.colors
import matplotlib.cm as cm
from tqdm import tqdm
from skimage import feature
from skimage.color import rgb2gray, gray2rgb
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import skimage.measure
import pickle as pkl
import gc
from scipy.signal import savgol_filter
import peakutils
import h5py
import polars as pl
from scipy.stats import mode
from tsmoothie.smoother import SpectralSmoother
import neurokit2 as nk
import traceback

#UTILITY FUNCTIONS
def safe_makedir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def img2uint8(img):
	return img_as_ubyte((img - np.min(img))/np.max(img))

def index_smallest_positive(l):
	non_neg = [i for i in l if i > 0]
	if len(non_neg) == 0:
		return None
	min = np.min(non_neg)
	return l.index(min)

def find_start_stop(arr):
	diffs = np.diff(arr)
	breaks = np.where(diffs != 1)[0] + 1
	clusters = []
	start_idx = 0
	for end_idx in breaks:
		clusters.append([arr[start_idx], arr[end_idx-1]])
		start_idx = end_idx
	clusters.append([arr[start_idx], arr[-1]])
	return clusters

def fix_ecg(ecg_arr, sampling_rate, smooth_fraction=0.2, pad_len=20):
	ecg = nk.ecg_clean(ecg_arr, sampling_rate=sampling_rate, method='vg')
	smoother_ecg = SpectralSmoother(smooth_fraction=smooth_fraction, pad_len=pad_len)
	smoother_ecg.smooth(ecg)
	filt_ecg = np.squeeze(smoother_ecg.smooth_data[0])
	return filt_ecg

#frame_times = array of time of video frames in msec
#intervals = list of lists of start stop times in msec
def timeinterval2index(intervals, frame_times):
	frame_i = []
	for interval in intervals:
		start, stop = interval
		frame_indices = np.squeeze(np.argwhere(np.logical_and(frame_times >= start, frame_times <= stop)))
		frame_i.append([int(frame_indices[0]), int(frame_indices[-1])])
	return frame_i

def frame2time(intervals, sampling_rate):
	time_intervals = []
	for interval in intervals:
		time_intervals.append([float(i)/float(sampling_rate) for i in interval])
	return time_intervals
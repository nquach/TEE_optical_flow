"""
Custom exceptions for optical flow processing.
"""


class OpticalFlowError(Exception):
	"""Base exception for optical flow processing errors."""
	pass


class DICOMReadError(OpticalFlowError):
	"""Raised when DICOM file cannot be read."""
	pass


class WaveformLoadError(OpticalFlowError):
	"""Raised when waveform file cannot be loaded."""
	pass


class WaveformValidationError(OpticalFlowError):
	"""Raised when waveform validation fails."""
	pass


class OpticalFlowCalculationError(OpticalFlowError):
	"""Raised when optical flow calculation fails."""
	pass


class ConfigurationError(OpticalFlowError):
	"""Raised when configuration is invalid."""
	pass


"""
File I/O module for optical flow data.

This module provides classes for reading/writing HDF5 files, CSV export,
and Pickle serialization.
"""

import os
import pickle as pkl
import h5py
import polars as pl
from typing import List, Optional, Any
from tqdm import tqdm

from optical_flow_utils import safe_makedir


class HDF5Reader:
    """Reader for HDF5 optical flow files."""
    
    def __init__(self, filepath: str, mode: str = 'r'):
        """
        Initialize HDF5 reader.
        
        Args:
            filepath: Path to HDF5 file
            mode: File mode ('r' for read, 'r+' for read-write)
        """
        self.filepath = filepath
        self.mode = mode
        self._file = None
    
    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.filepath, self.mode)
        return self._file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()
        return False
    
    def read_dataset(self, key: str) -> Any:
        """
        Read dataset from HDF5 file.
        
        Args:
            key: Dataset key
        
        Returns:
            Dataset array
        """
        with self as f:
            if key in f:
                return f[key][()]
            else:
                raise KeyError(f"Dataset '{key}' not found in HDF5 file")
    
    def read_attributes(self, key: str) -> dict:
        """
        Read attributes from dataset.
        
        Args:
            key: Dataset key
        
        Returns:
            Dictionary of attributes
        """
        with self as f:
            if key in f:
                return dict(f[key].attrs)
            else:
                raise KeyError(f"Dataset '{key}' not found in HDF5 file")


class HDF5Writer:
    """Writer for HDF5 optical flow files."""
    
    def __init__(self, filepath: str, mode: str = 'w'):
        """
        Initialize HDF5 writer.
        
        Args:
            filepath: Path to HDF5 file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.filepath = filepath
        self.mode = mode
        self._file = None
    
    def __enter__(self):
        """Context manager entry."""
        safe_makedir(os.path.dirname(self.filepath))
        self._file = h5py.File(self.filepath, self.mode)
        return self._file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()
        return False
    
    def write_dataset(self, key: str, data: Any, **attrs):
        """
        Write dataset to HDF5 file.
        
        Args:
            key: Dataset key
            data: Data array to write
            **attrs: Attributes to attach to dataset
        """
        with self as f:
            dset = f.create_dataset(key, data=data)
            for attr_key, attr_value in attrs.items():
                dset.attrs[attr_key] = attr_value


class PickleSerializer:
    """Serializer for Pickle files."""
    
    @staticmethod
    def save(data: Any, filepath: str):
        """
        Save data to pickle file.
        
        Args:
            data: Data to serialize
            filepath: Path to save file
        """
        safe_makedir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pkl.dump(data, f)
    
    @staticmethod
    def load(filepath: str) -> Any:
        """
        Load data from pickle file.
        
        Args:
            filepath: Path to pickle file
        
        Returns:
            Deserialized data
        """
        with open(filepath, 'rb') as f:
            return pkl.load(f)


class CSVExporter:
    """Exporter for CSV files."""
    
    @staticmethod
    def export_dataframe(data_list: List[list], header: List[str], filepath: str):
        """
        Export list of data rows to CSV file.
        
        Args:
            data_list: List of data rows
            header: Column headers
            filepath: Path to save CSV file
        """
        safe_makedir(os.path.dirname(filepath))
        df = pl.DataFrame(data_list, schema=header, orient='row')
        df.write_csv(filepath)
        print(f'Saved CSV file as {filepath}')
    
    @staticmethod
    def aggregate_pkl_files(param_list: List[str], label_list: List[str], save_dir: str):
        """
        Aggregate pickle files and export to CSV.
        
        Args:
            param_list: List of parameter names
            label_list: List of label names
            save_dir: Directory containing pickle files
        """
        for param in param_list:
            for label in label_list:
                save_subdir = os.path.join(save_dir, param + '_' + label)
                pkl_dir = os.path.join(save_subdir, 'pkl_files')
                csv_dir = os.path.join(save_dir, 'csv')
                safe_makedir(csv_dir)
                
                if not os.path.exists(pkl_dir):
                    print(f'Directory {pkl_dir} does not exist, skipping...')
                    continue
                
                file_list = os.listdir(pkl_dir)
                data_list = []
                print(f'Analyzing {pkl_dir}')
                
                for filename in tqdm(file_list):
                    if filename[-3:] == 'pkl':
                        pkl_path = os.path.join(pkl_dir, filename)
                        try:
                            data = PickleSerializer.load(pkl_path)
                            data_list.append(data)
                        except Exception as e:
                            print(f'Error loading {pkl_path}: {e}')
                            continue
                
                if not data_list:
                    print(f'No data found in {pkl_dir}, skipping CSV export...')
                    continue
                
                header = [
                    'Filename', 'MRN', 'FrameRate', 'PixelSpacing', 'HR', 'Frames',
                    'MeanART', 'MaxART', 'MinART', 'MeanCVP', 'MaxCVP', 'MinCVP',
                    'MeanPAP', 'MaxPAP', 'MinPAP',
                    f'ECGTotalPeakSystolic{param.capitalize()}',
                    f'ECGTotalMeanSystolic{param.capitalize()}',
                    f'ECGTotalPeakE{param.capitalize()}', f'ECGTotalMeanE{param.capitalize()}',
                    f'ECGTotalPeakL{param.capitalize()}', f'ECGTotalMeanL{param.capitalize()}',
                    f'ECGTotalPeakA{param.capitalize()}', f'ECGTotalMeanA{param.capitalize()}',
                    f'ECGCardiacCycles{param.capitalize()}',
                    f'ARTTotalPeakSystolic{param.capitalize()}',
                    f'ARTTotalMeanSystolic{param.capitalize()}',
                    f'ARTTotalPeakE{param.capitalize()}', f'ARTTotalMeanE{param.capitalize()}',
                    f'ARTTotalPeakL{param.capitalize()}', f'ARTTotalMeanL{param.capitalize()}',
                    f'ARTTotalPeakA{param.capitalize()}', f'ARTTotalMeanA{param.capitalize()}',
                    f'ARTCardiacCycles{param.capitalize()}',
                    f'ECGRadialPeakSystolic{param.capitalize()}',
                    f'ECGRadialMeanSystolic{param.capitalize()}',
                    f'ECGRadialPeakE{param.capitalize()}', f'ECGRadialMeanE{param.capitalize()}',
                    f'ECGRadialPeakL{param.capitalize()}', f'ECGRadialMeanL{param.capitalize()}',
                    f'ECGRadialPeakA{param.capitalize()}', f'ECGRadialMeanA{param.capitalize()}',
                    f'ECGLongPeakSystolic{param.capitalize()}',
                    f'ECGLongMeanSystolic{param.capitalize()}',
                    f'ECGLongPeakE{param.capitalize()}', f'ECGLongMeanE{param.capitalize()}',
                    f'ECGLongPeakL{param.capitalize()}', f'ECGLongMeanL{param.capitalize()}',
                    f'ECGLongPeakA{param.capitalize()}', f'ECGLongMeanA{param.capitalize()}',
                    f'ECGRadialCardiacCycles{param.capitalize()}',
                    f'ECGLongCardiacCycles{param.capitalize()}',
                    f'ARTRadialPeakSystolic{param.capitalize()}',
                    f'ARTRadialMeanSystolic{param.capitalize()}',
                    f'ARTRadialPeakE{param.capitalize()}', f'ARTRadialMeanE{param.capitalize()}',
                    f'ARTRadialPeakL{param.capitalize()}', f'ARTRadialMeanL{param.capitalize()}',
                    f'ARTRadialPeakA{param.capitalize()}', f'ARTRadialMeanA{param.capitalize()}',
                    f'ARTLongPeakSystolic{param.capitalize()}',
                    f'ARTLongMeanSystolic{param.capitalize()}',
                    f'ARTLongPeakE{param.capitalize()}', f'ARTLongMeanE{param.capitalize()}',
                    f'ARTLongPeakL{param.capitalize()}', f'ARTLongMeanL{param.capitalize()}',
                    f'ARTLongPeakA{param.capitalize()}', f'ARTLongMeanA{param.capitalize()}',
                    f'ARTRadialCardiacCycles{param.capitalize()}',
                    f'ARTLongCardiacCycles{param.capitalize()}'
                ]
                
                csv_name = label + '_' + param + '_data.csv'
                csv_path = os.path.join(csv_dir, csv_name)
                CSVExporter.export_dataframe(data_list, header, csv_path)


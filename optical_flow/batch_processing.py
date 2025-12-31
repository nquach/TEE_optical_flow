"""
Batch processing module for optical flow analysis.

This module provides classes for processing multiple HDF5 files
with progress tracking, error recovery, and parallel processing support.
"""

import os
import traceback
from typing import List, Optional, Callable
from tqdm import tqdm

from optical_flow.optical_flow_utils import safe_makedir
from optical_flow.file_io import PickleSerializer
from optical_flow.optical_flow_dataset import OpticalFlowDataset


class BatchProcessor:
    """Processes multiple HDF5 files in batch."""
    
    def __init__(self, hdf5_folder: str, save_dir: str, verbose: bool = True):
        """
        Initialize batch processor.
        
        Args:
            hdf5_folder: Folder containing HDF5 files
            save_dir: Directory to save results
            verbose: Whether to print progress
        """
        self.hdf5_folder = hdf5_folder
        self.save_dir = save_dir
        self.verbose = verbose
        self.error_list = []
    
    def process_single_file(self, filepath: str, process_func: Callable) -> bool:
        """
        Process a single file with error handling.
        
        Args:
            filepath: Path to HDF5 file
            process_func: Function to process the file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            process_func(filepath)
            return True
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            print(f'Error processing file {filepath}: {e}')
            if filepath not in self.error_list:
                self.error_list.append(filepath)
            return False
    
    def process_chunk(self, file_list: List[str], start_idx: int, end_idx: int,
                     process_func: Callable):
        """
        Process a chunk of files.
        
        Args:
            file_list: List of filenames
            start_idx: Start index
            end_idx: End index
            process_func: Function to process each file
        """
        for i in range(start_idx, end_idx):
            if i >= len(file_list):
                break
            filename = file_list[i]
            if self.verbose:
                print(f'Processing file {i + 1}/{end_idx}: {filename}')
            
            if filename[-4:] == 'hdf5':
                filepath = os.path.join(self.hdf5_folder, filename)
                self.process_single_file(filepath, process_func)
    
    def save_errors(self):
        """Save error list to pickle file."""
        error_dir = os.path.join(self.save_dir, 'errors')
        safe_makedir(error_dir)
        error_path = os.path.join(error_dir, 'error_filelist.pkl')
        PickleSerializer.save(self.error_list, error_path)
        print(f'Total files unable to be processed: {len(self.error_list)}')
        if self.error_list:
            print(f'Files unable to be processed: {self.error_list}')


def analyze_hdf5_folder(hdf5_folder: str, save_dir: str, param_list: List[str],
                        label_list: List[str], process_func: Callable,
                        nchunks: int = 10, chunk_index: int = 0,
                        recalculate: bool = False, verbose: bool = True):
    """
    Analyze folder of HDF5 files in chunks.
    
    Args:
        hdf5_folder: Folder containing HDF5 files
        save_dir: Directory to save results
        param_list: List of parameters to analyze
        label_list: List of labels to analyze
        process_func: Function to process each file
        nchunks: Number of chunks to split files into
        chunk_index: Index of chunk to process
        recalculate: Whether to recalculate existing results
        verbose: Whether to print progress
    """
    file_list = os.listdir(hdf5_folder)
    total_files = len(file_list)
    split_size = total_files // nchunks
    
    processor = BatchProcessor(hdf5_folder, save_dir, verbose=verbose)
    
    start_idx = chunk_index * split_size
    end_idx = (chunk_index + 1) * split_size
    
    processor.process_chunk(file_list, start_idx, end_idx, process_func)
    processor.save_errors()


"""
Caching module for expensive optical flow computations.

This module provides LRU cache functionality for expensive operations
like histogram calculations and centroid computations.
"""

import hashlib
import pickle
from functools import lru_cache, wraps
from typing import Callable, Any, Optional
import numpy as np


def hash_array(arr: np.ndarray) -> str:
    """
    Create hash of numpy array for cache key.
    
    Args:
        arr: Numpy array to hash
    
    Returns:
        Hexadecimal hash string
    """
    return hashlib.md5(arr.tobytes()).hexdigest()


def hash_args(*args, **kwargs) -> str:
    """
    Create hash of function arguments for cache key.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Hexadecimal hash string
    """
    # Convert args and kwargs to a tuple for hashing
    key = (args, tuple(sorted(kwargs.items())))
    return hashlib.md5(pickle.dumps(key)).hexdigest()


def cached_computation(maxsize: int = 128):
    """
    Decorator for caching expensive computations.
    
    Args:
        maxsize: Maximum cache size (LRU cache)
    
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @lru_cache(maxsize=maxsize)
        def cached_wrapper(*args, **kwargs):
            # Convert numpy arrays to hashable format
            new_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    new_args.append(hash_array(arg))
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    new_kwargs[key] = hash_array(value)
                else:
                    new_kwargs[key] = value
            
            return func(*new_args, **new_kwargs)
        
        # Preserve function metadata
        cached_wrapper.__name__ = func.__name__
        cached_wrapper.__doc__ = func.__doc__
        
        return cached_wrapper
    return decorator


class ComputationCache:
    """Manages cache for expensive computations."""
    
    def __init__(self, maxsize: int = 128):
        """
        Initialize computation cache.
        
        Args:
            maxsize: Maximum cache size
        """
        self.maxsize = maxsize
        self._cache = {}
        self._access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest if at capacity
        if len(self._cache) >= self.maxsize and self._access_order:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._access_order.clear()
    
    def invalidate(self, key: str):
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)


# Global cache instance
_global_cache = ComputationCache(maxsize=128)


def get_cache() -> ComputationCache:
    """Get global cache instance."""
    return _global_cache


def clear_cache():
    """Clear global cache."""
    _global_cache.clear()


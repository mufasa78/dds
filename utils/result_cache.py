import os
import json
import time
import hashlib
from pathlib import Path
import threading

class ResultCache:
    """
    Cache for storing and retrieving video processing results
    """
    def __init__(self, cache_dir="cache", max_size_mb=500, expiration_days=7):
        """
        Initialize the result cache
        
        Args:
            cache_dir (str): Directory to store cache files
            max_size_mb (int): Maximum cache size in MB
            expiration_days (int): Number of days after which cache entries expire
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.expiration_seconds = expiration_days * 24 * 60 * 60
        self.cache_lock = threading.Lock()
        self.cache_index = self._load_index()
        
        # Run initial cleanup
        self._cleanup_cache()
    
    def _load_index(self):
        """Load cache index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache index: {e}")
                return {"entries": {}, "size_bytes": 0, "last_cleanup": time.time()}
        else:
            return {"entries": {}, "size_bytes": 0, "last_cleanup": time.time()}
    
    def _save_index(self):
        """Save cache index to disk"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {e}")
    
    def _generate_key(self, video_path, params=None):
        """
        Generate a cache key for a video and processing parameters
        
        Args:
            video_path (str): Path to the video file
            params (dict, optional): Processing parameters
            
        Returns:
            str: Cache key
        """
        # Get video file stats
        try:
            file_stats = os.stat(video_path)
            file_size = file_stats.st_size
            file_mtime = file_stats.st_mtime
        except Exception:
            file_size = 0
            file_mtime = 0
        
        # Create a string with video path, size, modification time, and parameters
        key_str = f"{video_path}|{file_size}|{file_mtime}"
        
        if params:
            # Sort params to ensure consistent keys
            param_str = json.dumps(params, sort_keys=True)
            key_str += f"|{param_str}"
        
        # Hash the key string
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, video_path, params=None):
        """
        Get cached result for a video
        
        Args:
            video_path (str): Path to the video file
            params (dict, optional): Processing parameters
            
        Returns:
            dict: Cached result, or None if not found
        """
        key = self._generate_key(video_path, params)
        
        with self.cache_lock:
            if key in self.cache_index["entries"]:
                entry = self.cache_index["entries"][key]
                
                # Check if entry has expired
                if time.time() - entry["timestamp"] > self.expiration_seconds:
                    print(f"Cache entry for {video_path} has expired")
                    self._remove_entry(key)
                    return None
                
                # Load cached result
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            result = json.load(f)
                        
                        # Update access time
                        entry["last_accessed"] = time.time()
                        entry["access_count"] += 1
                        self._save_index()
                        
                        print(f"Cache hit for {video_path}")
                        return result
                    except Exception as e:
                        print(f"Error loading cached result: {e}")
                        self._remove_entry(key)
                else:
                    # Cache file missing, remove from index
                    print(f"Cache file missing for {video_path}")
                    self._remove_entry(key)
            
            return None
    
    def set(self, video_path, result, params=None):
        """
        Cache result for a video
        
        Args:
            video_path (str): Path to the video file
            result (dict): Processing result
            params (dict, optional): Processing parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = self._generate_key(video_path, params)
        
        with self.cache_lock:
            # Check if we need to clean up the cache
            if self.cache_index["size_bytes"] > self.max_size_bytes:
                self._cleanup_cache()
            
            # Serialize result to JSON
            try:
                result_json = json.dumps(result)
                result_size = len(result_json)
                
                # Check if result is too large
                if result_size > self.max_size_bytes * 0.1:  # Don't allow single entries > 10% of max
                    print(f"Result for {video_path} is too large to cache ({result_size} bytes)")
                    return False
                
                # Save result to cache file
                cache_file = self.cache_dir / f"{key}.json"
                with open(cache_file, 'w') as f:
                    f.write(result_json)
                
                # Update cache index
                self.cache_index["entries"][key] = {
                    "video_path": video_path,
                    "params": params,
                    "timestamp": time.time(),
                    "last_accessed": time.time(),
                    "size_bytes": result_size,
                    "access_count": 1
                }
                
                # Update total cache size
                self.cache_index["size_bytes"] += result_size
                
                # Save index
                self._save_index()
                
                print(f"Cached result for {video_path} ({result_size} bytes)")
                return True
            except Exception as e:
                print(f"Error caching result: {e}")
                return False
    
    def _remove_entry(self, key):
        """
        Remove a cache entry
        
        Args:
            key (str): Cache key
        """
        if key in self.cache_index["entries"]:
            entry = self.cache_index["entries"][key]
            
            # Update total cache size
            self.cache_index["size_bytes"] -= entry.get("size_bytes", 0)
            
            # Remove cache file
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    os.remove(cache_file)
                except Exception as e:
                    print(f"Error removing cache file: {e}")
            
            # Remove from index
            del self.cache_index["entries"][key]
            
            # Save index
            self._save_index()
    
    def _cleanup_cache(self):
        """Clean up expired and least recently used cache entries"""
        with self.cache_lock:
            current_time = time.time()
            entries = list(self.cache_index["entries"].items())
            
            # Sort entries by last accessed time (oldest first)
            entries.sort(key=lambda x: x[1]["last_accessed"])
            
            # Remove expired entries and reduce cache size if needed
            for key, entry in entries:
                # Remove if expired
                if current_time - entry["timestamp"] > self.expiration_seconds:
                    print(f"Removing expired cache entry for {entry['video_path']}")
                    self._remove_entry(key)
                # Remove if cache is still too large
                elif self.cache_index["size_bytes"] > self.max_size_bytes:
                    print(f"Removing least recently used cache entry for {entry['video_path']}")
                    self._remove_entry(key)
                else:
                    # Cache size is within limits
                    break
            
            # Update last cleanup time
            self.cache_index["last_cleanup"] = current_time
            self._save_index()
    
    def clear(self):
        """Clear all cache entries"""
        with self.cache_lock:
            # Remove all cache files
            for key in list(self.cache_index["entries"].keys()):
                self._remove_entry(key)
            
            # Reset cache index
            self.cache_index = {"entries": {}, "size_bytes": 0, "last_cleanup": time.time()}
            self._save_index()
            
            print("Cache cleared")
    
    def get_stats(self):
        """
        Get cache statistics
        
        Returns:
            dict: Cache statistics
        """
        with self.cache_lock:
            return {
                "total_entries": len(self.cache_index["entries"]),
                "size_bytes": self.cache_index["size_bytes"],
                "size_mb": self.cache_index["size_bytes"] / (1024 * 1024),
                "last_cleanup": self.cache_index["last_cleanup"],
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "expiration_days": self.expiration_seconds / (24 * 60 * 60)
            }

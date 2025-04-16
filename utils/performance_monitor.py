import time
import threading
import json
import os
import psutil
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import functools

from utils.logging_system import LoggingSystem

class PerformanceMonitor:
    """
    Monitors and tracks system and application performance
    """
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PerformanceMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, data_dir="performance_data", sampling_interval=5):
        """
        Initialize the performance monitor
        
        Args:
            data_dir (str): Directory to store performance data
            sampling_interval (int): Interval between performance samples in seconds
        """
        # Only initialize once
        if self._initialized:
            return
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sampling_interval = sampling_interval
        
        # Initialize logger
        self.logger = LoggingSystem().performance_logger
        
        # Initialize metrics storage
        self.metrics = {
            "system": {
                "cpu_usage": [],
                "memory_usage": [],
                "disk_usage": [],
                "timestamps": []
            },
            "operations": {},
            "model_performance": {}
        }
        
        # Start monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Initialize operation timing
        self.operation_timers = {}
        
        self._initialized = True
        self.logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start the performance monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring thread"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            self.logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Monitor system performance metrics"""
        while self.monitoring_active:
            try:
                # Get current timestamp
                timestamp = time.time()
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Store metrics
                self.metrics["system"]["cpu_usage"].append(cpu_percent)
                self.metrics["system"]["memory_usage"].append(memory_percent)
                self.metrics["system"]["disk_usage"].append(disk_percent)
                self.metrics["system"]["timestamps"].append(timestamp)
                
                # Limit the number of stored samples to prevent memory issues
                max_samples = 1000
                if len(self.metrics["system"]["timestamps"]) > max_samples:
                    self.metrics["system"]["cpu_usage"] = self.metrics["system"]["cpu_usage"][-max_samples:]
                    self.metrics["system"]["memory_usage"] = self.metrics["system"]["memory_usage"][-max_samples:]
                    self.metrics["system"]["disk_usage"] = self.metrics["system"]["disk_usage"][-max_samples:]
                    self.metrics["system"]["timestamps"] = self.metrics["system"]["timestamps"][-max_samples:]
                
                # Log performance data
                self.logger.info(
                    f"System metrics - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
                )
                
                # Save metrics periodically
                if len(self.metrics["system"]["timestamps"]) % 60 == 0:
                    self.save_metrics()
                
                # Sleep until next sample
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(self.sampling_interval)
    
    def time_operation(self, operation_name):
        """
        Context manager for timing operations
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            context manager: Context manager for timing
        """
        class OperationTimer:
            def __init__(self, monitor, operation_name):
                self.monitor = monitor
                self.operation_name = operation_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.monitor.record_operation_time(self.operation_name, duration)
        
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name, duration):
        """
        Record the time taken for an operation
        
        Args:
            operation_name (str): Name of the operation
            duration (float): Duration in seconds
        """
        if operation_name not in self.metrics["operations"]:
            self.metrics["operations"][operation_name] = {
                "durations": [],
                "timestamps": [],
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "avg_time": 0
            }
        
        # Update metrics
        metrics = self.metrics["operations"][operation_name]
        metrics["durations"].append(duration)
        metrics["timestamps"].append(time.time())
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
        
        # Limit the number of stored samples
        max_samples = 1000
        if len(metrics["durations"]) > max_samples:
            metrics["durations"] = metrics["durations"][-max_samples:]
            metrics["timestamps"] = metrics["timestamps"][-max_samples:]
        
        # Log performance data
        self.logger.info(
            f"Operation '{operation_name}' completed in {duration:.4f}s (avg: {metrics['avg_time']:.4f}s)"
        )
    
    def record_model_performance(self, model_id, metrics_data):
        """
        Record model performance metrics
        
        Args:
            model_id (str): Model identifier
            metrics_data (dict): Performance metrics
        """
        if model_id not in self.metrics["model_performance"]:
            self.metrics["model_performance"][model_id] = {
                "metrics_history": [],
                "timestamps": []
            }
        
        # Update metrics
        self.metrics["model_performance"][model_id]["metrics_history"].append(metrics_data)
        self.metrics["model_performance"][model_id]["timestamps"].append(time.time())
        
        # Limit the number of stored samples
        max_samples = 100
        if len(self.metrics["model_performance"][model_id]["timestamps"]) > max_samples:
            self.metrics["model_performance"][model_id]["metrics_history"] = \
                self.metrics["model_performance"][model_id]["metrics_history"][-max_samples:]
            self.metrics["model_performance"][model_id]["timestamps"] = \
                self.metrics["model_performance"][model_id]["timestamps"][-max_samples:]
        
        # Log performance data
        self.logger.info(f"Model '{model_id}' performance updated: {json.dumps(metrics_data)}")
    
    def save_metrics(self):
        """Save performance metrics to disk"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save system metrics
            system_file = self.data_dir / f"system_metrics_{timestamp}.json"
            with open(system_file, 'w') as f:
                json.dump(self.metrics["system"], f, indent=2)
            
            # Save operation metrics
            operations_file = self.data_dir / f"operation_metrics_{timestamp}.json"
            with open(operations_file, 'w') as f:
                json.dump(self.metrics["operations"], f, indent=2)
            
            # Save model performance metrics
            model_file = self.data_dir / f"model_metrics_{timestamp}.json"
            with open(model_file, 'w') as f:
                json.dump(self.metrics["model_performance"], f, indent=2)
            
            self.logger.info(f"Performance metrics saved to {self.data_dir}")
            
            # Clean up old files
            self._cleanup_old_files()
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {str(e)}")
    
    def _cleanup_old_files(self, max_age_days=7):
        """
        Clean up old performance data files
        
        Args:
            max_age_days (int): Maximum age of files in days
        """
        try:
            # Get current time
            now = datetime.now()
            
            # Get all files in the data directory
            for file_path in self.data_dir.glob("*.json"):
                # Get file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Check if file is older than max_age_days
                if now - mtime > timedelta(days=max_age_days):
                    # Remove file
                    os.remove(file_path)
                    self.logger.info(f"Removed old performance data file: {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up old performance data files: {str(e)}")
    
    def get_system_metrics(self, time_range=None):
        """
        Get system performance metrics
        
        Args:
            time_range (tuple, optional): Time range (start, end) in seconds
            
        Returns:
            dict: System performance metrics
        """
        if not time_range:
            return self.metrics["system"]
        
        # Filter metrics by time range
        start_time, end_time = time_range
        
        # Find indices within the time range
        timestamps = np.array(self.metrics["system"]["timestamps"])
        indices = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
        
        # Extract metrics within the time range
        filtered_metrics = {
            "cpu_usage": [self.metrics["system"]["cpu_usage"][i] for i in indices],
            "memory_usage": [self.metrics["system"]["memory_usage"][i] for i in indices],
            "disk_usage": [self.metrics["system"]["disk_usage"][i] for i in indices],
            "timestamps": [self.metrics["system"]["timestamps"][i] for i in indices]
        }
        
        return filtered_metrics
    
    def get_operation_metrics(self, operation_name=None):
        """
        Get operation performance metrics
        
        Args:
            operation_name (str, optional): Name of the operation
            
        Returns:
            dict: Operation performance metrics
        """
        if operation_name:
            return self.metrics["operations"].get(operation_name, {})
        else:
            return self.metrics["operations"]
    
    def get_model_metrics(self, model_id=None):
        """
        Get model performance metrics
        
        Args:
            model_id (str, optional): Model identifier
            
        Returns:
            dict: Model performance metrics
        """
        if model_id:
            return self.metrics["model_performance"].get(model_id, {})
        else:
            return self.metrics["model_performance"]
    
    def get_performance_summary(self):
        """
        Get a summary of performance metrics
        
        Returns:
            dict: Performance summary
        """
        summary = {
            "system": {
                "cpu_usage_avg": 0,
                "memory_usage_avg": 0,
                "disk_usage_avg": 0
            },
            "operations": {},
            "models": {}
        }
        
        # Calculate system metrics averages
        if self.metrics["system"]["cpu_usage"]:
            summary["system"]["cpu_usage_avg"] = sum(self.metrics["system"]["cpu_usage"]) / len(self.metrics["system"]["cpu_usage"])
            summary["system"]["memory_usage_avg"] = sum(self.metrics["system"]["memory_usage"]) / len(self.metrics["system"]["memory_usage"])
            summary["system"]["disk_usage_avg"] = sum(self.metrics["system"]["disk_usage"]) / len(self.metrics["system"]["disk_usage"])
        
        # Summarize operation metrics
        for op_name, op_metrics in self.metrics["operations"].items():
            summary["operations"][op_name] = {
                "count": op_metrics["count"],
                "avg_time": op_metrics["avg_time"],
                "min_time": op_metrics["min_time"],
                "max_time": op_metrics["max_time"]
            }
        
        # Summarize model metrics
        for model_id, model_metrics in self.metrics["model_performance"].items():
            if model_metrics["metrics_history"]:
                # Get the most recent metrics
                summary["models"][model_id] = model_metrics["metrics_history"][-1]
        
        return summary

# Decorator for timing functions
def time_function(operation_name=None):
    """
    Decorator for timing function execution
    
    Args:
        operation_name (str, optional): Name of the operation
        
    Returns:
        function: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get operation name
            op_name = operation_name
            if op_name is None:
                op_name = func.__name__
            
            # Get performance monitor
            monitor = PerformanceMonitor()
            
            # Time the function
            with monitor.time_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

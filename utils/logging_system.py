import os
import logging
import time
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler
import threading

class LoggingSystem:
    """
    Comprehensive logging system for the application
    """
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggingSystem, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, log_dir="logs", log_level=logging.INFO, max_size_mb=10, backup_count=5):
        """
        Initialize the logging system
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level
            max_size_mb (int): Maximum size of log files in MB
            backup_count (int): Number of backup log files to keep
        """
        # Only initialize once
        if self._initialized:
            return
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create loggers
        self.app_logger = self._setup_logger(
            "app", 
            self.log_dir / "app.log", 
            log_level, 
            max_size_mb, 
            backup_count
        )
        
        self.model_logger = self._setup_logger(
            "model", 
            self.log_dir / "model.log", 
            log_level, 
            max_size_mb, 
            backup_count
        )
        
        self.performance_logger = self._setup_logger(
            "performance", 
            self.log_dir / "performance.log", 
            log_level, 
            max_size_mb, 
            backup_count
        )
        
        self.user_logger = self._setup_logger(
            "user", 
            self.log_dir / "user.log", 
            log_level, 
            max_size_mb, 
            backup_count
        )
        
        self.error_logger = self._setup_logger(
            "error", 
            self.log_dir / "error.log", 
            logging.ERROR,  # Always log errors at ERROR level
            max_size_mb, 
            backup_count
        )
        
        # Create a console handler for the app logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.app_logger.addHandler(console_handler)
        
        self.app_logger.info("Logging system initialized")
        self._initialized = True
    
    def _setup_logger(self, name, log_file, level, max_size_mb, backup_count):
        """
        Set up a logger with file rotation
        
        Args:
            name (str): Logger name
            log_file (Path): Path to log file
            level (int): Logging level
            max_size_mb (int): Maximum size of log file in MB
            backup_count (int): Number of backup log files to keep
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create file handler with rotation
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def log_app(self, level, message, **kwargs):
        """
        Log an application message
        
        Args:
            level (int): Logging level
            message (str): Log message
            **kwargs: Additional data to log
        """
        if kwargs:
            message = f"{message} - {json.dumps(kwargs)}"
        self.app_logger.log(level, message)
        
        # Also log errors to the error logger
        if level >= logging.ERROR:
            self.error_logger.log(level, f"APP: {message}")
    
    def log_model(self, level, message, model_info=None, **kwargs):
        """
        Log a model-related message
        
        Args:
            level (int): Logging level
            message (str): Log message
            model_info (dict, optional): Model information
            **kwargs: Additional data to log
        """
        log_data = kwargs.copy()
        if model_info:
            log_data["model_info"] = model_info
        
        if log_data:
            message = f"{message} - {json.dumps(log_data)}"
        self.model_logger.log(level, message)
        
        # Also log errors to the error logger
        if level >= logging.ERROR:
            self.error_logger.log(level, f"MODEL: {message}")
    
    def log_performance(self, operation, duration, details=None):
        """
        Log performance metrics
        
        Args:
            operation (str): Operation being measured
            duration (float): Duration in seconds
            details (dict, optional): Additional performance details
        """
        log_data = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": time.time()
        }
        
        if details:
            log_data.update(details)
        
        self.performance_logger.info(json.dumps(log_data))
    
    def log_user_action(self, action, user_id=None, details=None):
        """
        Log user actions
        
        Args:
            action (str): User action
            user_id (str, optional): User identifier
            details (dict, optional): Additional action details
        """
        log_data = {
            "action": action,
            "timestamp": time.time()
        }
        
        if user_id:
            log_data["user_id"] = user_id
        
        if details:
            log_data.update(details)
        
        self.user_logger.info(json.dumps(log_data))
    
    def log_error(self, error, context=None):
        """
        Log an error
        
        Args:
            error (Exception): Error to log
            context (dict, optional): Error context
        """
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        }
        
        if context:
            log_data["context"] = context
        
        self.error_logger.error(json.dumps(log_data))
    
    def get_recent_logs(self, logger_name="app", lines=100):
        """
        Get recent log entries
        
        Args:
            logger_name (str): Name of the logger
            lines (int): Number of lines to retrieve
            
        Returns:
            list: Recent log entries
        """
        log_file = self.log_dir / f"{logger_name}.log"
        
        if not log_file.exists():
            return []
        
        try:
            with open(log_file, 'r') as f:
                # Read the last 'lines' lines
                return self._tail(f, lines)
        except Exception as e:
            self.log_error(e, {"context": f"Reading recent logs from {log_file}"})
            return []
    
    def _tail(self, file, lines):
        """
        Read the last 'lines' lines from a file
        
        Args:
            file: File object
            lines (int): Number of lines to read
            
        Returns:
            list: Last 'lines' lines from the file
        """
        # Move to the end of the file
        file.seek(0, os.SEEK_END)
        
        # Buffer to store lines
        buffer = []
        
        # Read the file backwards
        position = file.tell()
        line_count = 0
        
        while position > 0 and line_count < lines:
            # Move back one character
            position -= 1
            file.seek(position)
            
            # Read one character
            char = file.read(1)
            
            # If we hit a newline, we've found a complete line
            if char == '\n':
                line_count += 1
                
                # Read the line
                line = file.readline()
                buffer.append(line.strip())
        
        # If we're at the beginning of the file and haven't found enough lines
        if position == 0 and line_count < lines:
            # Read the first line
            file.seek(0)
            line = file.readline()
            buffer.append(line.strip())
        
        # Reverse the buffer to get lines in the correct order
        return buffer[::-1]

import cv2
import numpy as np
import os
import time
from pathlib import Path
import tempfile

class VideoPreprocessor:
    """
    Handles video preprocessing with various enhancement options
    """
    def __init__(self):
        """Initialize the video preprocessor"""
        self.supported_methods = {
            "none": self._no_preprocessing,
            "denoise": self._denoise,
            "enhance_contrast": self._enhance_contrast,
            "sharpen": self._sharpen,
            "normalize": self._normalize,
            "grayscale": self._grayscale,
            "resize": self._resize
        }
    
    def preprocess_frame(self, frame, methods=None, params=None):
        """
        Apply preprocessing methods to a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            methods (list): List of preprocessing methods to apply
            params (dict): Parameters for preprocessing methods
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        if frame is None:
            return None
        
        if methods is None:
            return frame
        
        if params is None:
            params = {}
        
        processed_frame = frame.copy()
        
        for method in methods:
            if method in self.supported_methods:
                method_params = params.get(method, {})
                processed_frame = self.supported_methods[method](processed_frame, **method_params)
            else:
                print(f"Warning: Unsupported preprocessing method '{method}'")
        
        return processed_frame
    
    def preprocess_video(self, video_path, output_path=None, methods=None, params=None):
        """
        Preprocess an entire video
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save preprocessed video
            methods (list): List of preprocessing methods to apply
            params (dict): Parameters for preprocessing methods
            
        Returns:
            str: Path to preprocessed video
        """
        if methods is None or not methods:
            print("No preprocessing methods specified, returning original video")
            return video_path
        
        # Create temporary output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"preprocessed_{int(time.time())}.mp4")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return video_path
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply preprocessing
            processed_frame = self.preprocess_frame(frame, methods, params)
            
            # Write to output video
            out.write(processed_frame)
            
            # Print progress
            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == 1 or frame_idx == frame_count:
                print(f"Preprocessing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Preprocessed video saved to {output_path}")
        return output_path
    
    def _no_preprocessing(self, frame, **kwargs):
        """No preprocessing, return frame as is"""
        return frame
    
    def _denoise(self, frame, strength=10, **kwargs):
        """
        Apply denoising to a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            strength (int): Denoising strength (higher = more denoising)
            
        Returns:
            numpy.ndarray: Denoised frame
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
    
    def _enhance_contrast(self, frame, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
        """
        Enhance contrast using CLAHE
        
        Args:
            frame (numpy.ndarray): Input frame
            clip_limit (float): Contrast limit
            tile_grid_size (tuple): Size of grid for histogram equalization
            
        Returns:
            numpy.ndarray: Contrast-enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def _sharpen(self, frame, kernel_size=5, sigma=1.0, amount=1.0, **kwargs):
        """
        Sharpen a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            kernel_size (int): Size of Gaussian kernel
            sigma (float): Sigma of Gaussian kernel
            amount (float): Sharpening amount
            
        Returns:
            numpy.ndarray: Sharpened frame
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
        
        # Subtract blurred image from original
        sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def _normalize(self, frame, **kwargs):
        """
        Normalize frame pixel values
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Normalized frame
        """
        normalized = np.zeros(frame.shape, dtype=np.float32)
        normalized = cv2.normalize(frame, normalized, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(normalized)
    
    def _grayscale(self, frame, **kwargs):
        """
        Convert frame to grayscale
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Grayscale frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    
    def _resize(self, frame, width=None, height=None, scale=None, **kwargs):
        """
        Resize a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            width (int, optional): Target width
            height (int, optional): Target height
            scale (float, optional): Scale factor
            
        Returns:
            numpy.ndarray: Resized frame
        """
        if scale is not None:
            return cv2.resize(frame, None, fx=scale, fy=scale)
        elif width is not None and height is not None:
            return cv2.resize(frame, (width, height))
        else:
            return frame

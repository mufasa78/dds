import numpy as np
import cv2
import os
from pathlib import Path

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        """
        Initialize a simplified face detector for demonstration purposes
        
        Args:
            min_confidence (float): Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        print("Initializing simplified face detection for demonstration purposes.")
    
    def detect_faces(self, image):
        """
        Detect faces in an image (simplified version that just returns a center face box)
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            list: List of face bounding boxes [x1, y1, x2, y2]
        """
        # Get image dimensions
        (h, w) = image.shape[:2]
        
        # Calculate face position based on image center
        center_x, center_y = w // 2, h // 2
        face_size = min(w, h) // 3
        
        # Create a simple box around the center of the image
        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(w, center_x + face_size // 2)
        y2 = min(h, center_y + face_size // 2)
        
        # Return a single face bounding box
        print("Using simplified face detection for demonstration purposes.")
        return [[x1, y1, x2, y2]]
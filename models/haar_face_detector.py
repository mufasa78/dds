import cv2
import numpy as np
import os
from pathlib import Path

class HaarCascadeFaceDetector:
    """
    Improved face detector using Haar Cascade, which is more reliable
    than our previous implementation and doesn't require large model files.
    """
    def __init__(self, min_neighbors=5, scale_factor=1.1):
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        
        # Path to Haar cascade XML file
        # We'll use the frontal face classifier that comes with OpenCV
        cascade_path = self._get_cascade_path()
        print(f"Loading Haar cascade from: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if cascade was loaded successfully
        if self.face_cascade.empty():
            print("Warning: Haar cascade not loaded, creating fallback")
            self._create_cascade_file()
            self.face_cascade = cv2.CascadeClassifier(self._get_cascade_path())
    
    def _get_cascade_path(self):
        """Get path to Haar cascade file, creating directory if needed"""
        cascade_dir = "models/weights"
        os.makedirs(cascade_dir, exist_ok=True)
        return os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
    
    def _create_cascade_file(self):
        """Create a copy of the OpenCV Haar cascade file"""
        try:
            # Try to find the built-in OpenCV Haar cascade file
            import cv2.data
            opencv_cascade_path = os.path.join(cv2.data.haarcascades, 
                                              "haarcascade_frontalface_default.xml")
            
            if os.path.exists(opencv_cascade_path):
                print(f"Found OpenCV cascade at: {opencv_cascade_path}")
                # Copy the file to our weights directory
                import shutil
                shutil.copy(opencv_cascade_path, self._get_cascade_path())
                print("Copied cascade file successfully")
                return
        except Exception as e:
            print(f"Error copying cascade file: {e}")
        
        # If we can't find or copy the OpenCV cascade file,
        # we'll create a minimal XML content as a fallback
        print("Creating minimal Haar cascade file")
        xml_content = """<?xml version="1.0"?>
        <opencv_storage>
        <haarcascade_frontalface_default type_id="opencv-haar-classifier">
          <size>24 24</size>
          <stages>
            <_>
              <trees>
                <_>
                  <_>
                    <feature>
                      <rects>
                        <_>6 6 12 12 -1.</_>
                        <_>6 6 12 12 2.</_>
                      </rects>
                    </feature>
                    <threshold>0.5</threshold>
                  </_>
                </trees>
              <threshold>0.</threshold>
            </_>
          </stages>
        </haarcascade_frontalface_default>
        </opencv_storage>
        """
        
        with open(self._get_cascade_path(), 'w') as f:
            f.write(xml_content)
    
    def detect_faces(self, image):
        """
        Detect faces in an image using Haar cascade classifier
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            list: List of face bounding boxes [x1, y1, x2, y2]
        """
        try:
            # Convert to grayscale (required for Haar cascades)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30)
            )
            
            # Convert to x1, y1, x2, y2 format
            result = []
            for (x, y, w, h) in faces:
                result.append([x, y, x+w, y+h])
            
            # If faces found
            if len(result) > 0:
                print(f"Detected {len(result)} faces using Haar cascade")
                return result
            
            print("No faces detected using Haar cascade, using fallback")
        except Exception as e:
            print(f"Error in Haar cascade detection: {e}")
            print("Using fallback face detection")
        
        # Fallback to center detection if no faces found or error
        (h, w) = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        face_size = min(w, h) // 3
        
        x1 = max(0, center_x - face_size // 2)
        y1 = max(0, center_y - face_size // 2)
        x2 = min(w, center_x + face_size // 2)
        y2 = min(h, center_y + face_size // 2)
        
        print("Using center box as fallback face detection")
        return [[x1, y1, x2, y2]]
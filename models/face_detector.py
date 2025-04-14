import numpy as np
import cv2

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        """
        Initialize face detector based on OpenCV's DNN face detection model
        
        Args:
            min_confidence (float): Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        
        # Path to pre-trained face detection model
        prototxt_path = "models/weights/deploy.prototxt"
        caffemodel_path = "models/weights/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Create directories if they don't exist
        import os
        os.makedirs(os.path.dirname(prototxt_path), exist_ok=True)
        
        # Download models if they don't exist
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            self._download_models(prototxt_path, caffemodel_path)
        
        # Load face detection model
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    def _download_models(self, prototxt_path, caffemodel_path):
        """
        Download the pre-trained face detection models
        
        Args:
            prototxt_path (str): Path to save the prototxt file
            caffemodel_path (str): Path to save the caffemodel file
        """
        import urllib.request
        
        # URLs for the models
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        print("Downloading face detection models...")
        
        # Download prototxt
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        # Download caffemodel
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        
        print("Download complete!")
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            
        Returns:
            list: List of face bounding boxes [x1, y1, x2, y2]
        """
        # Get image dimensions
        (h, w) = image.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and get the detections
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        # List to store face bounding boxes
        faces = []
        
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.min_confidence:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure the bounding boxes fall within the dimensions of the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                    continue
                
                # Add face to list
                faces.append([x1, y1, x2, y2])
        
        return faces

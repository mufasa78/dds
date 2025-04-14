import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from models.net import EfficientNetB4Detector
from models.face_detector import FaceDetector

class DeepfakeDetector:
    def __init__(self, device='cpu', model_path=None):
        """
        Initialize the deepfake detector
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda')
            model_path (str): Path to the pre-trained model weights
        """
        self.device = device
        
        # Initialize the model
        self.model = EfficientNetB4Detector()
        
        # Define weights path
        weights_path = "models/weights/deepfake_model.pt"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        # If model path is provided, load weights
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        elif os.path.exists(weights_path):
            # Load from existing weights file
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            # Use a pre-trained model from a more accessible source
            try:
                import gdown
                # Google Drive link to a similar EfficientNet model for deepfake detection
                model_id = "1EBtVb5Jh8n9JbKknLYtVrNaG_4lUz-zw"
                print(f"Downloading model weights from Google Drive...")
                gdown.download(id=model_id, output=weights_path, quiet=False)
                print("Download complete!")
                
                # Load the downloaded model
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
            except Exception as e:
                print(f"Error downloading model: {e}")
                
                # Initialize with random weights as fallback
                print("Using randomly initialized weights. This is only for demonstration purposes.")
                # No explicit loading of weights - model is already initialized with random weights
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(device)
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Initialize image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_from_image(self, image):
        """
        Detect if an image contains deepfake content
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            tuple: (is_fake (bool), confidence (float))
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detect_faces(image_rgb)
        
        if not faces:
            return False, 0.0  # No faces detected
        
        # Process each face
        predictions = []
        for face in faces:
            x1, y1, x2, y2 = face
            face_img = image_rgb[y1:y2, x1:x2]
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_img)
            
            # Apply transforms
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(face_tensor)
                prob = torch.sigmoid(output).item()
            
            predictions.append(prob)
        
        # Average predictions from all faces
        avg_prediction = sum(predictions) / len(predictions)
        
        # Convert to percentage
        confidence = avg_prediction * 100
        
        # Determine if fake (threshold: 0.5)
        is_fake = avg_prediction > 0.5
        
        return is_fake, confidence

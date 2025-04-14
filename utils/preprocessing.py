import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_frame(frame, target_size=(256, 256)):
    """
    Preprocess a frame for the deepfake detection model
    
    Args:
        frame (numpy.ndarray): Input frame in BGR format
        target_size (tuple): Target size for resizing
        
    Returns:
        torch.Tensor: Preprocessed frame tensor
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform the image
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def normalize_face(face, target_size=(256, 256)):
    """
    Normalize a face image for the deepfake detection model
    
    Args:
        face (numpy.ndarray): Face image in RGB format
        target_size (tuple): Target size for resizing
        
    Returns:
        torch.Tensor: Normalized face tensor
    """
    # Convert to PIL Image
    face_pil = Image.fromarray(face)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    face_tensor = transform(face_pil).unsqueeze(0)
    
    return face_tensor

def enhance_face(face):
    """
    Enhance face image quality for better detection
    
    Args:
        face (numpy.ndarray): Face image
        
    Returns:
        numpy.ndarray: Enhanced face image
    """
    # Apply some basic enhancements
    
    # Convert to float32
    face_float = face.astype(np.float32) / 255.0
    
    # Apply gamma correction
    gamma = 1.2
    face_gamma = np.power(face_float, 1.0/gamma)
    
    # Apply contrast stretching
    p5 = np.percentile(face_gamma, 5)
    p95 = np.percentile(face_gamma, 95)
    face_stretched = np.clip((face_gamma - p5) / (p95 - p5 + 1e-6), 0, 1)
    
    # Convert back to uint8
    face_enhanced = (face_stretched * 255).astype(np.uint8)
    
    return face_enhanced

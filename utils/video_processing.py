import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil

def extract_frames(video_path, max_frames=30):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: List of extracted frames
    """
    frames = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate frame extraction interval
    if frame_count <= max_frames:
        # Extract all frames if there are fewer than max_frames
        interval = 1
    else:
        # Sample frames evenly throughout the video
        interval = frame_count // max_frames
    
    # Extract frames
    count = 0
    frame_idx = 0
    
    while count < max_frames and frame_idx < frame_count:
        # Set position to the next frame to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Add the frame to the list
        frames.append(frame)
        
        # Update counters
        count += 1
        frame_idx += interval
    
    # Release the video capture object
    cap.release()
    
    return frames

def process_video(video_path, detector, frames_to_sample=20):
    """
    Process a video and determine if it contains deepfake content
    
    Args:
        video_path (str): Path to the video file
        detector: The deepfake detector model
        frames_to_sample (int): Number of frames to sample for analysis
        
    Returns:
        tuple: (result (str), confidence (float))
    """
    try:
        # Extract frames from the video
        frames = extract_frames(video_path, max_frames=frames_to_sample)
        
        if not frames:
            return "Error: No frames could be extracted", 0.0
        
        # Process each frame
        fake_scores = []
        real_scores = []
        
        for frame in frames:
            # Detect if the frame is fake
            is_fake, confidence = detector.detect_from_image(frame)
            
            if is_fake:
                fake_scores.append(confidence)
            else:
                real_scores.append(100 - confidence)
        
        # Calculate final prediction
        if not fake_scores and not real_scores:
            return "Error: Could not process frames", 0.0
        
        # If we have both fake and real predictions
        if fake_scores and real_scores:
            avg_fake = sum(fake_scores) / len(fake_scores)
            avg_real = sum(real_scores) / len(real_scores)
            
            if avg_fake > avg_real:
                return "Fake", avg_fake
            else:
                return "Real", avg_real
        
        # If we only have fake predictions
        elif fake_scores:
            avg_fake = sum(fake_scores) / len(fake_scores)
            return "Fake", avg_fake
        
        # If we only have real predictions
        else:
            avg_real = sum(real_scores) / len(real_scores)
            return "Real", avg_real
            
    except Exception as e:
        return f"Error: {str(e)}", 0.0

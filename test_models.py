#!/usr/bin/env python
"""
Test script for the Multilingual Deepfake Detection System models.
This script tests if the models are working correctly.
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        import cv2
        import numpy as np
        import streamlit
        import flask
        import PIL
        print("✓ All required packages imported successfully.")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_torch():
    """Test if PyTorch is working correctly."""
    print("\nTesting PyTorch...")
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Simple tensor operation
        x = torch.rand(5, 3)
        y = torch.rand(5, 3)
        z = x + y
        print("✓ PyTorch tensor operations working correctly.")
        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def test_opencv():
    """Test if OpenCV is working correctly."""
    print("\nTesting OpenCV...")
    try:
        print(f"OpenCV version: {cv2.__version__}")
        
        # Create a simple image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (0, 255, 0), -1)
        
        # Test face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("✗ Failed to load face cascade classifier.")
            return False
        
        print("✓ OpenCV working correctly.")
        return True
    except Exception as e:
        print(f"✗ OpenCV error: {e}")
        return False

def test_models():
    """Test if the models can be loaded."""
    print("\nTesting model loading...")
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            print("✗ Models directory not found.")
            return False
        
        # Import model classes
        sys.path.append('.')
        from models.detector import DeepfakeDetector
        from models.simple_model import SimpleDeepfakeDetector
        
        # Try to initialize a simple model
        model = SimpleDeepfakeDetector()
        print("✓ Model classes imported successfully.")
        
        # Check if model files exist
        model_files = list(Path('models').glob('*.pth'))
        if not model_files:
            print("ℹ No model files found. Models will be downloaded when needed.")
        else:
            print(f"✓ Found {len(model_files)} model files.")
        
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def test_data():
    """Test if the data files can be loaded."""
    print("\nTesting data loading...")
    try:
        # Check if data directory exists
        if not os.path.exists('data'):
            print("✗ Data directory not found.")
            return False
        
        # Import data modules
        sys.path.append('.')
        from data.examples import EXAMPLES
        from data.translations import TRANSLATIONS
        
        # Check examples
        print(f"✓ Found {len(EXAMPLES)} example videos.")
        
        # Check translations
        print(f"✓ Found translations for {len(TRANSLATIONS)} text elements.")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("Multilingual Deepfake Detection System - Model Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    imports_ok = test_imports()
    torch_ok = test_torch()
    opencv_ok = test_opencv()
    models_ok = test_models()
    data_ok = test_data()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"PyTorch: {'✓' if torch_ok else '✗'}")
    print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
    print(f"Models: {'✓' if models_ok else '✗'}")
    print(f"Data: {'✓' if data_ok else '✗'}")
    
    # Overall result
    all_ok = imports_ok and torch_ok and opencv_ok and models_ok and data_ok
    print("\nOverall result:", "✓ All tests passed!" if all_ok else "✗ Some tests failed.")
    
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds.")
    print("=" * 50)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

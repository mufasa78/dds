import os
import time
import tempfile
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from models.detector import DeepfakeDetector
from utils.video_processing import extract_frames, process_video
from utils.language import get_translation, LANGUAGES

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource
def load_detector():
    """Load the deepfake detector model"""
    detector = DeepfakeDetector(device='cpu')
    return detector

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary location and return the path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def main():
    # Load detector model
    detector = load_detector()
    
    # Language selection
    language = st.sidebar.selectbox(
        "Language / 语言",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: f"{LANGUAGES[x]['native']} ({LANGUAGES[x]['english']})"
    )
    t = get_translation(language)
    
    # Main header
    st.title(t("deepfake_detection_system"))
    st.write(t("system_description"))
    
    # File uploader for videos
    uploaded_files = st.file_uploader(
        t("upload_videos"), 
        type=["mp4", "avi", "mov", "mkv"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button(t("analyze_videos")):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {}
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(t("processing_video").format(file=uploaded_file.name))
                
                # Save uploaded file temporarily
                temp_path = save_uploaded_file(uploaded_file)
                
                if temp_path:
                    try:
                        # Process the video and get prediction
                        prediction, confidence = process_video(temp_path, detector)
                        
                        # Store result
                        results[uploaded_file.name] = {
                            'prediction': prediction,
                            'confidence': confidence
                        }
                        
                        # Clean up temporary file
                        os.unlink(temp_path)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        results[uploaded_file.name] = {
                            'prediction': t("error"),
                            'confidence': 0.0
                        }
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display results
            status_text.text(t("analysis_complete"))
            
            st.subheader(t("results"))
            for filename, result in results.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{filename}**")
                with col2:
                    if result['prediction'] == t("error"):
                        st.error(t("error"))
                    elif result['prediction'] == t("real"):
                        st.success(t("real"))
                    else:
                        st.error(t("fake"))
                with col3:
                    if result['prediction'] != t("error"):
                        st.write(f"{t('confidence')}: {result['confidence']:.2f}%")
    
    # Information about the model
    with st.expander(t("about_model")):
        st.write(t("model_info"))
        st.write(t("datasets_used"))
        st.write(t("performance_metrics"))
    
    # Training guide
    with st.expander(t("training_guide")):
        st.write(t("training_guide_description"))
        st.write(t("guide_instruction"))
        
    # Footer
    st.markdown("---")
    st.markdown(t("footer_text"))

if __name__ == "__main__":
    main()

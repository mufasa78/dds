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
from models.haar_face_detector import HaarCascadeFaceDetector
from models.simple_model import SimpleDeepfakeDetector
from utils.video_processing import extract_frames, process_video
from utils.enhanced_video_processing import extract_frames_optimized, enhanced_process_video
from utils.language import get_translation, LANGUAGES
from utils.example_generator import generate_placeholder_examples
from data.examples import EXAMPLES

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource
def load_detector():
    """Load the deepfake detector model"""
    try:
        # First try to load the regular model
        st.info("Loading deepfake detection model...")
        detector = DeepfakeDetector(device='cpu')
        st.success("✅ Model loaded successfully")
        return detector
    except Exception as e:
        st.warning(f"Failed to load EfficientNet model: {e}")
        try:
            # Fallback to the simplified model
            st.info("Trying to load simplified model...")
            # Create a SimpleDeepfakeDetector instance
            model = SimpleDeepfakeDetector(num_classes=1)
            
            # Use the model in the DeepfakeDetector class
            detector = DeepfakeDetector(device='cpu', custom_model=model)
            st.success("✅ Simplified model loaded successfully")
            return detector
        except Exception as e2:
            st.error(f"Failed to load simplified model: {e2}")
            st.error("Using placeholder detection logic as fallback")
            # Return a minimal working detector as last resort
            return DeepfakeDetector(device='cpu', use_fallback=True)

# Initialize the face detector
@st.cache_resource
def load_face_detector():
    """Load the face detector"""
    try:
        return HaarCascadeFaceDetector()
    except Exception as e:
        st.warning(f"Error loading Haar cascade face detector: {e}")
        # Return a simple face detector as fallback
        from models.face_detector import FaceDetector
        return FaceDetector()

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
    # Generate example videos if needed
    with st.spinner("Initializing examples..."):
        generate_placeholder_examples()
    
    # Load detector model and face detector
    detector = load_detector()
    face_detector = load_face_detector()
    
    # Language selection
    language = st.sidebar.selectbox(
        "Language / 语言",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: f"{LANGUAGES[x]['native_name']}"
    )
    t = get_translation(language)
    
    # Main header
    st.title(t("app_title"))
    st.write(t("app_description"))
    
    # Create tabs for upload and examples
    upload_tab, examples_tab, about_tab = st.tabs([
        t("upload_title"), 
        t("examples_title"),
        t("about_title")
    ])
    
    # Upload tab
    with upload_tab:
        # File uploader for videos
        uploaded_files = st.file_uploader(
            t("upload_button"), 
            type=["mp4", "avi", "mov", "mkv"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button(t("analyze_button")):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"{t('analyzing_video')} {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    temp_path = save_uploaded_file(uploaded_file)
                    
                    if temp_path:
                        try:
                            # Process the video with enhanced processing
                            result, confidence, stats = enhanced_process_video(
                                temp_path, 
                                detector, 
                                frames_to_sample=20,
                                sampling_strategy='scene_change'
                            )
                            
                            # Store result with stats
                            results[uploaded_file.name] = {
                                'prediction': result,
                                'confidence': confidence,
                                'stats': stats
                            }
                            
                            # Clean up temporary file
                            os.unlink(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            results[uploaded_file.name] = {
                                'prediction': t("error_processing"),
                                'confidence': 0.0,
                                'stats': {'error': str(e)}
                            }
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                status_text.text(t("analysis_complete"))
                
                st.subheader(t("results_title"))
                for filename, result in results.items():
                    with st.expander(f"**{filename}**", expanded=True):
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if result['prediction'] == t("error_processing"):
                                st.error(t("error_processing"))
                            elif result['prediction'] == "Real":
                                st.success(t("result_real"))
                            else:
                                st.error(t("result_fake"))
                            
                            if result['prediction'] != t("error_processing"):
                                st.write(f"**{t('confidence')}:** {result['confidence']:.2f}%")
                                
                            # Display stats if available
                            if 'stats' in result and isinstance(result['stats'], dict):
                                stats = result['stats']
                                if 'processing_time' in stats:
                                    st.write(f"**{t('processing_time')}:** {stats['processing_time']:.2f} {t('seconds')}")
                                if 'frames_sampled' in stats:
                                    st.write(f"**{t('frames_analyzed')}:** {stats['frames_sampled']}")
                                if 'sampling_strategy' in stats:
                                    st.write(f"**Sampling Strategy:** {stats['sampling_strategy']}")
    
    # Examples tab
    with examples_tab:
        st.subheader(t("examples_title"))
        
        # Display examples as a grid of cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            example_id = "example1"
            example = EXAMPLES[example_id]
            st.image(example["thumbnail"], use_container_width=True)
            st.write(f"**{example['name'][language]}**")
            st.write(example["description"][language])
            if st.button(t("analyze_button"), key=f"analyze_{example_id}"):
                with st.spinner(t("processing")):
                    # Check if the example file exists
                    if os.path.exists(example["path"]):
                        # Process the video
                        result, confidence, stats = enhanced_process_video(
                            example["path"], 
                            detector,
                            frames_to_sample=20,
                            sampling_strategy='uniform'
                        )
                        
                        # Display result
                        if result == "Real":
                            st.success(t("result_real"))
                        else:
                            st.error(t("result_fake"))
                        
                        st.write(f"**{t('confidence')}:** {confidence:.2f}%")
                        st.write(f"**{t('processing_time')}:** {stats['processing_time']:.2f} {t('seconds')}")
                    else:
                        st.error(f"Example file not found: {example['path']}")
        
        with col2:
            example_id = "example2"
            example = EXAMPLES[example_id]
            st.image(example["thumbnail"], use_container_width=True)
            st.write(f"**{example['name'][language]}**")
            st.write(example["description"][language])
            if st.button(t("analyze_button"), key=f"analyze_{example_id}"):
                with st.spinner(t("processing")):
                    # Check if the example file exists
                    if os.path.exists(example["path"]):
                        # Process the video
                        result, confidence, stats = enhanced_process_video(
                            example["path"], 
                            detector,
                            frames_to_sample=20,
                            sampling_strategy='uniform'
                        )
                        
                        # Display result
                        if result == "Real":
                            st.success(t("result_real"))
                        else:
                            st.error(t("result_fake"))
                        
                        st.write(f"**{t('confidence')}:** {confidence:.2f}%")
                        st.write(f"**{t('processing_time')}:** {stats['processing_time']:.2f} {t('seconds')}")
                    else:
                        st.error(f"Example file not found: {example['path']}")
        
        with col3:
            example_id = "example3"
            example = EXAMPLES[example_id]
            st.image(example["thumbnail"], use_container_width=True)
            st.write(f"**{example['name'][language]}**")
            st.write(example["description"][language])
            if st.button(t("analyze_button"), key=f"analyze_{example_id}"):
                with st.spinner(t("processing")):
                    # Check if the example file exists
                    if os.path.exists(example["path"]):
                        # Process the video
                        result, confidence, stats = enhanced_process_video(
                            example["path"], 
                            detector,
                            frames_to_sample=20,
                            sampling_strategy='uniform'
                        )
                        
                        # Display result
                        if result == "Real":
                            st.success(t("result_real"))
                        else:
                            st.error(t("result_fake"))
                        
                        st.write(f"**{t('confidence')}:** {confidence:.2f}%")
                        st.write(f"**{t('processing_time')}:** {stats['processing_time']:.2f} {t('seconds')}")
                    else:
                        st.error(f"Example file not found: {example['path']}")
    
    # About tab
    with about_tab:
        st.subheader(t("about_title"))
        st.write(t("about_content"))
        
        st.subheader(t("how_it_works_title"))
        st.write(t("how_it_works_content"))
        
        # Add diagram of the detection process
        st.image("./static/examples/detection_process.png", use_container_width=True)
        
        # Disclaimer
        st.warning(t("disclaimer"))
    
    # Footer
    st.markdown("---")
    st.markdown(f"<div style='text-align: center'>{t('disclaimer')}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

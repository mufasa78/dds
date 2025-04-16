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
from utils.enhanced_video_processing import enhanced_process_video
from utils.language import get_translation, LANGUAGES
from data.examples import EXAMPLES

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inline CSS for Streamlit Cloud
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333;
    line-height: 1.6;
}

h1, h2, h3 {
    color: #2c3e50;
    font-weight: 600;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.real-result {
    background-color: rgba(40, 167, 69, 0.05);
    border-left: 5px solid #28a745;
    padding: 1rem;
    border-radius: 8px;
}

.fake-result {
    background-color: rgba(220, 53, 69, 0.05);
    border-left: 5px solid #dc3545;
    padding: 1rem;
    border-radius: 8px;
}

.diagram-box {
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    transition: transform 0.3s ease;
}

.diagram-box:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

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
            # Fallback to simple model
            st.info("Loading simple detection model...")
            detector = SimpleDeepfakeDetector()
            st.success("✅ Simple model loaded successfully")
            return detector
        except Exception as e:
            st.error(f"Failed to load any model: {e}")
            return None

@st.cache_resource
def load_face_detector():
    """Load the face detector model"""
    try:
        st.info("Loading face detector...")
        face_detector = HaarCascadeFaceDetector()
        st.success("✅ Face detector loaded successfully")
        return face_detector
    except Exception as e:
        st.error(f"Failed to load face detector: {e}")
        return None

def main():
    """Main function for the Streamlit app"""
    # Load detector model
    detector = load_detector()
    
    # Language selection
    language = st.sidebar.selectbox(
        "Language / 语言",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]["native"]
    )
    
    # Translation function
    def t(key):
        return get_translation(key, language)
    
    # Main header with custom styling
    st.markdown(f"<h1 style='text-align: center;'><i class='fas fa-shield-alt' style='margin-right: 10px;'></i>{t('app_title')}</h1>", unsafe_allow_html=True)
    
    # Subtitle with custom styling
    st.markdown(f"<p style='text-align: center; font-size: 1.2rem; color: #6c757d; margin-bottom: 2rem;'>{t('app_subtitle')}</p>", unsafe_allow_html=True)
    
    # Description
    st.write(t("app_description"))
    
    # Create tabs
    upload_tab, examples_tab, about_tab = st.tabs([
        t("upload_tab"), 
        t("examples_tab"), 
        t("about_tab")
    ])
    
    # Upload tab
    with upload_tab:
        st.subheader(t("upload_title"))
        
        # File uploader
        uploaded_files = st.file_uploader(
            t("upload_prompt"),
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True
        )
        
        # Analysis options
        with st.expander(t("advanced_options"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                frames_to_sample = st.slider(
                    t("frames_to_sample"),
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5
                )
            with col2:
                sampling_strategy = st.selectbox(
                    t("sampling_strategy"),
                    options=["uniform", "random", "first_frames"],
                    format_func=lambda x: t(f"strategy_{x}")
                )
        
        # Process button
        if uploaded_files:
            if st.button(t("analyze_button"), key="analyze_upload"):
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each video
                results = {}
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"{t('processing')} {uploaded_file.name}...")
                    
                    # Save uploaded file to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        video_path = tmp_file.name
                    
                    try:
                        # Process the video
                        result, confidence, stats = enhanced_process_video(
                            video_path, 
                            detector,
                            frames_to_sample=frames_to_sample,
                            sampling_strategy=sampling_strategy
                        )
                        
                        # Store results
                        results[uploaded_file.name] = {
                            'prediction': result,
                            'confidence': confidence,
                            'stats': stats
                        }
                    except Exception as e:
                        results[uploaded_file.name] = {
                            'prediction': t("error_processing"),
                            'error': str(e)
                        }
                    
                    # Clean up temp file
                    try:
                        os.unlink(video_path)
                    except:
                        pass
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                # Display results with enhanced styling
                status_text.text(t("analysis_complete"))
                
                st.markdown(f"<h2 style='text-align: center; margin-top: 2rem;'>{t('results_title')}</h2>", unsafe_allow_html=True)
                
                for filename, result in results.items():
                    with st.expander(f"**{filename}**", expanded=True):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            # Create a styled result card
                            if result['prediction'] == t("error_processing"):
                                st.error(t("error_processing"))
                            elif result['prediction'] == "Real":
                                st.markdown(f"""
                                <div class='real-result'>
                                    <h3><i class='fas fa-check-circle' style='color: #28a745; margin-right: 10px;'></i>{t('result_real')}</h3>
                                    <p style='font-size: 1.2rem;'><strong>{t('confidence')}:</strong> {result['confidence']:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='fake-result'>
                                    <h3><i class='fas fa-exclamation-triangle' style='color: #dc3545; margin-right: 10px;'></i>{t('result_fake')}</h3>
                                    <p style='font-size: 1.2rem;'><strong>{t('confidence')}:</strong> {result['confidence']:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display stats in a more visually appealing way
                            if 'stats' in result and isinstance(result['stats'], dict) and result['prediction'] != t("error_processing"):
                                stats = result['stats']
                                st.markdown("<h4 style='margin-top: 1.5rem;'>Analysis Details</h4>", unsafe_allow_html=True)
                                
                                # Create a metrics display
                                metric_cols = st.columns(3)
                                
                                if 'processing_time' in stats:
                                    with metric_cols[0]:
                                        st.metric(label=t('processing_time'), value=f"{stats['processing_time']:.2f}s")
                                        
                                if 'frames_sampled' in stats:
                                    with metric_cols[1]:
                                        st.metric(label=t('frames_analyzed'), value=stats['frames_sampled'])
                                        
                                if 'sampling_strategy' in stats:
                                    with metric_cols[2]:
                                        st.metric(label="Sampling Strategy", value=stats['sampling_strategy'].replace('_', ' ').title())
    
    # Examples tab with enhanced styling
    with examples_tab:
        st.markdown(f"<h2 style='text-align: center;'>{t('examples_title')}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; margin-bottom: 2rem;'>{t('examples_description')}</p>", unsafe_allow_html=True)
        
        # Display examples as a grid of cards with enhanced styling
        col1, col2, col3 = st.columns(3)
        
        # Function to display example card with consistent styling
        def display_example_card(column, example_id):
            example = EXAMPLES[example_id]
            with column:
                # Create a card-like container with CSS
                st.markdown("""
                <div style="border-radius: 10px; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1); overflow: hidden; margin-bottom: 20px; transition: transform 0.3s ease;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
                </div>
                """, unsafe_allow_html=True)
                
                # Display thumbnail with hover effect
                st.image(example["thumbnail"], use_container_width=True)
                
                # Example title and description
                st.markdown(f"<h3 style='margin-top: 0.5rem; font-size: 1.3rem;'>{example['name'][language]}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #6c757d;'>{example['description'][language]}</p>", unsafe_allow_html=True)
                
                # Analyze button with custom styling
                analyze_btn = st.button(t("analyze_button"), key=f"analyze_{example_id}", use_container_width=True)
                
                if analyze_btn:
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
                            
                            # Display result with enhanced styling
                            if result == "Real":
                                st.markdown(f"""
                                <div class='real-result' style='margin-top: 1rem;'>
                                    <h4><i class='fas fa-check-circle' style='color: #28a745; margin-right: 10px;'></i>{t('result_real')}</h4>
                                    <p><strong>{t('confidence')}:</strong> {confidence:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='fake-result' style='margin-top: 1rem;'>
                                    <h4><i class='fas fa-exclamation-triangle' style='color: #dc3545; margin-right: 10px;'></i>{t('result_fake')}</h4>
                                    <p><strong>{t('confidence')}:</strong> {confidence:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display metrics in a cleaner way
                            metric_cols = st.columns(2)
                            with metric_cols[0]:
                                st.metric(label=t('processing_time'), value=f"{stats['processing_time']:.2f}s")
                            with metric_cols[1]:
                                st.metric(label=t('frames_analyzed'), value=stats['frames_sampled'])
                        else:
                            st.error(f"Example file not found: {example['path']}")
        
        # Display all example cards using the function
        display_example_card(col1, "example1")
        display_example_card(col2, "example2")
        display_example_card(col3, "example3")
    
    # About tab with enhanced styling
    with about_tab:
        # Main about section with card styling
        st.markdown(f"<h2 style='text-align: center;'>{t('about_title')}</h2>", unsafe_allow_html=True)
        
        # About content in a card
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
        """, unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; line-height: 1.6;'>{t('about_content')}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # How it works section
        st.markdown(f"<h3 style='margin-top: 2rem;'>{t('how_it_works_title')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; margin-bottom: 2rem;'>{t('how_it_works_content')}</p>", unsafe_allow_html=True)
        
        # Enhanced diagram of the detection process with icons
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("""
            <div class="diagram-box" style="background-color:#f0f8ff; border:2px solid #4682b4; border-radius:10px; text-align:center; height:100px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <strong>Video Input</strong>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown("""
            <div class="diagram-box" style="background-color:#f0fff0; border:2px solid #228b22; border-radius:10px; text-align:center; height:100px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <strong>Frame Extraction</strong>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[2]:
            st.markdown("""
            <div class="diagram-box" style="background-color:#fff0f5; border:2px solid #cd5c5c; border-radius:10px; text-align:center; height:100px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <strong>Face Detection</strong>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[3]:
            st.markdown("""
            <div class="diagram-box" style="background-color:#e6e6fa; border:2px solid #483d8b; border-radius:10px; text-align:center; height:100px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <strong>Deepfake<br>Classification</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Model information section
        st.markdown(f"<h3 style='margin-top: 2rem;'>{t('model_info_title')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem;'>{t('model_info')}</p>", unsafe_allow_html=True)
        
        # Datasets section
        st.markdown(f"<h3 style='margin-top: 2rem;'>{t('datasets_title')}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem;'>{t('datasets_used')}</p>", unsafe_allow_html=True)
        
        # Disclaimer with enhanced styling
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; border-radius: 5px; margin-top: 30px;">
            <h4 style="color: #856404; margin-top: 0;">Disclaimer</h4>
        """, unsafe_allow_html=True)
        st.markdown(f"<p style='color: #856404;'>{t('disclaimer')}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(f"""
    <footer style='text-align: center; padding: 20px 0; color: #6c757d;'>
        <div>
            Deepfake Detection System - Powered by AI
        </div>
        <div style='margin-top: 10px; font-size: 0.9rem;'>
            All rights reserved &copy; {time.strftime('%Y')}
        </div>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

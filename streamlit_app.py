import os
import time
import tempfile
import logging
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import threading

from models.detector import DeepfakeDetector
from models.haar_face_detector import HaarCascadeFaceDetector
from models.simple_model import SimpleDeepfakeDetector
from utils.enhanced_video_processing import enhanced_process_video
from utils.video_preprocessing import VideoPreprocessor
from utils.batch_processing import BatchProcessor
from utils.result_cache import ResultCache
from utils.logging_system import LoggingSystem
from utils.performance_monitor import PerformanceMonitor, time_function
from utils.feedback_system import FeedbackSystem
from utils.model_versioning import ModelVersionManager
from utils.language import get_translation, LANGUAGES
from data.examples import EXAMPLES

# Set page configuration
st.set_page_config(
    page_title="伪造视频识别系统 - Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"  # Changed to expanded to make language selector visible
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

/* Floating language selector button */
.language-float-button {
    position: fixed;
    top: 70px;
    right: 20px;
    z-index: 9999;
    background-color: #007BFF;
    color: white;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    cursor: pointer;
    transition: all 0.3s ease;
}

.language-float-button:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}

/* Make sidebar more prominent */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e9ecef;
}

/* Highlight language selector */
[data-testid="stSelectbox"] {
    border: 2px solid #e9ecef;
    border-radius: 8px;
    transition: all 0.3s ease;
}

[data-testid="stSelectbox"]:hover {
    border-color: #007BFF;
    box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}
</style>

<!-- Floating language button -->
<div class="language-float-button" onclick="document.querySelector('[data-testid=\'stSidebar\']').classList.toggle('collapsed')">
    🌐
</div>
""", unsafe_allow_html=True)

# Initialize global systems
logging_system = LoggingSystem()
performance_monitor = PerformanceMonitor()
feedback_system = FeedbackSystem()
result_cache = ResultCache()
model_version_manager = ModelVersionManager()
video_preprocessor = VideoPreprocessor()

@st.cache_resource
@time_function("load_detector")
def load_detector():
    """Load the deepfake detector model"""
    logger = logging_system.app_logger
    logger.info("Loading deepfake detector model")

    try:
        # Check if we have a current model version
        current_version = model_version_manager.current_version
        if current_version:
            model_path = model_version_manager.get_model_path()
            if model_path:
                logger.info(f"Loading model version {current_version} from {model_path}")
                st.info(f"Loading model version {current_version}...")

                # Auto-detect device (will use GPU if available)
                detector = DeepfakeDetector(device=None, model_path=model_path, model_version=current_version)

                st.success(f"✅ Model version {current_version} loaded successfully on {detector.device}")
                logger.info(f"Model loaded successfully on {detector.device}")
                return detector

        # If no version is set or model path not found, load default model
        logger.info("Loading default model")
        st.info("Loading default deepfake detection model...")

        # Auto-detect device (will use GPU if available)
        detector = DeepfakeDetector(device=None)

        device_info = "GPU" if detector.device == "cuda" else "CPU"
        st.success(f"✅ Model loaded successfully on {device_info}")
        logger.info(f"Default model loaded successfully on {device_info}")
        return detector

    except Exception as e:
        logger.error(f"Failed to load EfficientNet model: {e}")
        st.warning(f"Failed to load EfficientNet model: {e}")

        try:
            # Fallback to simple model
            logger.info("Attempting to load simple model")
            st.info("Loading simple detection model...")

            model = SimpleDeepfakeDetector()
            detector = DeepfakeDetector(device=None, custom_model=model)

            device_info = "GPU" if detector.device == "cuda" else "CPU"
            st.success(f"✅ Simple model loaded successfully on {device_info}")
            logger.info(f"Simple model loaded successfully on {device_info}")
            return detector

        except Exception as e2:
            logger.error(f"Failed to load any model: {e2}")
            st.error(f"Failed to load any model: {e2}")
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

def ensure_translations(key, default_text):
    """Ensure a translation key exists, if not, return the default text"""
    from data.translations import TRANSLATIONS

    # Check if the key exists in translations
    if key not in TRANSLATIONS:
        # Add warning in the console but don't show to user
        print(f"Warning: Translation key '{key}' not found, using default text")
        return default_text

    # Check if Chinese translation exists
    if 'zh' not in TRANSLATIONS[key]:
        print(f"Warning: No Chinese translation for key '{key}', using default text")
        return default_text

    # Return the key for normal translation flow
    return key

def main():
    """Main function for the Streamlit app"""
    # Load detector model
    detector = load_detector()

    # Create a very prominent language selector at the top
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 10px; margin-bottom: 20px;
              border: 2px solid #007BFF; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="text-align: center; margin-bottom: 10px; color: #007BFF;">🌐 语言选择 / Language Selection / Selección de idioma</h3>
    </div>
    """, unsafe_allow_html=True)

    # Create a row of language buttons for quick selection with Chinese highlighted by default
    lang_cols = st.columns(3)

    with lang_cols[0]:
        en_button = st.button("English 🇺🇸", use_container_width=True, help="Switch to English")
    with lang_cols[1]:
        # Use a different style for Chinese to highlight it as the default
        zh_button = st.button(
            "中文 🇨🇳",
            use_container_width=True,
            help="切换到中文",
            type="primary"  # Highlight Chinese as the primary/default option
        )
    with lang_cols[2]:
        es_button = st.button("Español 🇪🇸", use_container_width=True, help="Cambiar a Español")

    # Add language selector to sidebar with prominent styling
    with st.sidebar:
        st.markdown("### 🌐 语言设置 / Language Settings")
        st.markdown("""
        <div style="padding: 15px; background-color: #e6f3ff; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #007BFF;">
            <p style="font-weight: bold; margin-bottom: 5px; color: #0056b3;">选择您的首选语言:</p>
            <p style="font-size: 0.9rem; color: #555;">Select your preferred language / Seleccione su idioma preferido</p>
        </div>
        """, unsafe_allow_html=True)

        # Language selection in sidebar with Chinese as default
        sidebar_lang = st.selectbox(
            "语言 / Language / Idioma",
            options=list(LANGUAGES.keys()),
            index=1,  # Index 1 is 'zh' (Chinese)
            format_func=lambda x: LANGUAGES[x]["native_name"],
            help="选择您的首选语言 / Select your preferred language / Seleccione su idioma preferido"
        )

        # Add some space and a divider
        st.markdown("---")

    # Set language based on button clicks or sidebar selection
    # Default to Chinese if no selection is made
    language = sidebar_lang if sidebar_lang else "zh"

    # Store the language selection in session state to persist between reruns
    if 'language' not in st.session_state:
        st.session_state['language'] = "zh"  # Default to Chinese

    # Update language based on button clicks
    if en_button:
        language = "en"
        st.session_state['language'] = "en"
    elif zh_button:
        language = "zh"
        st.session_state['language'] = "zh"
    elif es_button:
        language = "es"
        st.session_state['language'] = "es"
    else:
        # Use the stored language if no button was clicked
        language = st.session_state['language']

    # Create columns for the main content
    col1, col2 = st.columns([3, 1])

    # Translation function
    def t(key):
        return get_translation(key, language)

    # Add a language selector container with styling at the top
    st.markdown("""
    <div style="position: absolute; top: 0.5rem; right: 1rem; z-index: 1000; background-color: rgba(255,255,255,0.7);
             padding: 0.5rem; border-radius: 0.5rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
        <div id="language-container"></div>
    </div>
    """, unsafe_allow_html=True)

    # Main header with custom styling
    with col1:
        st.markdown(f"<h1 style='text-align: center;'><i class='fas fa-shield-alt' style='margin-right: 10px;'></i>{t('app_title')}</h1>", unsafe_allow_html=True)

        # Subtitle with custom styling
        st.markdown(f"<p style='text-align: center; font-size: 1.2rem; color: #6c757d; margin-bottom: 2rem;'>{t('app_subtitle')}</p>", unsafe_allow_html=True)

    # Description
    st.write(t("app_description"))

    # Create tabs
    upload_tab, examples_tab, about_tab, performance_tab = st.tabs([
        t("upload_tab"),
        t("examples_tab"),
        t("about_tab"),
        t("performance_tab")
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
            # Create tabs for different option categories
            sampling_tab, preprocessing_tab, performance_tab = st.tabs([
                t("sampling_options"),
                t("preprocessing_options"),
                t("performance_options")
            ])

            # Sampling options
            with sampling_tab:
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

            # Preprocessing options
            with preprocessing_tab:
                st.write(t("preprocessing_description"))

                # Preprocessing methods
                preprocessing_methods = st.multiselect(
                    t("select_preprocessing"),
                    options=["denoise", "enhance_contrast", "sharpen", "normalize", "grayscale"],
                    format_func=lambda x: t(f"preprocess_{x}")
                )

                # Store in session state
                st.session_state['preprocessing_methods'] = preprocessing_methods

                # Method-specific parameters
                if preprocessing_methods:
                    st.subheader(t("preprocessing_params"))

                    # Initialize params dict
                    preprocessing_params = {}

                    # Denoise parameters
                    if "denoise" in preprocessing_methods:
                        st.write(t("denoise_params"))
                        denoise_strength = st.slider(
                            t("denoise_strength"),
                            min_value=1,
                            max_value=30,
                            value=10
                        )
                        preprocessing_params["denoise"] = {"strength": denoise_strength}

                    # Contrast enhancement parameters
                    if "enhance_contrast" in preprocessing_methods:
                        st.write(t("contrast_params"))
                        contrast_limit = st.slider(
                            t("contrast_limit"),
                            min_value=1.0,
                            max_value=5.0,
                            value=2.0,
                            step=0.1
                        )
                        preprocessing_params["enhance_contrast"] = {"clip_limit": contrast_limit}

                    # Sharpen parameters
                    if "sharpen" in preprocessing_methods:
                        st.write(t("sharpen_params"))
                        sharpen_amount = st.slider(
                            t("sharpen_amount"),
                            min_value=0.1,
                            max_value=3.0,
                            value=1.0,
                            step=0.1
                        )
                        preprocessing_params["sharpen"] = {"amount": sharpen_amount}

                    # Store params in session state
                    st.session_state['preprocessing_params'] = preprocessing_params

            # Performance options
            with performance_tab:
                st.write(t("performance_description"))

                # Enable GPU acceleration
                use_gpu = st.checkbox(
                    t("use_gpu"),
                    value=torch.cuda.is_available(),
                    disabled=not torch.cuda.is_available()
                )

                if not torch.cuda.is_available() and use_gpu:
                    st.warning(t("gpu_not_available"))

                # Enable result caching
                use_cache = st.checkbox(
                    t("use_cache"),
                    value=True
                )

                # Cache settings if enabled
                if use_cache:
                    cache_expiration = st.slider(
                        t("cache_expiration"),
                        min_value=1,
                        max_value=30,
                        value=7,
                        help=t("cache_expiration_help")
                    )

                    # Clear cache button
                    if st.button(t("clear_cache")):
                        result_cache.clear()
                        st.success(t("cache_cleared"))

                # Store in session state
                st.session_state['use_gpu'] = use_gpu
                st.session_state['use_cache'] = use_cache
                if use_cache:
                    st.session_state['cache_expiration'] = cache_expiration

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
                        # Check if result is in cache
                        cache_key = f"{video_path}_{frames_to_sample}_{sampling_strategy}"
                        cached_result = result_cache.get(video_path, {
                            'frames_to_sample': frames_to_sample,
                            'sampling_strategy': sampling_strategy
                        })

                        if cached_result:
                            logging_system.log_app(logging.INFO, f"Using cached result for {uploaded_file.name}")
                            result = cached_result['prediction']
                            confidence = cached_result['confidence']
                            stats = cached_result['stats']
                            stats['from_cache'] = True
                        else:
                            # Apply preprocessing if selected
                            preprocessing_methods = st.session_state.get('preprocessing_methods', [])
                            if preprocessing_methods:
                                logging_system.log_app(logging.INFO, f"Applying preprocessing to {uploaded_file.name}: {preprocessing_methods}")
                                with st.spinner(f"Preprocessing video with {', '.join(preprocessing_methods)}..."):
                                    video_path = video_preprocessor.preprocess_video(
                                        video_path,
                                        methods=preprocessing_methods,
                                        params=st.session_state.get('preprocessing_params', {})
                                    )

                            # Process the video with performance monitoring
                            with performance_monitor.time_operation("process_video"):
                                result, confidence, stats = enhanced_process_video(
                                    video_path,
                                    detector,
                                    frames_to_sample=frames_to_sample,
                                    sampling_strategy=sampling_strategy
                                )

                            # Cache the result
                            result_cache.set(video_path, {
                                'prediction': result,
                                'confidence': confidence,
                                'stats': stats
                            }, {
                                'frames_to_sample': frames_to_sample,
                                'sampling_strategy': sampling_strategy
                            })

                        # Store results
                        results[uploaded_file.name] = {
                            'prediction': result,
                            'confidence': confidence,
                            'stats': stats
                        }

                        # Log the result
                        logging_system.log_app(
                            logging.INFO,
                            f"Processed {uploaded_file.name}: {result} ({confidence:.2f}%)",
                            video_name=uploaded_file.name,
                            result=result,
                            confidence=confidence
                        )

                    except Exception as e:
                        error_msg = str(e)
                        results[uploaded_file.name] = {
                            'prediction': t("error_processing"),
                            'error': error_msg
                        }

                        # Log the error
                        logging_system.log_error(
                            e,
                            {
                                "video_name": uploaded_file.name,
                                "operation": "process_video"
                            }
                        )

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

                # Log completion
                logging_system.log_app(
                    logging.INFO,
                    f"Analysis complete for {len(uploaded_files)} files",
                    file_count=len(uploaded_files)
                )

                # Update model performance metrics
                if detector:
                    performance_monitor.record_model_performance(
                        detector.model_info.get("model_type", "unknown"),
                        detector.get_performance_metrics()
                    )

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

                                # Show if result was from cache
                                if 'from_cache' in stats and stats['from_cache']:
                                    st.info(t('result_from_cache'))

                            # Add feedback section
                            st.markdown("<h4 style='margin-top: 1.5rem;'>Feedback</h4>", unsafe_allow_html=True)
                            st.write(t('feedback_prompt'))

                            # Create columns for rating and feedback
                            feedback_cols = st.columns([1, 3])

                            with feedback_cols[0]:
                                # Rating selection
                                rating = st.select_slider(
                                    t('rating_label'),
                                    options=[1, 2, 3, 4, 5],
                                    value=3,
                                    format_func=lambda x: "⭐" * x
                                )

                            with feedback_cols[1]:
                                # Feedback comment
                                comment = st.text_area(t('feedback_comment'), key=f"feedback_{filename}")

                            # Submit feedback button
                            if st.button(t('submit_feedback'), key=f"submit_{filename}"):
                                # Create metadata about the result
                                metadata = {
                                    "filename": filename,
                                    "prediction": result['prediction'],
                                    "confidence": float(result['confidence']) if 'confidence' in result else 0.0,
                                    "processing_params": {
                                        "frames_to_sample": frames_to_sample,
                                        "sampling_strategy": sampling_strategy,
                                        "preprocessing": st.session_state.get('preprocessing_methods', [])
                                    }
                                }

                                # Submit feedback
                                feedback_id = feedback_system.add_feedback(
                                    "detection_result",
                                    rating,
                                    comment,
                                    metadata
                                )

                                if feedback_id:
                                    st.success(t('feedback_submitted'))

                                    # Log feedback submission
                                    logging_system.log_user_action(
                                        "submit_feedback",
                                        details={
                                            "feedback_id": feedback_id,
                                            "rating": rating,
                                            "filename": filename
                                        }
                                    )
                                else:
                                    st.error(t('feedback_error'))

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

    # Performance tab
    with performance_tab:
        st.markdown(f"<h2 style='text-align: center;'>{t('performance_dashboard_title')}</h2>", unsafe_allow_html=True)

        # System performance overview
        st.subheader(t('system_performance'))

        # Get performance summary
        perf_summary = performance_monitor.get_performance_summary()

        # Display system metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(t('cpu_usage'), f"{perf_summary['system']['cpu_usage_avg']:.1f}%")

        with col2:
            st.metric(t('memory_usage'), f"{perf_summary['system']['memory_usage_avg']:.1f}%")

        with col3:
            st.metric(t('disk_usage'), f"{perf_summary['system']['disk_usage_avg']:.1f}%")

        # Model performance
        st.subheader(t('model_performance'))

        if detector:
            # Get model info
            model_info = detector.get_model_info()
            model_perf = detector.get_performance_metrics()

            # Display model info
            st.write(f"**{t('model_type')}:** {model_info.get('model_type', 'Unknown')}")
            st.write(f"**{t('model_version')}:** {model_info.get('version', 'Unknown')}")
            st.write(f"**{t('device')}:** {model_info.get('device', 'CPU')}")

            # Display performance metrics
            if model_perf and 'total_processed' in model_perf and model_perf['total_processed'] > 0:
                st.write(f"**{t('total_processed')}:** {model_perf['total_processed']}")
                st.write(f"**{t('avg_processing_time')}:** {model_perf['avg_time']:.4f}s")
                st.write(f"**{t('fake_detections')}:** {model_perf['fake_count']}")
                st.write(f"**{t('real_detections')}:** {model_perf['real_count']}")

                # Create a pie chart for fake vs real
                data = {
                    'Category': [t('fake'), t('real')],
                    'Count': [model_perf['fake_count'], model_perf['real_count']]
                }

                st.bar_chart(data, x='Category', y='Count')
            else:
                st.info(t('no_model_metrics'))
        else:
            st.warning(t('model_not_loaded'))

        # Cache statistics
        st.subheader(t('cache_statistics'))
        cache_stats = result_cache.get_stats()

        st.write(f"**{t('total_entries')}:** {cache_stats['total_entries']}")
        st.write(f"**{t('cache_size')}:** {cache_stats['size_mb']:.2f} MB")
        st.write(f"**{t('max_cache_size')}:** {cache_stats['max_size_mb']} MB")

        # Clear cache button
        if st.button(t('clear_cache'), key="clear_cache_perf"):
            result_cache.clear()
            st.success(t('cache_cleared'))

        # Feedback statistics
        st.subheader(t('feedback_statistics'))
        feedback_stats = feedback_system.get_stats()

        if 'overall' in feedback_stats and feedback_stats['overall']['count'] > 0:
            st.write(f"**{t('total_feedback')}:** {feedback_stats['overall']['count']}")
            st.write(f"**{t('avg_rating')}:** {feedback_stats['overall']['avg_rating']:.1f}/5.0")

            # Display rating distribution
            ratings = feedback_stats['overall']['ratings']
            rating_data = {
                'Rating': list(map(str, ratings.keys())),
                'Count': list(ratings.values())
            }

            st.bar_chart(rating_data, x='Rating', y='Count')
        else:
            st.info(t('no_feedback'))

        # Link to full dashboard
        st.markdown("---")
        st.markdown(f"<p style='text-align: center;'><a href='/performance' target='_self'>{t('view_full_dashboard')}</a></p>", unsafe_allow_html=True)

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

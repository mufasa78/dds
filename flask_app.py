import os
import tempfile
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import cv2
import numpy as np
from werkzeug.utils import secure_filename

from models.detector import DeepfakeDetector
from models.haar_face_detector import HaarCascadeFaceDetector
from models.simple_model import SimpleDeepfakeDetector
from utils.video_processing import process_video
from utils.enhanced_video_processing import enhanced_process_video
from utils.language import get_translation, LANGUAGES
from utils.example_generator import generate_placeholder_examples
from data.examples import EXAMPLES

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
detector = None
face_detector = None
load_lock = threading.Lock()
loading_messages = {
    'en': "Loading detector model... This may take a few minutes.",
    'zh': "正在加载检测模型...这可能需要几分钟时间。"
}
model_loaded = False

# Generate example videos when the app starts
threading.Thread(target=generate_placeholder_examples, daemon=True).start()

def load_detector_in_background():
    """Load the deepfake detector model in background thread"""
    global detector, face_detector, model_loaded
    with load_lock:
        if detector is None:
            try:
                # First try to load the standard detector
                detector = DeepfakeDetector(device='cpu')
            except Exception as e:
                print(f"Failed to load EfficientNet model: {e}")
                try:
                    # Fallback to simplified model
                    model = SimpleDeepfakeDetector(num_classes=1)
                    detector = DeepfakeDetector(device='cpu', custom_model=model)
                except Exception as e2:
                    print(f"Failed to load simplified model: {e2}")
                    # Use fallback detector as last resort
                    detector = DeepfakeDetector(device='cpu', use_fallback=True)
            
            # Load face detector
            try:
                face_detector = HaarCascadeFaceDetector()
            except Exception as e:
                print(f"Failed to load Haar cascade face detector: {e}")
                # Use simple face detector as fallback
                from models.face_detector import FaceDetector
                face_detector = FaceDetector()
            
            model_loaded = True

# Start loading the model in background
threading.Thread(target=load_detector_in_background, daemon=True).start()

def save_uploaded_file(file):
    """Save an uploaded file and return the path"""
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    return temp_path

@app.route('/')
def index():
    # Get language from session or set default
    language = session.get('language', 'en')
    t = get_translation(language)
    
    # Get appropriate loading message based on language
    loading_message = loading_messages.get(language, loading_messages['en'])
    
    # Check if model is loaded
    return render_template('index.html', 
                          languages=LANGUAGES, 
                          current_language=language,
                          model_loaded=model_loaded,
                          loading_message=loading_message,
                          t=t)

@app.route('/set_language/<language>')
def set_language(language):
    session['language'] = language
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    global detector
    
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'error': 'Model is still loading, please wait and try again.'}), 400
    
    # Get language for translations
    language = session.get('language', 'en')
    t = get_translation(language)
    
    # Check if files were uploaded
    if 'videos' not in request.files:
        return jsonify({'error': t('no_file_selected')}), 400
    
    files = request.files.getlist('videos')
    
    # Check if any file was selected
    if not files or files[0].filename == '':
        return jsonify({'error': t('no_file_selected')}), 400
    
    results = {}
    
    # Process each uploaded file
    for file in files:
        try:
            # Save uploaded file
            temp_path = save_uploaded_file(file)
            
            # Process the video with enhanced processing
            result, confidence, stats = enhanced_process_video(
                temp_path, 
                detector, 
                frames_to_sample=20,
                sampling_strategy='scene_change'
            )
            
            # Store result with stats
            results[file.filename] = {
                'prediction': result,
                'confidence': confidence,
                'stats': {
                    'processing_time': stats.get('processing_time', 0),
                    'frames_sampled': stats.get('frames_sampled', 0),
                    'sampling_strategy': stats.get('sampling_strategy', 'unknown')
                }
            }
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            results[file.filename] = {
                'prediction': f"Error: {str(e)}",
                'confidence': 0.0,
                'stats': {'error': str(e)}
            }
    
    return jsonify({'results': results})

@app.route('/api/model_status')
def model_status():
    return jsonify({'loaded': model_loaded})

@app.route('/api/example')
def get_example():
    """Get example video information"""
    example_id = request.args.get('id')
    
    # Check if the example exists
    if example_id not in EXAMPLES:
        return jsonify({'status': 'error', 'error': 'Example not found'}), 404
    
    # Get language from session
    language = session.get('language', 'en')
    
    # Get example data
    example = EXAMPLES[example_id]
    
    # Return example data with proper translation
    return jsonify({
        'status': 'success',
        'id': example_id,
        'name': example['name'].get(language, example['name']['en']),
        'description': example['description'].get(language, example['description']['en']),
        'path': example['path'],
        'thumbnail': example['thumbnail'],
        'type': example['type']
    })

@app.route('/api/analyze_example', methods=['POST'])
def analyze_example():
    """Analyze an example video"""
    global detector
    
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'status': 'error', 'error': 'Model is still loading, please wait and try again.'}), 400
    
    # Get example ID from request
    example_id = request.form.get('id')
    
    # Check if the example exists
    if example_id not in EXAMPLES:
        return jsonify({'status': 'error', 'error': 'Example not found'}), 404
    
    # Get example data
    example = EXAMPLES[example_id]
    
    # Check if the example file exists
    if not os.path.exists(example['path']):
        return jsonify({'status': 'error', 'error': 'Example file not found'}), 404
    
    try:
        # Process the video
        result, confidence, stats = enhanced_process_video(
            example['path'], 
            detector,
            frames_to_sample=20,
            sampling_strategy='uniform'
        )
        
        # Return results
        return jsonify({
            'status': 'success',
            'results': {
                'prediction': result,
                'confidence': confidence,
                'stats': {
                    'processing_time': stats.get('processing_time', 0),
                    'frames_sampled': stats.get('frames_sampled', 0),
                    'sampling_strategy': stats.get('sampling_strategy', 'uniform')
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
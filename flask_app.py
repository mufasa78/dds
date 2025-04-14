import os
import tempfile
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import cv2
import numpy as np
from werkzeug.utils import secure_filename

from models.detector import DeepfakeDetector
from utils.video_processing import process_video
from utils.language import get_translation, LANGUAGES

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
detector = None
load_lock = threading.Lock()
loading_messages = {
    'en': "Loading detector model... This may take a few minutes.",
    'zh': "正在加载检测模型...这可能需要几分钟时间。"
}
model_loaded = False

def load_detector_in_background():
    """Load the deepfake detector model in background thread"""
    global detector, model_loaded
    with load_lock:
        if detector is None:
            detector = DeepfakeDetector(device='cpu')
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
            
            # Process the video
            prediction, confidence = process_video(temp_path, detector)
            
            # Store result
            results[file.filename] = {
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            results[file.filename] = {
                'prediction': f"Error: {str(e)}",
                'confidence': 0.0
            }
    
    return jsonify({'results': results})

@app.route('/api/model_status')
def model_status():
    return jsonify({'loaded': model_loaded})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
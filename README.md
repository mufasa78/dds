# Multilingual Deepfake Detection System

A comprehensive system for detecting deepfake videos using advanced AI techniques. The application is available in both Flask and Streamlit interfaces with multilingual support.

## Features

- **Deepfake Detection**: Analyze videos to determine if they are real or AI-generated fakes
- **Multilingual Support**: Available in English and Chinese
- **Dual Interfaces**: Choose between Flask web application or Streamlit interface
- **Enhanced Video Processing**: Intelligent frame sampling for efficient analysis
- **Face Detection**: Automatically detects and analyzes faces in videos
- **Frame Analysis**: Detailed analysis of individual frames with detection points and visual indicators
- **Visual Indicators**: Color-coded borders and markers to highlight potential manipulations
- **Example Videos**: Pre-loaded examples to demonstrate the system's capabilities
- **Responsive Design**: Works on desktop and mobile devices
- **Batch Processing**: Analyze multiple videos at once

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Windows, macOS, or Linux operating system

### Quick Start

1. Run the startup script:
   ```
   start_apps.bat
   ```

   This script will:
   - Create a virtual environment if it doesn't exist
   - Install all required dependencies
   - Provide options to run Flask, Streamlit, or both applications

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:
   ```
   python -m venv venv_new
   ```

2. Activate the virtual environment:
   - Windows: `venv_new\Scripts\activate`
   - macOS/Linux: `source venv_new/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the applications:
   - Flask: `python flask_app.py`
   - Streamlit: `streamlit run app.py`

## Usage

### Flask Web Application (Port 5001)

1. Open your browser and navigate to `http://localhost:5001`
2. Select your preferred language
3. Upload a video or choose from the examples
4. Click "Analyze Video" to process
5. View the results showing whether the video is real or fake
6. Examine the detailed frame analysis with detection points

### Streamlit Application (Port 5000)

1. Open your browser and navigate to `http://localhost:5000`
2. Select your preferred language from the sidebar
3. Upload a video or choose from the examples
4. Click "Analyze Video" to process
5. View the detailed analysis results

## System Architecture

The system follows a pipeline architecture:

1. **Video Input**: Accept video files in various formats
2. **Frame Extraction**: Extract key frames using scene detection
3. **Face Detection**: Identify faces in the extracted frames
4. **Deepfake Classification**: Analyze faces to determine authenticity
5. **Frame Analysis**: Add detection points and visual indicators to analyzed frames
6. **Result Visualization**: Display comprehensive results with frame-by-frame analysis

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, Streamlit
- **AI/ML**: PyTorch, OpenCV
- **Data Processing**: NumPy, PIL

## Project Structure

```
project/
├── app.py                  # Streamlit application
├── flask_app.py            # Flask web application
├── requirements.txt        # Python dependencies
├── start_apps.bat          # Startup script
├── .streamlit/             # Streamlit configuration
├── data/                   # Data files and examples
├── models/                 # ML models and detectors
├── static/                 # Static assets for Flask
├── templates/              # HTML templates for Flask
└── utils/                  # Utility functions
```

## Disclaimer

This is a demonstration system and may not detect all types of deepfakes with perfect accuracy. The technology is constantly evolving, and this system represents a snapshot of current detection capabilities.

# Translations for multilingual support

# Dictionary of translations
# Each key is a UI text element, and the value is a dictionary of translations
# keyed by language code
TRANSLATIONS = {
    # Common UI elements
    "app_title": {
        "en": "DeepFake Detection System",
        "zh": "伪造视频识别系统"
    },
    "app_description": {
        "en": "Upload a video to detect if it's a deepfake",
        "zh": "上传视频检测是否为AI伪造"
    },
    "upload_title": {
        "en": "Upload Video",
        "zh": "上传视频"
    },
    "upload_button": {
        "en": "Choose Video",
        "zh": "选择视频"
    },
    "analyze_button": {
        "en": "Analyze Video",
        "zh": "分析视频"
    },
    "examples_title": {
        "en": "Example Videos",
        "zh": "示例视频"
    },
    "results_title": {
        "en": "Analysis Results",
        "zh": "分析结果"
    },
    
    # Language selector
    "language_selector": {
        "en": "Language",
        "zh": "语言"
    },
    "language_english": {
        "en": "English",
        "zh": "英文"
    },
    "language_chinese": {
        "en": "Chinese",
        "zh": "中文"
    },
    
    # Results
    "result_real": {
        "en": "Real Video",
        "zh": "真实视频"
    },
    "result_fake": {
        "en": "Fake Video",
        "zh": "伪造视频"
    },
    "confidence": {
        "en": "Confidence",
        "zh": "置信度"
    },
    "processing_time": {
        "en": "Processing Time",
        "zh": "处理时间"
    },
    "seconds": {
        "en": "seconds",
        "zh": "秒"
    },
    "frames_analyzed": {
        "en": "Frames Analyzed",
        "zh": "分析帧数"
    },
    
    # Progress indicators
    "loading": {
        "en": "Loading...",
        "zh": "加载中..."
    },
    "processing": {
        "en": "Processing...",
        "zh": "处理中..."
    },
    "analyzing_video": {
        "en": "Analyzing video...",
        "zh": "正在分析视频..."
    },
    "extracting_frames": {
        "en": "Extracting frames...",
        "zh": "提取视频帧..."
    },
    "detecting_faces": {
        "en": "Detecting faces...",
        "zh": "检测人脸..."
    },
    "analyzing_frames": {
        "en": "Analyzing frames...",
        "zh": "分析帧内容..."
    },
    
    # Errors
    "error_no_video": {
        "en": "Please select a video to analyze",
        "zh": "请选择要分析的视频"
    },
    "error_invalid_format": {
        "en": "Invalid video format. Please upload MP4, AVI, or MOV files.",
        "zh": "无效的视频格式。请上传MP4、AVI或MOV文件。"
    },
    "error_too_large": {
        "en": "Video file too large. Maximum size is 100MB.",
        "zh": "视频文件太大。最大大小为100MB。"
    },
    "error_processing": {
        "en": "Error processing video. Please try another file.",
        "zh": "处理视频时出错。请尝试其他文件。"
    },
    "error_no_faces": {
        "en": "No faces detected in the video. Please try a video with visible faces.",
        "zh": "视频中未检测到人脸。请尝试使用有可见人脸的视频。"
    },
    
    # Other UI elements
    "about_title": {
        "en": "About",
        "zh": "关于"
    },
    "about_content": {
        "en": "This system uses advanced deep learning models to detect manipulated facial videos (deepfakes).",
        "zh": "该系统使用先进的深度学习模型来检测经过操纵的面部视频（深度伪造）。"
    },
    "how_it_works_title": {
        "en": "How It Works",
        "zh": "工作原理"
    },
    "how_it_works_content": {
        "en": "The system extracts frames from the video, detects faces, and analyzes them for signs of manipulation.",
        "zh": "系统从视频中提取帧，检测人脸，并分析它们是否有被操纵的迹象。"
    },
    "disclaimer": {
        "en": "Disclaimer: This is a demonstration system and may not detect all types of deepfakes with perfect accuracy.",
        "zh": "免责声明：这是一个演示系统，可能无法以完美的准确度检测所有类型的深度伪造。"
    }
}
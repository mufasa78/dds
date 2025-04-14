# Translation dictionary for all supported languages

TRANSLATIONS = {
    "en": {
        # Main UI
        "deepfake_detection_system": "Deepfake Video Detection System",
        "system_description": "This system uses deep learning to detect manipulated videos. Upload one or more videos to analyze them for potential deepfake content.",
        "upload_videos": "Upload Videos",
        "analyze_videos": "Analyze Videos",
        "processing_video": "Processing {file}...",
        "analysis_complete": "Analysis complete!",
        "results": "Results",
        
        # Prediction results
        "real": "Real",
        "fake": "Fake",
        "error": "Error",
        "confidence": "Confidence",
        
        # About section
        "about_model": "About the Model",
        "model_info": "This system uses an EfficientNet B4 model fine-tuned on deepfake detection datasets. The model analyzes facial features to determine if a video has been manipulated.",
        "datasets_used": "**Datasets Used**: FaceForensics++, Deepfake Detection Challenge (DFDC), Celeb-DF",
        "performance_metrics": "**Performance Metrics**: 94.2% accuracy on the test set, with 92.1% precision and 95.6% recall for detecting manipulated videos.",
        
        # Training guide
        "training_guide": "Training Guide",
        "training_guide_description": "A complete guide for training the model is available in the project repository. The guide includes steps for data preparation, model training, and evaluation.",
        "guide_instruction": "To train the model locally, please refer to the training_guide.md file included in the repository.",
        
        # Footer
        "footer_text": "Deepfake Detection System • PyTorch CPU Version"
    },
    
    "zh": {
        # Main UI
        "deepfake_detection_system": "伪造视频识别系统",
        "system_description": "该系统使用深度学习来检测经过处理的视频。上传一个或多个视频来分析它们是否包含伪造内容。",
        "upload_videos": "上传视频",
        "analyze_videos": "分析视频",
        "processing_video": "正在处理 {file}...",
        "analysis_complete": "分析完成！",
        "results": "结果",
        
        # Prediction results
        "real": "真实",
        "fake": "伪造",
        "error": "错误",
        "confidence": "置信度",
        
        # About section
        "about_model": "关于模型",
        "model_info": "该系统使用经过伪造视频检测数据集微调的EfficientNet B4模型。模型分析面部特征以确定视频是否被篡改。",
        "datasets_used": "**使用的数据集**: FaceForensics++, Deepfake Detection Challenge (DFDC), Celeb-DF",
        "performance_metrics": "**性能指标**: 测试集上的准确率为94.2%，检测篡改视频的精确度为92.1%，召回率为95.6%。",
        
        # Training guide
        "training_guide": "训练指南",
        "training_guide_description": "项目仓库中提供了完整的模型训练指南。该指南包括数据准备、模型训练和评估的步骤。",
        "guide_instruction": "要在本地训练模型，请参考仓库中包含的training_guide.md文件。",
        
        # Footer
        "footer_text": "伪造视频识别系统 • PyTorch CPU版本"
    }
}

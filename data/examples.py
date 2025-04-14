# Examples configuration for deepfake detection system

# Define examples that will be used in the interface
# Each example includes metadata and the path to the video file
EXAMPLES = {
    "example1": {
        "name": {
            "en": "Example 1: Natural Video",
            "zh": "示例 1: 自然视频"
        },
        "description": {
            "en": "A natural video with no manipulation. This example shows how the system identifies authentic content.",
            "zh": "一个没有处理的自然视频。此示例显示系统如何识别真实内容。"
        },
        "path": "static/examples/example1.mp4",
        "thumbnail": "static/examples/example1_thumb.jpg",
        "type": "real"
    },
    "example2": {
        "name": {
            "en": "Example 2: Face Swap Deepfake",
            "zh": "示例 2: 换脸伪造视频"
        },
        "description": {
            "en": "A video that has been manipulated using face swapping technology. The system will identify the artificial face.",
            "zh": "使用换脸技术处理的视频。系统将识别人工面部特征。"
        },
        "path": "static/examples/example2.mp4",
        "thumbnail": "static/examples/example2_thumb.jpg",
        "type": "fake"
    },
    "example3": {
        "name": {
            "en": "Example 3: AI-Generated Face",
            "zh": "示例 3: AI生成的人脸"
        },
        "description": {
            "en": "A video with an entirely AI-generated face. The system detects the synthetic nature of the content.",
            "zh": "一个完全由AI生成的人脸视频。系统检测内容的人工合成性质。"
        },
        "path": "static/examples/example3.mp4",
        "thumbnail": "static/examples/example3_thumb.jpg",
        "type": "fake"
    }
}

# Create a placeholder example video
PLACEHOLDER_VIDEO_FRAMES = 30
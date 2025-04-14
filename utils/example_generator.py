import os
import cv2
import numpy as np
from pathlib import Path

def generate_placeholder_examples():
    """
    Generate placeholder example videos for demonstration purposes
    
    Note: In a real application, you should use actual videos for testing
    rather than generated ones, but this is useful for demonstration
    when no real videos are available.
    """
    from data.examples import EXAMPLES, PLACEHOLDER_VIDEO_FRAMES
    
    # Create examples directory if it doesn't exist
    os.makedirs("static/examples", exist_ok=True)
    
    # Check if we need to generate examples
    all_exist = True
    for example_id, example in EXAMPLES.items():
        if not os.path.exists(example["path"]):
            all_exist = False
            break
    
    if all_exist:
        print("All example videos already exist, skipping generation")
        return
    
    print("Generating placeholder example videos...")
    
    # Generate each example video
    for example_id, example in EXAMPLES.items():
        video_path = example["path"]
        thumb_path = example["thumbnail"]
        
        # Skip if file already exists
        if os.path.exists(video_path):
            print(f"Example {example_id} already exists at {video_path}")
            continue
            
        print(f"Generating example {example_id} at {video_path}")
        
        # Generate based on type
        if example["type"] == "real":
            _generate_real_example(video_path, thumb_path, PLACEHOLDER_VIDEO_FRAMES)
        else:  # fake
            _generate_fake_example(video_path, thumb_path, PLACEHOLDER_VIDEO_FRAMES, example_id)
            
    print("Example generation complete")

def _generate_real_example(output_path, thumb_path, num_frames):
    """Generate a natural-looking video"""
    # Video settings
    width, height = 640, 480
    fps = 30
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(num_frames):
        # Create a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with a blue gradient
        for y in range(height):
            blue_val = int(255 * (y / height))
            frame[y, :] = [blue_val, 180, 100]  # BGR format
        
        # Add a moving circle to simulate movement
        center_x = int(width * 0.5 + width * 0.3 * np.sin(i / 20))
        center_y = int(height * 0.5 + height * 0.2 * np.cos(i / 15))
        
        cv2.circle(frame, (center_x, center_y), 50, (200, 200, 200), -1)
        cv2.circle(frame, (center_x, center_y), 20, (50, 50, 250), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i+1}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add "REAL VIDEO" text
        cv2.putText(frame, "REAL VIDEO EXAMPLE", (width//2 - 150, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame
        writer.write(frame)
        
        # Save the first frame as thumbnail
        if i == 0:
            cv2.imwrite(thumb_path, frame)
    
    writer.release()
    print(f"Generated real example video: {output_path}")

def _generate_fake_example(output_path, thumb_path, num_frames, example_id):
    """Generate a video with artificial elements"""
    # Video settings
    width, height = 640, 480
    fps = 30
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Choose different colors based on example_id for variety
    if example_id == "example2":
        primary_color = (50, 100, 255)  # Red-orange in BGR
        secondary_color = (30, 255, 255)  # Yellow in BGR
    else:
        primary_color = (255, 50, 100)  # Purple in BGR
        secondary_color = (255, 150, 30)  # Cyan in BGR
    
    # Generate frames
    for i in range(num_frames):
        # Create a gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with a gradient
        for y in range(height):
            val = int(255 * (y / height))
            frame[y, :] = [val, val//2, 255-val]  # BGR format
        
        # Create an artificial face-like shape
        center_x = width // 2
        center_y = height // 2
        
        # Face oval
        cv2.ellipse(frame, (center_x, center_y), (100, 130), 0, 0, 360, (255, 255, 255), -1)
        
        # Add slightly moving eyes (uncanny effect)
        eye_x_offset = int(10 * np.sin(i / 5))
        left_eye_x = center_x - 40 + eye_x_offset
        right_eye_x = center_x + 40 + eye_x_offset
        eye_y = center_y - 30
        
        # Eyes
        cv2.circle(frame, (left_eye_x, eye_y), 15, primary_color, -1)
        cv2.circle(frame, (right_eye_x, eye_y), 15, primary_color, -1)
        
        # Unnatural pupils that move independently
        left_pupil_x = left_eye_x + int(5 * np.sin(i / 3))
        right_pupil_x = right_eye_x + int(5 * np.cos(i / 4))
        
        cv2.circle(frame, (left_pupil_x, eye_y), 5, (0, 0, 0), -1)
        cv2.circle(frame, (right_pupil_x, eye_y), 5, (0, 0, 0), -1)
        
        # Mouth with unnatural movement
        mouth_y = center_y + 40
        mouth_width = 60 + int(20 * np.sin(i / 7))
        mouth_height = 20 + int(10 * np.sin(i / 9))
        
        cv2.ellipse(frame, (center_x, mouth_y), (mouth_width, mouth_height), 
                   0, 0, 180, secondary_color, -1)
        
        # Add a glitch effect occasionally
        if i % 10 == 0:
            # Random horizontal lines
            for _ in range(5):
                y = np.random.randint(center_y - 100, center_y + 100)
                cv2.line(frame, (center_x - 120, y), (center_x + 120, y), (0, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i+1}", (30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add "FAKE VIDEO" text
        cv2.putText(frame, "FAKE VIDEO EXAMPLE", (width//2 - 150, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame
        writer.write(frame)
        
        # Save the first frame as thumbnail
        if i == 0:
            cv2.imwrite(thumb_path, frame)
    
    writer.release()
    print(f"Generated fake example video: {output_path}")

if __name__ == "__main__":
    generate_placeholder_examples()
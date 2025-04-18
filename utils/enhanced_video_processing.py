import cv2
import numpy as np
import os
import time
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_optimized(video_path, max_frames=20, sampling_strategy='uniform'):
    """
    Extract frames from a video file with improved strategies

    Args:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
        sampling_strategy (str): Strategy for frame sampling ('uniform', 'scene_change', 'keyframes')

    Returns:
        list: List of extracted frames
    """
    frames = []

    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        print(f"Video info: {frame_count} frames, {fps} FPS, {duration:.2f}s duration")

        # Handle different sampling strategies
        if sampling_strategy == 'scene_change':
            frames = _extract_scene_changes(cap, frame_count, max_frames)
        elif sampling_strategy == 'keyframes':
            frames = _extract_keyframes(cap, frame_count, max_frames)
        else:  # Default to uniform sampling
            frames = _extract_uniform_frames(cap, frame_count, max_frames)

        # Close the video
        cap.release()

    except Exception as e:
        print(f"Error extracting frames: {e}")

    # If we failed to extract frames with the chosen strategy, fall back to basic uniform sampling
    if not frames:
        print(f"Falling back to basic frame extraction for {video_path}")
        frames = _extract_basic_frames(video_path, max_frames)

    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def _extract_uniform_frames(cap, frame_count, max_frames):
    """Extract frames at uniformly spaced intervals"""
    frames = []

    if frame_count <= 0:
        return frames

    # Calculate interval between frames
    interval = max(1, frame_count // max_frames)

    for i in range(0, frame_count, interval):
        if len(frames) >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    return frames

def _extract_scene_changes(cap, frame_count, max_frames):
    """Extract frames at points of scene changes"""
    frames = []
    prev_frame = None

    # Set threshold for scene change detection
    threshold = 30.0

    # Process video and detect scene changes
    frame_indices = []

    # First, get a uniform sampling of frames to check for scene changes
    check_count = min(frame_count, max_frames * 3)
    interval = max(1, frame_count // check_count)

    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            continue

        if prev_frame is not None:
            # Convert to grayscale for simpler comparison
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate difference between frames
            diff = cv2.absdiff(gray1, gray2)
            non_zero_count = np.count_nonzero(diff)
            score = non_zero_count * 100.0 / diff.size

            if score > threshold:
                frame_indices.append(i)

        prev_frame = frame

    # If we didn't detect enough scene changes, add some uniformly sampled frames
    if len(frame_indices) < max_frames:
        additional_indices = _get_uniform_indices(frame_count, max_frames - len(frame_indices))
        frame_indices.extend(additional_indices)

    # Sort indices and limit to max_frames
    frame_indices = sorted(list(set(frame_indices)))[:max_frames]

    # Extract the selected frames
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    return frames

def _extract_keyframes(cap, frame_count, max_frames):
    """Try to extract keyframes (I-frames) from the video"""
    frames = []

    # This is a simplified approach; true keyframe extraction would require
    # codec-specific knowledge or libraries like ffmpeg

    # For simplicity, we'll just sample some frames from the beginning,
    # middle and end, where keyframes are often found

    points = []

    # Beginning frames
    beginning_count = max_frames // 3
    for i in range(min(beginning_count, frame_count)):
        points.append(i)

    # Middle frames
    middle_start = frame_count // 2 - (max_frames // 6)
    middle_count = max_frames // 3
    for i in range(middle_start, min(middle_start + middle_count, frame_count)):
        points.append(i)

    # End frames
    end_start = max(0, frame_count - (max_frames // 3))
    for i in range(end_start, frame_count):
        points.append(i)

    # Extract the frames
    for idx in points[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    return frames

def _extract_basic_frames(video_path, max_frames):
    """Very basic frame extraction as fallback"""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return frames

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Take every 5th frame to avoid too similar frames
        if count % 5 == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames

def _get_uniform_indices(total, count):
    """Helper function to get uniformly distributed indices"""
    if total <= 0 or count <= 0:
        return []

    indices = []
    interval = total / float(count)

    for i in range(count):
        idx = int(i * interval)
        if idx < total:
            indices.append(idx)

    return indices

def parallel_process_frames(frames, detector, max_workers=4):
    """
    Process frames in parallel for faster analysis

    Args:
        frames (list): List of video frames
        detector: The deepfake detector model
        max_workers (int): Maximum number of parallel workers

    Returns:
        list: List of (is_fake, confidence) tuples for each frame
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all frame processing tasks
        future_to_frame = {
            executor.submit(detector.detect_from_image, frame): i
            for i, frame in enumerate(frames)
        }

        # Process results as they complete
        for future in as_completed(future_to_frame):
            frame_idx = future_to_frame[future]
            try:
                is_fake, confidence = future.result()
                results.append((frame_idx, is_fake, confidence))
                if (frame_idx + 1) % 5 == 0 or frame_idx == 0 or frame_idx == len(frames) - 1:
                    print(f"Processed frame {frame_idx+1}/{len(frames)}: {'Fake' if is_fake else 'Real'} ({confidence:.2f}% confidence)")
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Add a default result for failed frames
                results.append((frame_idx, False, 0.0))

    # Sort by frame index to maintain order
    results.sort(key=lambda x: x[0])

    # Remove frame indices
    return [(is_fake, confidence) for _, is_fake, confidence in results]

def enhanced_process_video(video_path, detector, frames_to_sample=20, sampling_strategy='uniform', face_detector=None):
    """
    Enhanced video processing with better sampling and parallel processing

    Args:
        video_path (str): Path to the video file
        detector: The deepfake detector model
        frames_to_sample (int): Number of frames to sample for analysis
        sampling_strategy (str): Strategy for frame sampling
        face_detector: Optional face detector to highlight detected faces

    Returns:
        tuple: (result (str), confidence (float), processing_stats (dict), analyzed_frames (list))
    """
    start_time = time.time()
    processing_stats = {
        'video_path': video_path,
        'frames_sampled': 0,
        'sampling_strategy': sampling_strategy,
        'processing_time': 0
    }

    analyzed_frames = []  # Will store tuples of (frame, is_fake, confidence)

    try:
        # Extract frames from the video
        frames = extract_frames_optimized(
            video_path,
            max_frames=frames_to_sample,
            sampling_strategy=sampling_strategy
        )

        if not frames:
            print(f"Error: No frames could be extracted from {video_path}")
            return "Error", 0.0, processing_stats, []

        processing_stats['frames_sampled'] = len(frames)

        # Process frames in parallel
        frame_results = parallel_process_frames(frames, detector)

        # Store frames with their analysis results
        for frame, (is_fake, confidence) in zip(frames, frame_results):
            # Add visual indicators to the frame
            annotated_frame = frame.copy()
            height, width = annotated_frame.shape[:2]

            # Try to detect faces if a face detector is provided
            if face_detector is not None:
                try:
                    # Convert to RGB if needed (face detector might expect RGB)
                    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = annotated_frame

                    # Detect faces
                    faces = face_detector.detect_faces(rgb_frame)

                    # Draw detection points on faces
                    for face in faces:
                        x, y, w, h = face
                        # Draw rectangle around face
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                        # Add detection points at key facial landmarks (approximated)
                        # Eyes (approximate positions)
                        left_eye_x, left_eye_y = x + w//4, y + h//3
                        right_eye_x, right_eye_y = x + 3*w//4, y + h//3

                        # Nose (approximate position)
                        nose_x, nose_y = x + w//2, y + h//2

                        # Mouth (approximate position)
                        mouth_x, mouth_y = x + w//2, y + 2*h//3

                        # Draw detection points
                        point_color = (0, 0, 255) if is_fake else (0, 255, 0)  # Red for fake, Green for real
                        point_size = max(3, int(width / 200))  # Scale point size based on image width

                        cv2.circle(annotated_frame, (left_eye_x, left_eye_y), point_size, point_color, -1)
                        cv2.circle(annotated_frame, (right_eye_x, right_eye_y), point_size, point_color, -1)
                        cv2.circle(annotated_frame, (nose_x, nose_y), point_size, point_color, -1)
                        cv2.circle(annotated_frame, (mouth_x, mouth_y), point_size, point_color, -1)

                        # Add a label near the face
                        face_label = "FAKE" if is_fake else "REAL"
                        cv2.putText(annotated_frame, face_label, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 2)
                except Exception as e:
                    print(f"Error detecting faces: {e}")

            # Add a colored border based on fakeness
            border_color = (0, 0, 255) if is_fake else (0, 255, 0)  # Red for fake, Green for real
            annotated_frame = cv2.copyMakeBorder(annotated_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

            # Add text with confidence score
            text = f"{'FAKE' if is_fake else 'REAL'}: {confidence:.1f}%"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = width / 1000  # Scale font based on image width
            thickness = max(1, int(width / 500))

            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(annotated_frame, (10, height - text_height - 20), (text_width + 20, height + 20), (0, 0, 0), -1)

            # Add text
            cv2.putText(annotated_frame, text, (15, height - 10), font, font_scale, (255, 255, 255), thickness)

            # Add the annotated frame to the results
            analyzed_frames.append((annotated_frame, is_fake, confidence))

        # Analyze results
        fake_count = sum(1 for is_fake, _ in frame_results if is_fake)
        fake_confidence = sum(confidence for is_fake, confidence in frame_results if is_fake)

        # Calculate the percentage of frames detected as fake
        fake_percentage = (fake_count / len(frames)) * 100 if frames else 0

        # Calculate average confidence
        avg_confidence = fake_confidence / fake_count if fake_count > 0 else 0

        # Determine final result
        if fake_percentage > 40:
            result = "Fake"
            final_confidence = avg_confidence
        else:
            result = "Real"
            final_confidence = 100 - (avg_confidence * fake_count / len(frames)) if frames else 0

        # Update processing stats
        processing_stats['processing_time'] = time.time() - start_time
        processing_stats['fake_frame_percentage'] = fake_percentage
        processing_stats['frame_count'] = len(frames)
        processing_stats['fake_frame_count'] = fake_count

        return result, final_confidence, processing_stats, analyzed_frames

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Error processing video: {e}")
        processing_stats['error'] = str(e)
        processing_stats['processing_time'] = processing_time
        return f"Error: {str(e)}", 0.0, processing_stats, []
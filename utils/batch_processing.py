import os
import time
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils.enhanced_video_processing import enhanced_process_video

class BatchProcessor:
    """
    Handles batch processing of multiple videos
    """
    def __init__(self, detector, max_workers=4, max_queue_size=100):
        """
        Initialize the batch processor
        
        Args:
            detector: The deepfake detector model
            max_workers (int): Maximum number of worker threads
            max_queue_size (int): Maximum size of the processing queue
        """
        self.detector = detector
        self.max_workers = max_workers
        self.task_queue = queue.Queue(maxsize=max_queue_size)
        self.results = {}
        self.processing_thread = None
        self.is_processing = False
        self.callbacks = {}
        self.stats = {
            "total_processed": 0,
            "total_time": 0,
            "avg_time": 0,
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "in_progress": 0
        }
    
    def add_task(self, video_path, task_id=None, metadata=None, callback=None):
        """
        Add a video processing task to the queue
        
        Args:
            video_path (str): Path to the video file
            task_id (str, optional): Unique identifier for the task
            metadata (dict, optional): Additional metadata for the task
            callback (callable, optional): Function to call when task completes
            
        Returns:
            str: Task ID
        """
        if task_id is None:
            task_id = f"task_{int(time.time())}_{len(self.results)}"
        
        if metadata is None:
            metadata = {}
        
        task = {
            "id": task_id,
            "video_path": video_path,
            "status": "queued",
            "added_time": time.time(),
            "metadata": metadata
        }
        
        try:
            self.task_queue.put(task, block=False)
            self.results[task_id] = task
            self.stats["queued"] += 1
            
            if callback:
                self.callbacks[task_id] = callback
            
            # Start processing thread if not already running
            if not self.is_processing:
                self._start_processing()
            
            return task_id
        except queue.Full:
            print(f"Task queue is full. Cannot add task for {video_path}")
            return None
    
    def add_batch(self, video_paths, metadata_list=None, callback=None):
        """
        Add multiple videos for batch processing
        
        Args:
            video_paths (list): List of paths to video files
            metadata_list (list, optional): List of metadata dictionaries
            callback (callable, optional): Function to call when all tasks complete
            
        Returns:
            list: List of task IDs
        """
        task_ids = []
        
        if metadata_list is None:
            metadata_list = [None] * len(video_paths)
        
        for i, video_path in enumerate(video_paths):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            task_id = self.add_task(video_path, metadata=metadata)
            if task_id:
                task_ids.append(task_id)
        
        # Add a callback for the entire batch if provided
        if callback and task_ids:
            batch_id = f"batch_{int(time.time())}"
            self.callbacks[batch_id] = {
                "callback": callback,
                "task_ids": task_ids,
                "completed": 0
            }
        
        return task_ids
    
    def _start_processing(self):
        """Start the processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.processing_thread.start()
            print("Started batch processing thread")
    
    def _process_queue(self):
        """Process tasks from the queue"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}
            
            while self.is_processing:
                # Submit new tasks to the thread pool
                while len(future_to_task) < self.max_workers and not self.task_queue.empty():
                    try:
                        task = self.task_queue.get(block=False)
                        task_id = task["id"]
                        
                        # Update task status
                        task["status"] = "processing"
                        task["start_time"] = time.time()
                        self.results[task_id] = task
                        self.stats["in_progress"] += 1
                        self.stats["queued"] -= 1
                        
                        # Submit task to thread pool
                        future = executor.submit(
                            self._process_video,
                            task["video_path"],
                            task_id,
                            task.get("metadata", {})
                        )
                        future_to_task[future] = task_id
                        
                    except queue.Empty:
                        break
                
                # Check for completed tasks
                for future in list(as_completed(future_to_task.keys(), timeout=0.1)):
                    task_id = future_to_task[future]
                    del future_to_task[task_id]
                    
                    try:
                        result, confidence, stats = future.result()
                        
                        # Update task with results
                        task = self.results[task_id]
                        task["status"] = "completed"
                        task["result"] = result
                        task["confidence"] = confidence
                        task["stats"] = stats
                        task["end_time"] = time.time()
                        task["processing_time"] = task["end_time"] - task["start_time"]
                        
                        # Update stats
                        self.stats["completed"] += 1
                        self.stats["in_progress"] -= 1
                        self.stats["total_processed"] += 1
                        self.stats["total_time"] += task["processing_time"]
                        self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_processed"]
                        
                        # Call task callback if exists
                        if task_id in self.callbacks and callable(self.callbacks[task_id]):
                            try:
                                self.callbacks[task_id](task)
                                del self.callbacks[task_id]
                            except Exception as e:
                                print(f"Error in task callback for {task_id}: {e}")
                        
                        # Check if this task is part of a batch
                        for batch_id, batch_info in list(self.callbacks.items()):
                            if isinstance(batch_info, dict) and "task_ids" in batch_info:
                                if task_id in batch_info["task_ids"]:
                                    batch_info["completed"] += 1
                                    
                                    # If all tasks in batch are complete, call batch callback
                                    if batch_info["completed"] == len(batch_info["task_ids"]):
                                        try:
                                            batch_results = {tid: self.results[tid] for tid in batch_info["task_ids"]}
                                            batch_info["callback"](batch_results)
                                            del self.callbacks[batch_id]
                                        except Exception as e:
                                            print(f"Error in batch callback for {batch_id}: {e}")
                        
                    except Exception as e:
                        # Update task with error
                        task = self.results[task_id]
                        task["status"] = "failed"
                        task["error"] = str(e)
                        task["end_time"] = time.time()
                        
                        # Update stats
                        self.stats["failed"] += 1
                        self.stats["in_progress"] -= 1
                
                # If no tasks are being processed and queue is empty, we can stop
                if len(future_to_task) == 0 and self.task_queue.empty():
                    time.sleep(1)  # Wait a bit to see if new tasks arrive
                    if self.task_queue.empty():
                        break
                
                # Small sleep to prevent CPU hogging
                time.sleep(0.1)
        
        self.is_processing = False
        print("Batch processing thread stopped")
    
    def _process_video(self, video_path, task_id, metadata):
        """
        Process a single video
        
        Args:
            video_path (str): Path to the video file
            task_id (str): Task identifier
            metadata (dict): Additional metadata
            
        Returns:
            tuple: (result, confidence, stats)
        """
        frames_to_sample = metadata.get("frames_to_sample", 20)
        sampling_strategy = metadata.get("sampling_strategy", "uniform")
        
        print(f"Processing video {video_path} (Task ID: {task_id})")
        
        # Process the video
        result, confidence, stats = enhanced_process_video(
            video_path,
            self.detector,
            frames_to_sample=frames_to_sample,
            sampling_strategy=sampling_strategy
        )
        
        print(f"Completed processing {video_path}: {result} ({confidence:.2f}%)")
        return result, confidence, stats
    
    def get_task_status(self, task_id):
        """
        Get the status of a task
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            dict: Task status and results
        """
        if task_id in self.results:
            return self.results[task_id]
        else:
            return None
    
    def get_all_results(self):
        """
        Get all task results
        
        Returns:
            dict: Dictionary of all task results
        """
        return self.results
    
    def get_stats(self):
        """
        Get batch processing statistics
        
        Returns:
            dict: Processing statistics
        """
        return self.stats
    
    def stop(self):
        """Stop the processing thread"""
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
            print("Stopped batch processing thread")
    
    def clear_completed(self):
        """Clear completed tasks from results"""
        for task_id in list(self.results.keys()):
            if self.results[task_id]["status"] in ["completed", "failed"]:
                del self.results[task_id]
        
        print("Cleared completed and failed tasks")

import os
import json
import time
import uuid
from pathlib import Path
import threading
from datetime import datetime

from utils.logging_system import LoggingSystem

class FeedbackSystem:
    """
    System for collecting and managing user feedback
    """
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FeedbackSystem, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, feedback_dir="feedback"):
        """
        Initialize the feedback system
        
        Args:
            feedback_dir (str): Directory to store feedback data
        """
        # Only initialize once
        if self._initialized:
            return
        
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feedback storage
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback_data = self._load_feedback()
        
        # Initialize logger
        self.logger = LoggingSystem().user_logger
        
        self._initialized = True
        self.logger.info("Feedback system initialized")
    
    def _load_feedback(self):
        """Load feedback data from disk"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback data: {e}")
                return {"feedback": [], "stats": {}}
        else:
            return {"feedback": [], "stats": {}}
    
    def _save_feedback(self):
        """Save feedback data to disk"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            
            # Also save a backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.feedback_dir / f"feedback_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
                
            self.logger.info(f"Feedback data saved to {self.feedback_file}")
        except Exception as e:
            self.logger.error(f"Error saving feedback data: {e}")
    
    def add_feedback(self, feedback_type, rating, comment=None, metadata=None):
        """
        Add user feedback
        
        Args:
            feedback_type (str): Type of feedback (e.g., 'detection_result', 'ui', 'general')
            rating (int): Numerical rating (1-5)
            comment (str, optional): User comment
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Feedback ID
        """
        # Validate rating
        rating = max(1, min(5, rating))
        
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        
        # Create feedback entry
        feedback_entry = {
            "id": feedback_id,
            "type": feedback_type,
            "rating": rating,
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if comment:
            feedback_entry["comment"] = comment
        
        if metadata:
            feedback_entry["metadata"] = metadata
        
        # Add to feedback data
        with self._lock:
            self.feedback_data["feedback"].append(feedback_entry)
            
            # Update statistics
            if "by_type" not in self.feedback_data["stats"]:
                self.feedback_data["stats"]["by_type"] = {}
            
            if feedback_type not in self.feedback_data["stats"]["by_type"]:
                self.feedback_data["stats"]["by_type"][feedback_type] = {
                    "count": 0,
                    "ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                    "avg_rating": 0,
                    "total_rating": 0
                }
            
            # Update type stats
            type_stats = self.feedback_data["stats"]["by_type"][feedback_type]
            type_stats["count"] += 1
            type_stats["ratings"][rating] += 1
            type_stats["total_rating"] += rating
            type_stats["avg_rating"] = type_stats["total_rating"] / type_stats["count"]
            
            # Update overall stats
            if "overall" not in self.feedback_data["stats"]:
                self.feedback_data["stats"]["overall"] = {
                    "count": 0,
                    "ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                    "avg_rating": 0,
                    "total_rating": 0
                }
            
            overall_stats = self.feedback_data["stats"]["overall"]
            overall_stats["count"] += 1
            overall_stats["ratings"][rating] += 1
            overall_stats["total_rating"] += rating
            overall_stats["avg_rating"] = overall_stats["total_rating"] / overall_stats["count"]
            
            # Save feedback data
            self._save_feedback()
        
        self.logger.info(f"Added feedback: {feedback_type}, rating: {rating}")
        return feedback_id
    
    def get_feedback(self, feedback_id=None, feedback_type=None, limit=None):
        """
        Get feedback data
        
        Args:
            feedback_id (str, optional): Specific feedback ID to retrieve
            feedback_type (str, optional): Type of feedback to retrieve
            limit (int, optional): Maximum number of feedback entries to retrieve
            
        Returns:
            list: Feedback entries
        """
        with self._lock:
            if feedback_id:
                # Find specific feedback entry
                for entry in self.feedback_data["feedback"]:
                    if entry["id"] == feedback_id:
                        return [entry]
                return []
            
            elif feedback_type:
                # Filter by feedback type
                filtered = [entry for entry in self.feedback_data["feedback"] if entry["type"] == feedback_type]
                
                # Apply limit if specified
                if limit and limit > 0:
                    return filtered[-limit:]
                else:
                    return filtered
            
            else:
                # Get all feedback
                if limit and limit > 0:
                    return self.feedback_data["feedback"][-limit:]
                else:
                    return self.feedback_data["feedback"]
    
    def get_stats(self, feedback_type=None):
        """
        Get feedback statistics
        
        Args:
            feedback_type (str, optional): Type of feedback to get statistics for
            
        Returns:
            dict: Feedback statistics
        """
        with self._lock:
            if feedback_type:
                if feedback_type in self.feedback_data["stats"]["by_type"]:
                    return self.feedback_data["stats"]["by_type"][feedback_type]
                else:
                    return {"count": 0, "ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, "avg_rating": 0, "total_rating": 0}
            else:
                return self.feedback_data["stats"]
    
    def get_recent_feedback(self, count=10):
        """
        Get recent feedback entries
        
        Args:
            count (int): Number of recent entries to retrieve
            
        Returns:
            list: Recent feedback entries
        """
        with self._lock:
            # Sort by timestamp (newest first)
            sorted_feedback = sorted(
                self.feedback_data["feedback"],
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_feedback[:count]
    
    def export_feedback(self, output_file=None):
        """
        Export feedback data to a file
        
        Args:
            output_file (str, optional): Path to output file
            
        Returns:
            str: Path to the exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.feedback_dir / f"feedback_export_{timestamp}.json"
        
        with self._lock:
            try:
                with open(output_file, 'w') as f:
                    json.dump(self.feedback_data, f, indent=2)
                
                self.logger.info(f"Feedback data exported to {output_file}")
                return str(output_file)
            except Exception as e:
                self.logger.error(f"Error exporting feedback data: {e}")
                return None
    
    def analyze_feedback(self):
        """
        Analyze feedback data to generate insights
        
        Returns:
            dict: Feedback analysis results
        """
        with self._lock:
            analysis = {
                "overall_rating": 0,
                "rating_distribution": {},
                "by_type": {},
                "recent_trend": {},
                "common_issues": [],
                "positive_aspects": []
            }
            
            # Get overall stats
            if "overall" in self.feedback_data["stats"]:
                analysis["overall_rating"] = self.feedback_data["stats"]["overall"]["avg_rating"]
                analysis["rating_distribution"] = self.feedback_data["stats"]["overall"]["ratings"]
            
            # Get stats by type
            if "by_type" in self.feedback_data["stats"]:
                for feedback_type, stats in self.feedback_data["stats"]["by_type"].items():
                    analysis["by_type"][feedback_type] = {
                        "avg_rating": stats["avg_rating"],
                        "count": stats["count"]
                    }
            
            # Analyze recent trend (last 30 days)
            now = time.time()
            thirty_days_ago = now - (30 * 24 * 60 * 60)
            
            recent_feedback = [
                entry for entry in self.feedback_data["feedback"]
                if entry["timestamp"] >= thirty_days_ago
            ]
            
            if recent_feedback:
                # Group by day
                days = {}
                for entry in recent_feedback:
                    day = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d")
                    if day not in days:
                        days[day] = {"count": 0, "total_rating": 0}
                    
                    days[day]["count"] += 1
                    days[day]["total_rating"] += entry["rating"]
                
                # Calculate daily averages
                for day, stats in days.items():
                    days[day]["avg_rating"] = stats["total_rating"] / stats["count"]
                
                analysis["recent_trend"] = days
            
            # Analyze comments for common issues and positive aspects
            comments = [
                entry for entry in self.feedback_data["feedback"]
                if "comment" in entry and entry["comment"]
            ]
            
            # Group comments by rating
            negative_comments = [entry for entry in comments if entry["rating"] <= 2]
            positive_comments = [entry for entry in comments if entry["rating"] >= 4]
            
            # Extract common issues (simplified approach)
            if negative_comments:
                analysis["common_issues"] = [
                    {
                        "id": comment["id"],
                        "type": comment["type"],
                        "rating": comment["rating"],
                        "comment": comment["comment"]
                    }
                    for comment in negative_comments[:5]
                ]
            
            # Extract positive aspects
            if positive_comments:
                analysis["positive_aspects"] = [
                    {
                        "id": comment["id"],
                        "type": comment["type"],
                        "rating": comment["rating"],
                        "comment": comment["comment"]
                    }
                    for comment in positive_comments[:5]
                ]
            
            return analysis

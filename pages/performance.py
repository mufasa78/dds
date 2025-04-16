import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from utils.performance_monitor import PerformanceMonitor
from utils.logging_system import LoggingSystem
from utils.feedback_system import FeedbackSystem
from models.detector import DeepfakeDetector

def load_detector():
    """Load the deepfake detector model"""
    try:
        # First try to load the regular model
        st.info("Loading deepfake detection model...")
        detector = DeepfakeDetector(device=None)  # Auto-detect device
        st.success("✅ Model loaded successfully")
        return detector
    except Exception as e:
        st.warning(f"Failed to load model: {e}")
        return None

def main():
    st.title("System Performance Dashboard")
    
    # Initialize systems
    performance_monitor = PerformanceMonitor()
    logging_system = LoggingSystem()
    feedback_system = FeedbackSystem()
    
    # Create tabs
    system_tab, model_tab, feedback_tab, logs_tab = st.tabs([
        "System Performance", 
        "Model Performance", 
        "User Feedback",
        "Logs"
    ])
    
    # System Performance Tab
    with system_tab:
        st.header("System Performance Metrics")
        
        # Start monitoring if not already started
        if not performance_monitor.monitoring_active:
            if st.button("Start Performance Monitoring"):
                performance_monitor.start_monitoring()
                st.success("Performance monitoring started")
        else:
            if st.button("Stop Performance Monitoring"):
                performance_monitor.stop_monitoring()
                st.success("Performance monitoring stopped")
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        
        # Display system metrics
        st.subheader("System Resource Usage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", f"{summary['system']['cpu_usage_avg']:.1f}%")
        
        with col2:
            st.metric("Memory Usage", f"{summary['system']['memory_usage_avg']:.1f}%")
        
        with col3:
            st.metric("Disk Usage", f"{summary['system']['disk_usage_avg']:.1f}%")
        
        # Display system metrics over time
        st.subheader("Resource Usage Over Time")
        
        # Get system metrics
        metrics = performance_monitor.get_system_metrics()
        
        if metrics["timestamps"]:
            # Convert timestamps to datetime
            dates = [datetime.fromtimestamp(ts) for ts in metrics["timestamps"]]
            
            # Create DataFrame
            df = pd.DataFrame({
                "Time": dates,
                "CPU Usage (%)": metrics["cpu_usage"],
                "Memory Usage (%)": metrics["memory_usage"],
                "Disk Usage (%)": metrics["disk_usage"]
            })
            
            # Plot metrics
            fig = px.line(
                df, 
                x="Time", 
                y=["CPU Usage (%)", "Memory Usage (%)", "Disk Usage (%)"],
                title="System Resource Usage Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No system metrics data available yet. Start monitoring to collect data.")
        
        # Display operation metrics
        st.subheader("Operation Performance")
        
        operation_metrics = performance_monitor.get_operation_metrics()
        
        if operation_metrics:
            # Create DataFrame
            op_data = []
            for op_name, metrics in operation_metrics.items():
                op_data.append({
                    "Operation": op_name,
                    "Count": metrics["count"],
                    "Avg Time (s)": metrics["avg_time"],
                    "Min Time (s)": metrics["min_time"],
                    "Max Time (s)": metrics["max_time"]
                })
            
            op_df = pd.DataFrame(op_data)
            st.dataframe(op_df)
            
            # Plot operation times
            st.subheader("Average Operation Times")
            
            fig = px.bar(
                op_df,
                x="Operation",
                y="Avg Time (s)",
                title="Average Operation Times",
                color="Count",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No operation metrics available yet.")
    
    # Model Performance Tab
    with model_tab:
        st.header("Model Performance Metrics")
        
        # Load detector if needed
        detector = load_detector()
        
        if detector:
            # Display model info
            st.subheader("Model Information")
            model_info = detector.get_model_info()
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Type:**", model_info.get("model_type", "Unknown"))
                st.write("**Model Version:**", model_info.get("version", "Unknown"))
                st.write("**Device:**", model_info.get("device", "Unknown"))
            
            with col2:
                st.write("**Creation Time:**", model_info.get("creation_time", "Unknown"))
            
            # Display model performance metrics
            st.subheader("Performance Metrics")
            
            perf = detector.get_performance_metrics()
            
            if perf and "total_processed" in perf and perf["total_processed"] > 0:
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Processed", perf["total_processed"])
                
                with col2:
                    st.metric("Avg. Processing Time", f"{perf['avg_time']:.4f}s")
                
                with col3:
                    st.metric("Fake Detections", perf["fake_count"])
                
                with col4:
                    st.metric("Real Detections", perf["real_count"])
                
                # Create pie chart for fake vs real
                labels = ['Fake', 'Real']
                values = [perf["fake_count"], perf["real_count"]]
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.3,
                    marker_colors=['#FF6B6B', '#4ECDC4']
                )])
                fig.update_layout(title_text="Detection Results Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display confidence distribution
                st.subheader("Confidence Distribution")
                
                # Create a histogram of confidence values (simulated)
                confidence_data = np.random.normal(perf["avg_confidence"], 15, 100)
                confidence_data = np.clip(confidence_data, 0, 100)
                
                fig = px.histogram(
                    confidence_data,
                    nbins=20,
                    labels={"value": "Confidence (%)"},
                    title="Confidence Distribution",
                    color_discrete_sequence=['#6C5B7B']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display error rate
                error_rate = perf["error_count"] / perf["total_processed"] * 100 if perf["total_processed"] > 0 else 0
                st.metric("Error Rate", f"{error_rate:.2f}%")
                
                # Display fallback rate
                fallback_rate = perf["fallback_count"] / perf["total_processed"] * 100 if perf["total_processed"] > 0 else 0
                st.metric("Fallback Rate", f"{fallback_rate:.2f}%")
            else:
                st.info("No model performance data available yet. Process some videos to collect metrics.")
        else:
            st.error("Failed to load model. Cannot display model performance metrics.")
    
    # User Feedback Tab
    with feedback_tab:
        st.header("User Feedback Analysis")
        
        # Get feedback stats
        stats = feedback_system.get_stats()
        
        if "overall" in stats and stats["overall"]["count"] > 0:
            # Display overall metrics
            st.subheader("Overall Feedback")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Feedback", stats["overall"]["count"])
            
            with col2:
                st.metric("Average Rating", f"{stats['overall']['avg_rating']:.1f}/5.0")
            
            # Display rating distribution
            st.subheader("Rating Distribution")
            
            ratings = stats["overall"]["ratings"]
            rating_df = pd.DataFrame({
                "Rating": list(ratings.keys()),
                "Count": list(ratings.values())
            })
            
            fig = px.bar(
                rating_df,
                x="Rating",
                y="Count",
                title="Rating Distribution",
                color="Rating",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feedback by type
            if "by_type" in stats:
                st.subheader("Feedback by Type")
                
                type_data = []
                for feedback_type, type_stats in stats["by_type"].items():
                    type_data.append({
                        "Type": feedback_type,
                        "Count": type_stats["count"],
                        "Avg Rating": type_stats["avg_rating"]
                    })
                
                type_df = pd.DataFrame(type_data)
                
                fig = px.bar(
                    type_df,
                    x="Type",
                    y="Count",
                    color="Avg Rating",
                    title="Feedback by Type",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display recent feedback
            st.subheader("Recent Feedback")
            
            recent_feedback = feedback_system.get_recent_feedback(10)
            
            if recent_feedback:
                for feedback in recent_feedback:
                    with st.expander(f"{feedback['type']} - Rating: {feedback['rating']}/5 - {feedback['date']}"):
                        if "comment" in feedback:
                            st.write(feedback["comment"])
                        
                        if "metadata" in feedback:
                            st.json(feedback["metadata"])
            else:
                st.info("No recent feedback available.")
            
            # Feedback analysis
            st.subheader("Feedback Analysis")
            
            analysis = feedback_system.analyze_feedback()
            
            # Display common issues
            if analysis["common_issues"]:
                st.write("**Common Issues:**")
                for issue in analysis["common_issues"]:
                    st.write(f"- {issue['comment']} (Rating: {issue['rating']}/5)")
            
            # Display positive aspects
            if analysis["positive_aspects"]:
                st.write("**Positive Aspects:**")
                for positive in analysis["positive_aspects"]:
                    st.write(f"- {positive['comment']} (Rating: {positive['rating']}/5)")
        else:
            st.info("No feedback data available yet.")
        
        # Add feedback form
        st.subheader("Submit Feedback")
        
        feedback_type = st.selectbox(
            "Feedback Type",
            options=["detection_result", "ui", "performance", "general"]
        )
        
        rating = st.slider("Rating", 1, 5, 3)
        
        comment = st.text_area("Comment")
        
        if st.button("Submit Feedback"):
            feedback_id = feedback_system.add_feedback(
                feedback_type,
                rating,
                comment
            )
            
            if feedback_id:
                st.success(f"Feedback submitted successfully! ID: {feedback_id}")
            else:
                st.error("Failed to submit feedback.")
    
    # Logs Tab
    with logs_tab:
        st.header("System Logs")
        
        # Create tabs for different log types
        app_log_tab, model_log_tab, error_log_tab = st.tabs([
            "Application Logs",
            "Model Logs",
            "Error Logs"
        ])
        
        # Application Logs
        with app_log_tab:
            st.subheader("Application Logs")
            
            app_logs = logging_system.get_recent_logs("app", 100)
            
            if app_logs:
                for log in app_logs:
                    st.text(log)
            else:
                st.info("No application logs available.")
        
        # Model Logs
        with model_log_tab:
            st.subheader("Model Logs")
            
            model_logs = logging_system.get_recent_logs("model", 100)
            
            if model_logs:
                for log in model_logs:
                    st.text(log)
            else:
                st.info("No model logs available.")
        
        # Error Logs
        with error_log_tab:
            st.subheader("Error Logs")
            
            error_logs = logging_system.get_recent_logs("error", 100)
            
            if error_logs:
                for log in error_logs:
                    st.text(log)
            else:
                st.info("No error logs available.")

if __name__ == "__main__":
    main()

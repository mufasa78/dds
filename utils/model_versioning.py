import os
import json
import time
import shutil
from pathlib import Path

class ModelVersionManager:
    """
    Manages model versions, metadata, and history
    """
    def __init__(self, models_dir="models/versions"):
        """
        Initialize the model version manager
        
        Args:
            models_dir (str): Directory to store model versions
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.models_dir / "versions.json"
        self.current_version = None
        self._load_versions()
    
    def _load_versions(self):
        """Load version information from disk"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    self.versions = json.load(f)
                    
                # Set current version to the latest one
                if self.versions and "current_version" in self.versions:
                    self.current_version = self.versions["current_version"]
                    print(f"Loaded model version information. Current version: {self.current_version}")
                else:
                    self.versions = {"versions": {}, "current_version": None}
                    print("No current version found in version file.")
            except Exception as e:
                print(f"Error loading version information: {e}")
                self.versions = {"versions": {}, "current_version": None}
        else:
            # Initialize empty versions dictionary
            self.versions = {"versions": {}, "current_version": None}
            print("No version file found. Initializing empty version history.")
    
    def _save_versions(self):
        """Save version information to disk"""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
            print(f"Saved version information to {self.versions_file}")
        except Exception as e:
            print(f"Error saving version information: {e}")
    
    def register_model(self, model_path, version, model_type, description=""):
        """
        Register a new model version
        
        Args:
            model_path (str): Path to the model file
            version (str): Version identifier
            model_type (str): Type of model (e.g., "EfficientNetB4", "MesoNet")
            description (str): Description of the model version
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            # Create version directory
            version_dir = self.models_dir / version
            version_dir.mkdir(exist_ok=True)
            
            # Copy model file to version directory
            model_filename = Path(model_path).name
            target_path = version_dir / model_filename
            shutil.copy2(model_path, target_path)
            
            # Create metadata
            metadata = {
                "version": version,
                "model_type": model_type,
                "description": description,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_path": str(target_path),
                "performance": {}
            }
            
            # Save metadata
            with open(version_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update versions registry
            self.versions["versions"][version] = metadata
            
            # Set as current version if no current version exists
            if not self.versions["current_version"]:
                self.versions["current_version"] = version
                self.current_version = version
            
            # Save updated versions
            self._save_versions()
            
            print(f"Successfully registered model version {version}")
            return True
            
        except Exception as e:
            print(f"Error registering model version: {e}")
            return False
    
    def set_current_version(self, version):
        """
        Set the current model version
        
        Args:
            version (str): Version identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        if version in self.versions["versions"]:
            self.versions["current_version"] = version
            self.current_version = version
            self._save_versions()
            print(f"Set current model version to {version}")
            return True
        else:
            print(f"Version {version} not found")
            return False
    
    def get_model_path(self, version=None):
        """
        Get the path to a model version
        
        Args:
            version (str, optional): Version identifier. If None, uses current version.
            
        Returns:
            str: Path to the model file, or None if not found
        """
        if version is None:
            version = self.current_version
        
        if not version:
            print("No version specified and no current version set")
            return None
        
        if version in self.versions["versions"]:
            return self.versions["versions"][version]["file_path"]
        else:
            print(f"Version {version} not found")
            return None
    
    def get_version_info(self, version=None):
        """
        Get information about a model version
        
        Args:
            version (str, optional): Version identifier. If None, uses current version.
            
        Returns:
            dict: Version information, or None if not found
        """
        if version is None:
            version = self.current_version
        
        if not version:
            print("No version specified and no current version set")
            return None
        
        if version in self.versions["versions"]:
            return self.versions["versions"][version]
        else:
            print(f"Version {version} not found")
            return None
    
    def list_versions(self):
        """
        List all available model versions
        
        Returns:
            list: List of version identifiers
        """
        return list(self.versions["versions"].keys())
    
    def update_performance(self, version, performance_data):
        """
        Update performance metrics for a model version
        
        Args:
            version (str): Version identifier
            performance_data (dict): Performance metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        if version in self.versions["versions"]:
            self.versions["versions"][version]["performance"] = performance_data
            self._save_versions()
            
            # Also update the metadata file
            version_dir = self.models_dir / version
            metadata_path = version_dir / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["performance"] = performance_data
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    print(f"Error updating metadata file: {e}")
            
            return True
        else:
            print(f"Version {version} not found")
            return False

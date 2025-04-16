#!/usr/bin/env python
"""
Dependency checker for the Multilingual Deepfake Detection System.
This script checks if all required packages are installed and installs them if needed.
"""

import subprocess
import sys
import os
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

def check_venv():
    """Check if running in a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists('venv_new'):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', 'venv_new'])
        return True
    return False

def activate_venv():
    """Activate the virtual environment."""
    if os.name == 'nt':  # Windows
        activate_script = os.path.join('venv_new', 'Scripts', 'activate')
    else:  # Unix/Linux/Mac
        activate_script = os.path.join('venv_new', 'bin', 'activate')
    
    if os.path.exists(activate_script):
        print(f"To activate the virtual environment, run:\n\n{activate_script}\n")
        return True
    return False

def get_requirements():
    """Get requirements from requirements.txt."""
    requirements = []
    try:
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        print("requirements.txt not found. Creating a default one...")
        default_requirements = [
            'flask>=3.1.0',
            'gdown>=5.2.0',
            'numpy>=2.2.4',
            'opencv-python>=4.11.0.86',
            'pillow>=11.2.1',
            'streamlit>=1.44.1',
            'torch>=2.6.0',
            'torchvision>=0.21.0'
        ]
        with open('requirements.txt', 'w') as f:
            f.write('\n'.join(default_requirements))
        requirements = default_requirements
    
    return requirements

def check_dependencies():
    """Check if all required packages are installed."""
    requirements = get_requirements()
    missing = []
    
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
        except (DistributionNotFound, VersionConflict):
            missing.append(requirement)
    
    return missing

def install_dependencies(missing):
    """Install missing dependencies."""
    if not missing:
        print("All dependencies are already installed.")
        return True
    
    print(f"Installing {len(missing)} missing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("All dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")
        return False

def main():
    """Main function."""
    print("Checking dependencies for Multilingual Deepfake Detection System...")
    
    # Check if running in a virtual environment
    if not check_venv():
        print("Not running in a virtual environment.")
        created = create_venv()
        if created:
            activate_venv()
            print("Please activate the virtual environment and run this script again.")
            return
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing {len(missing)} dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        
        # Install missing dependencies
        install_dependencies(missing)
    else:
        print("All dependencies are installed.")
    
    print("\nSystem is ready to run!")
    print("To start the applications, run: start_apps.bat")

if __name__ == "__main__":
    main()

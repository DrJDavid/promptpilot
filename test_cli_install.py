#!/usr/bin/env python
"""
Test script to verify PromptPilot CLI installation.
This script checks if the promptpilot package is installed and 
the CLI command is available.
"""

import subprocess
import sys
import importlib.util
import os

def check_module_installed(module_name):
    """Check if a module is installed."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def check_command_in_path(command):
    """Check if a command is in the system PATH."""
    path = os.environ["PATH"].split(os.pathsep)
    # Add paths for pip-installed scripts
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment
        scripts_dir = os.path.join(sys.prefix, 'Scripts' if sys.platform == 'win32' else 'bin')
        path.append(scripts_dir)
    
    for directory in path:
        cmd_path = os.path.join(directory, command)
        cmd_path_exe = os.path.join(directory, command + '.exe')  # For Windows
        
        if os.path.isfile(cmd_path) and os.access(cmd_path, os.X_OK):
            return cmd_path
        if os.path.isfile(cmd_path_exe) and os.access(cmd_path_exe, os.X_OK):
            return cmd_path_exe
    
    return None

def test_cli_command():
    """Test running the promptpilot CLI command."""
    try:
        result = subprocess.run(['promptpilot', '--version'], 
                               capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return False, str(e)

def main():
    """Main test function."""
    print("Testing PromptPilot CLI Installation\n" + "-" * 35)
    
    # Check if module is installed
    module_installed = check_module_installed('promptpilot')
    print(f"✓ Python package 'promptpilot' is installed: {module_installed}")
    
    # Check for command in PATH
    cmd_path = check_command_in_path('promptpilot')
    if cmd_path:
        print(f"✓ CLI command 'promptpilot' found at: {cmd_path}")
    else:
        print("✗ CLI command 'promptpilot' not found in PATH")
        print("\nSuggestions:")
        print("  1. Make sure you installed with 'pip install -e .' or 'pip install .'")
        print("  2. If using a virtual environment, ensure it's activated")
        print("  3. Verify your PATH includes Python's script directory")
        return
    
    # Test running the command
    success, output = test_cli_command()
    if success:
        print(f"✓ CLI command runs successfully. Version: {output}")
        print("\nCongratulations! The PromptPilot CLI is correctly installed.")
    else:
        print(f"✗ Failed to run CLI command: {output}")
        print("\nTroubleshooting:")
        print("  1. Try running 'which promptpilot' (Unix) or 'where promptpilot' (Windows)")
        print("  2. Check for permission issues on the script file")
        print("  3. Verify all dependencies are installed correctly")

if __name__ == "__main__":
    main() 
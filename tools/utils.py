"""
Utility functions for the image analysis system.
"""

import json
import os
from pathlib import Path

def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_dir):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    # TODO: Implement logging configuration
"""
Main entry point for the Image Analysis Assistant.
"""

import os
from pathlib import Path
from agents.image_analysis import ImageAnalysisAgent
from agents.utils import load_config

def main():
    # Load configurations
    config_dir = Path(__file__).parent / "config"
    api_keys = load_config(config_dir / "api_keys.json")
    model_config = load_config(config_dir / "models.json")
    
    # Initialize agent
    agent = ImageAnalysisAgent(config={
        **api_keys,
        **model_config
    })
    agent.initialize()
    
    # TODO: Add interactive loop or API endpoint here

if __name__ == "__main__":
    main()
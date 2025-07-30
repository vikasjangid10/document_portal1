import yaml


import os

def load_config(config_path: str = None) -> dict:
    # Use forward slashes and absolute path for compatibility
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config_path = os.path.abspath(config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

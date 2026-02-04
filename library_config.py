import json
import os
from pathlib import Path

CONFIG_FILE = "library_config.json"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {"folders": []}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"folders": []}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def add_folder_to_config(folder_path):
    config = load_config()
    folder_path = str(Path(folder_path).resolve())
    if folder_path not in config["folders"]:
        config["folders"].append(folder_path)
        save_config(config)

def remove_folder_from_config(folder_path):
    config = load_config()
    folder_path = str(Path(folder_path).resolve())
    if folder_path in config["folders"]:
        config["folders"].remove(folder_path)
        save_config(config)

def get_folders():
    return load_config().get("folders", [])

import json
import os
import sys
from pathlib import Path

# Get stable user directory
if sys.platform == "win32":
    APP_DATA = Path(os.getenv('LOCALAPPDATA')) / "spice"
else:
    APP_DATA = Path.home() / ".local" / "share" / "spice"

APP_DATA.mkdir(parents=True, exist_ok=True)

# Update paths to use this directory
CONFIG_FILE = APP_DATA / "library_config.json"
DB_PATH = APP_DATA / "spice.lance"

def load_config():
    if not CONFIG_FILE.exists():
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

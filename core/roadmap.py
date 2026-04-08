# core/roadmap.py
import os
import json

def get_state_path(scout_dir):
    return os.path.join(scout_dir, "project_state.json")

def load_project_data(scout_dir):
    """Loads the project roadmap and active tasks."""
    path = get_state_path(scout_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Default Structure if none exists
    return {
        "global_goal": "Build an awesome Godot game.",
        "milestones": [
            {"name": "Core Movement", "status": "completed"},
            {"name": "Inventory System", "status": "in_progress"}
        ],
        "active_tasks": [],
        "architectural_rules": "Use Godot 4.4 static funcs. Keep UI logic decoupled from player logic."
    }

def save_project_data(scout_dir, data):
    """Saves the project roadmap to disk."""
    path = get_state_path(scout_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
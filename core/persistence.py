import json, os

def get_prefs_path(scout_dir):
    return os.path.join(scout_dir, "user_prefs.json")

def save_user_prefs(scout_dir, prefs):
    with open(get_prefs_path(scout_dir), "w") as f:
        json.dump(prefs, f, indent=4)

def load_user_prefs(scout_dir):
    path = get_prefs_path(scout_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # Default fallback
    return {
        "num_ctx": 8192, 
        "max_tokens": 2048, 
        "context_limit": 20,
        "persona": "Senior Architect"
    }
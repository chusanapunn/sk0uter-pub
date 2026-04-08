import gc, pynvml, requests
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_URL

def get_gpu_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3, info.total / 1024**3
    except:
        return 0, 16 # Default for 3080 fallback

def nuke_vram(active_model):
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": active_model, "keep_alive": 0},
                      timeout=5)
    except Exception as e:
        print(f"Ollama release failed: {e}")
    gc.collect()
    return True
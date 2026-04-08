# check_gpu.py
import sys
import os

print(f"Python Version: {sys.version}")
print("Checking for NVIDIA GPU (Lightweight Mode, No Torch)...")

try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"✅ GPU Found: {name}")
    else:
        print("❌ ERROR: No NVIDIA GPUs detected by NVML.")
    pynvml.nvmlShutdown()
except Exception as e:
    print("⚠️ NVML Check failed. Falling back to system SMI...")
    res = os.system("nvidia-smi")
    if res != 0:
        print("❌ CRITICAL: Windows cannot see your NVIDIA GPU (nvidia-smi failed). Check drivers!")
    else:
        print("✅ nvidia-smi executed successfully. GPU is visible to the OS.")
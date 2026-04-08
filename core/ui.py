# core/ui.py
import psutil
import streamlit as st

def show_hardware_stats():
    st.subheader("🖥️ Hardware Monitor")
    
    # 1. CPU & RAM (Upgraded to Gigabytes with 1 decimal point)
    cpu_usage = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / (1024**3)
    ram_total_gb = ram.total / (1024**3)
    
    st.caption(f"CPU: {cpu_usage:.1f}% | RAM: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB")
    st.progress(cpu_usage / 100.0)

    # 2. Grab the clean hardware name we already cached!
    compute_name = st.session_state.get('compute_unit', 'CPU')

    # 3. GPU VRAM (NVIDIA Only)
    if "Apple" in compute_name or "CPU" in compute_name:
        # Mac and CPU-only users share system RAM (Unified Memory)
        st.caption(f"Engine: {compute_name} (Uses System RAM)")
    else:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Upgraded to Gigabytes
            vram_used_gb = info.used / (1024**3)
            vram_total_gb = info.total / (1024**3)
            
            st.caption(f"GPU: {compute_name} | VRAM: {vram_used_gb:.1f}GB / {vram_total_gb:.1f}GB")
            
            # Safe division to prevent ZeroDivisionError
            if info.total > 0:
                st.progress(info.used / info.total)
                
        except Exception:
            st.caption(f"GPU: {compute_name} (VRAM info currently locked)")
            
        finally:
            # CRITICAL: Always shut down NVML to prevent memory leaks in the background
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
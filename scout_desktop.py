# scout_desktop.py
import webview
import threading
import subprocess
import sys
import time
import socket

def get_open_port():
    """Finds a free port so we don't crash if 8501 is taken."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return str(port)

def start_streamlit(port):
    """Runs Streamlit in 'headless' mode so it doesn't open Chrome."""
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py", 
        "--server.port", port, 
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == '__main__':
    print("🚀 Starting Scout Director Engine...")
    
    # 1. Get a safe port and start the backend
    port = get_open_port()
    t = threading.Thread(target=start_streamlit, args=(port,))
    t.daemon = True # Ensures the server dies when the window closes
    t.start()

    # 2. Give the 3080/Torch a moment to load before showing the UI
    time.sleep(3) 

    # 3. Open the Native Window
    print("🖥️ Launching Native Interface...")
    webview.create_window(
        title='Scout Director - Local AI Architect', 
        url=f'http://localhost:{port}', 
        width=1280, 
        height=1080,
        background_color='#1c1f26' # Matches your CSS theme
    )
    webview.start()
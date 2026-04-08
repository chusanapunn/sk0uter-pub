# app.py
import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import os, json, shutil, time
from datetime import datetime

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your custom modules
from config import *
from config import get_model_dim
from core.parser import parse_file, SYSTEM_KEYWORDS
from core.manager import *
from core.persistence import load_user_prefs, save_user_prefs
from core.graph_db import ScoutGraph
from core.roadmap import load_project_data, save_project_data
from utils.telemetry import get_gpu_usage, nuke_vram
from utils.db_ops import wipe_graph_db, nuke_and_reset

from Ask import build_surgical_prompt, ask_local_llm, build_cloud_master_prompt
from core.pipeline_log import PipelineLog
from core.verifier import verify_response, compute_prompt_efficiency
from qdrant_client import QdrantClient
from core.qdrant_client import get_qdrant_client, ensure_collection, create_hybrid_collection as create_hybrid_coll
from config import QDRANT_MODE
from core.ui import *

st.set_page_config(page_title="Scout Director", page_icon="☕", layout="wide")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

LAST_PROJECT_FILE = "last_project.json"

def initialize_project_context(path):
    """The central 'Brain' for switching projects."""
    pid, pdir = ProjectManager.get_project_dirs(path)
    
    # 1. Identity
    st.session_state.active_project = {"id": pid, "path": path, "scout_dir": pdir}
    st.session_state.active_collection = pid
    
    # 2. Chat Recovery
    sessions = get_chat_sessions(pdir)
    st.session_state.active_session = sessions[0] if sessions else "default_session"
    st.session_state.messages = load_chat(pdir, st.session_state.active_session)
    
    # 3. Preferences Recovery
    try:
        prefs = load_user_prefs(pdir)
        st.session_state.ui_num_ctx = prefs.get('num_ctx', 32768)
        st.session_state.ui_max_tokens = prefs.get('max_tokens', 8192)
        st.session_state.ui_context_limit = prefs.get('context_limit', 50)
        st.session_state.ui_detail_threshold = prefs.get('detail_threshold', 50)
        st.session_state.ui_persona = prefs.get('persona', "Senior Architect")
        st.session_state.ui_model_name = prefs.get('llm_model', "llama3")
    except: pass
    
    # 4. Sticky Persistence (Set as Default)
    save_last_project(st.session_state.active_project, st.session_state.active_session)

def save_last_project(project_data, session_id):
    data = project_data.copy()
    data['last_session'] = session_id
    with open(LAST_PROJECT_FILE, "w") as f: # Use the path from config
        json.dump(data, f)

def get_chat_sessions(scout_dir):
    chats_dir = os.path.join(scout_dir, "chats")
    os.makedirs(chats_dir, exist_ok=True)
    return sorted([f.replace(".json", "") for f in os.listdir(chats_dir) if f.endswith(".json")], reverse=True)

def load_chat(scout_dir, session_id):
    path = os.path.join(scout_dir, "chats", f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return []

def save_chat(scout_dir, session_id, messages):
    path = os.path.join(scout_dir, "chats", f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(messages, f, indent=4)

def load_last_project():
    if os.path.exists(LAST_PROJECT_FILE):
        try:
            with open(LAST_PROJECT_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def load_project_to_ui(path, session_id=None):
    """The master 'rehydration' function for the UI."""
    pid, pdir = ProjectManager.get_project_dirs(path)
    st.session_state.active_project = {"id": pid, "path": path, "scout_dir": pdir}
    st.session_state.active_collection = pid
    
    # 1. Resolve Session ID
    sessions = get_chat_sessions(pdir)
    if not session_id or session_id not in sessions:
        st.session_state.active_session = sessions[0] if sessions else "default_session"
    else:
        st.session_state.active_session = session_id

    # 2. Load Messages
    st.session_state.messages = load_chat(pdir, st.session_state.active_session)
    
    # 3. Load Preferences
    try:
        prefs = load_user_prefs(pdir)
        st.session_state.ui_num_ctx = prefs.get('num_ctx', 32768)
        st.session_state.ui_max_tokens = prefs.get('max_tokens', 8192)
        st.session_state.ui_persona = prefs.get('persona', "Senior Architect")
        # st.session_state.ui_model_name = prefs.get('llm_model', "llama3")
    except: pass

# --- UPDATED GLOBAL SESSION STATE ---
if 'active_project' not in st.session_state:
    st.session_state.active_project = None
    
    # Try to load the 'Sticky' project from disk
    saved_data = load_last_project() # From your config.py LAST_PROJECT_FILE
    if saved_data and os.path.exists(saved_data['path']):
        # Deep Load everything: path, session, and messages
        load_project_to_ui(saved_data['path'], saved_data.get('last_session'))
    else:
        # Fallback for brand new users
        st.session_state.active_session = "default_session"
        st.session_state.messages = []

# Safety Check: If active_project exists, ensure its collection is set
if st.session_state.active_project and 'active_collection' not in st.session_state:
    st.session_state.active_collection = st.session_state.active_project['id']
elif 'active_collection' not in st.session_state:
    st.session_state.active_collection = BASE_COLLECTION


import platform

@st.cache_resource
def get_compute_unit_name():
    """Dynamically detects the local hardware without using PyTorch."""
    # 1. Check for NVIDIA GPUs (Windows/Linux)
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            pynvml.nvmlShutdown()
            
            # pynvml returns bytes in some older versions, decode if necessary
            if isinstance(name, bytes):
                name = name.decode('utf-8')
                
            # Clean up the name for the UI (e.g., "NVIDIA GeForce RTX 3080" -> "RTX 3080")
            return name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    except Exception:
        pass # NVML not installed or no NVIDIA GPU found

    # 2. Check for Apple Silicon (macOS)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "Apple Silicon (M-Series)"

    # 3. Fallback
    return "CPU / Generic GPU"

def detect_collection_model(collection_name):
    """Query the live Qdrant collection to find its vector dimension,
    then return the embedding model that matches."""
    from config import MODEL_DIMS
    dim_to_model = {v: k for k, v in MODEL_DIMS.items()}
    try:
        qc = _get_qc()
        info = qc.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            dim = vectors_config["dense"].size
        else:
            dim = vectors_config.size
        qc.close()
        return dim_to_model.get(dim, "nomic-embed-text")
    except Exception:
        return "nomic-embed-text"

# Initialize it into session state
if 'compute_unit' not in st.session_state: 
    st.session_state.compute_unit = get_compute_unit_name()

# --- GLOBAL SESSION STATE ---
if 'qdrant_mode' not in st.session_state: st.session_state.qdrant_mode = QDRANT_MODE
if 'qdrant_host' not in st.session_state: st.session_state.qdrant_host = QDRANT_HOST
if 'qdrant_port' not in st.session_state: st.session_state.qdrant_port = QDRANT_PORT
if 'active_collection' not in st.session_state: st.session_state.active_collection = BASE_COLLECTION
if 'messages' not in st.session_state: st.session_state.messages = []
if 'prompt_history' not in st.session_state: st.session_state.prompt_history = []
if 'terminal_log' not in st.session_state:
    st.session_state.terminal_log = ["[SYSTEM] Scout Director Initialized..."]
if 'active_session' not in st.session_state: st.session_state.active_session = "default_session"

def _get_qc():
    """Get a Qdrant client using current session state settings."""
    scout_dir = st.session_state.active_project.get("scout_dir") if st.session_state.get("active_project") else None
    return get_qdrant_client(
        mode=st.session_state.qdrant_mode,
        scout_dir=scout_dir,
        host=st.session_state.qdrant_host,
        port=st.session_state.qdrant_port,
    )

def apply_custom_theme():
    st.markdown("""
    <style>
        /* Main Background and Fonts */
        .stApp {
            background-color: #1c1f26;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #242933;
            border-right: 1px solid #478CBF33;
        }

        /* Chat Message Styling */
        .stChatMessage {
            background-color: #242933 !important;
            border: 1px solid #3b4252 !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-bottom: 10px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* The "Glow" for Assistant Messages */
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            border-left: 4px solid #478CBF !important;
            background: linear-gradient(90deg, #242933 0%, #2a303c 100%) !important;
        }

        /* Terminal-style Code Blocks */
        code {
            color: #81a1c1 !important;
            background-color: #1c1f26 !important;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1c1f26; }
        ::-webkit-scrollbar-thumb { background: #478CBF; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

def index_batch(batch_name, chunks, embed_model, collection_name, host, port, batch_size=32):
    """Highly optimized Dense-Only indexing with exact schema matching."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    import uuid

    if not chunks: return 0

    texts = [c['content'] for c in chunks]
    
    # 1. Ask Ollama to generate embeddings
    dense_vectors = embed_model.encode(texts)

    # 🛡️ Safety check
    if not dense_vectors or len(dense_vectors) != len(chunks):
        print(f"[WARN] Skipped {batch_name}: Ollama failed to generate embeddings.")
        return 0

    points = []
    for i, chunk in enumerate(chunks):
        semantic_list = dense_vectors[i] 

        # 🛡️ THE FIX: Force forward slashes so Qdrant exact string matching always works!
        raw_file = chunk.get('file', batch_name)
        safe_file = raw_file.replace("\\", "/")
        func_name = chunk.get('name', 'unknown')

        # 🛡️ THE FIX: Deterministic IDs prevent duplicates and allow clean overwriting
        unique_string = f"{safe_file}::{func_name}"
        deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

        points.append(PointStruct(
            id=deterministic_id,
            vector=semantic_list,
            payload={
                "file": safe_file, # Uses the normalized path!
                "content": chunk.get('content', ""),
                "line_start": chunk.get('line_start', 0),
                "type": chunk.get('type', 'logic'),
                "name": func_name,
                # 🛡️ THE FIX: Restored the missing variables for the Inspector UI
                "identity": chunk.get('identity', ''),
                "global_state": chunk.get('global_state', ''),
                "topic": chunk.get('topic', 'Unknown')
            }
        ))

    try:
        qc = _get_qc()
        qc.upsert(collection_name=collection_name, points=points)
        qc.close()
        return len(points)
    except Exception as e:
        print(f"[ERROR] Qdrant Upsert Error in {batch_name}: {e}")
        return 0

def sync_prefs():
    """Gathers all current UI values and saves them to the project folder."""
    if st.session_state.active_project:
        pdir = st.session_state.active_project['scout_dir']
        current_settings = {
            "context_limit": st.session_state.get('ui_context_limit', 20),
            "detail_threshold": st.session_state.get("ui_detail_threshold",5),
            "num_ctx": st.session_state.get('ui_num_ctx', 8192),
            "max_tokens": st.session_state.get('ui_max_tokens', 2048),
            "persona": st.session_state.get('ui_persona', "Senior Architect")
        }
        save_user_prefs(pdir, current_settings)
        

def push_log(message, update_ui=None):
    """Pushes a message to the internal log and optionally updates a UI element."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    st.session_state.terminal_log.append(log_line)
    
    # Keep it from eating too much RAM
    if len(st.session_state.terminal_log) > 100:
        st.session_state.terminal_log.pop(0)
        
    # If a Streamlit container was passed, update it live
    if update_ui:
        log_text = "\n".join(st.session_state.terminal_log)
        update_ui.code(log_text, language="bash")

def select_folder_subprocess():
    """Uses OS-native folder pickers instead of Tkinter (which is missing in Portable Python)."""
    import platform
    import subprocess
    
    if platform.system() == "Windows":
        # THE FIX: We use OpenFileDialog instead of FolderBrowserDialog to get the modern UI.
        # We filter out all files (so only folders show) and pre-fill a dummy file name.
        ps_script = """
        Add-Type -AssemblyName System.Windows.Forms
        $f = New-Object System.Windows.Forms.OpenFileDialog
        $f.Title = 'Select your Project Folder for Scout'
        $f.ValidateNames = $false
        $f.CheckFileExists = $false
        $f.CheckPathExists = $true
        $f.FileName = 'Select Folder - Click Open'
        $f.Filter = 'Folders Only|*.none' 
        
        if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
            # Since we faked a file, we extract just the directory path from the result
            Write-Output ([System.IO.Path]::GetDirectoryName($f.FileName))
        }
        """
        try:
            result = subprocess.run(["powershell", "-Command", ps_script], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            return result.stdout.strip()
        except: return None
        
    elif platform.system() == "Darwin": # macOS
        # macOS already uses the modern Finder dialog by default
        mac_script = 'tell application "Finder" to POSIX path of (choose folder with prompt "Select your Project Folder")'
        try:
            result = subprocess.run(["osascript", "-e", mac_script], capture_output=True, text=True)
            return result.stdout.strip()
        except: return None
        
    return None




apply_custom_theme()
# ==========================================
# SIDEBAR: HUB & CONTROLS
# ==========================================
with st.sidebar:
    st.title("☕ Scout Hub")
    
    # 📂 BROWSE: Immediately loads project into view
    if st.button("📂 Browse Project Folder", use_container_width=True):
        path = select_folder_subprocess()
        if path and os.path.exists(path):
            load_project_to_ui(path)
            st.rerun()

    # ⌨️ MANUAL / LINK
    project = st.session_state.active_project
    current_path = project.get('path', "") if project else ""
    manual_path = st.text_input("Project Path:", value=current_path, placeholder="C:\\Projects\\MyGame")
    
    if manual_path and manual_path != current_path:
        if os.path.exists(manual_path):
            if st.button("🔗 Switch to this Project", type="primary", use_container_width=True):
                load_project_to_ui(manual_path)
                st.rerun()

    # 📌 ACTIVE PROJECT & SESSIONS
    if st.session_state.active_project:
        st.success(f"📍 Viewing: {st.session_state.active_project['id']}")
        
        if st.button("⭐ Set as Startup Default", use_container_width=True):
            save_last_project(st.session_state.active_project, st.session_state.active_session)
            st.toast("Project pinned for next startup!")

        st.divider()
        st.subheader("💬 Chat Sessions")
        pdir = st.session_state.active_project['scout_dir']
        
        new_chat_name = st.text_input("New Session Name", placeholder="e.g., Engine Refactor")
        if st.button("➕ Create Session") and new_chat_name:
            sid = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{new_chat_name.replace(' ', '_')}"
            st.session_state.active_session = sid
            st.session_state.messages = []
            save_chat(pdir, sid, [])
            save_last_project(st.session_state.active_project, sid)
            st.rerun()

        sessions = get_chat_sessions(pdir)
        if sessions:
            current_idx = sessions.index(st.session_state.active_session) if st.session_state.active_session in sessions else 0
            selected_session = st.selectbox("Active Session", options=sessions, index=current_idx)
            if selected_session != st.session_state.active_session:
                st.session_state.active_session = selected_session
                st.session_state.messages = load_chat(pdir, selected_session)
                save_last_project(st.session_state.active_project, selected_session)
                st.rerun()

    st.divider()

    # 🤖 LLM & VRAM SETTINGS
    with st.expander("🤖 LLM Configuration", expanded=True):
        # Embedding-only models cannot generate text — hide them from the LLM picker
        EMBED_ONLY_TAGS = {"nomic-embed", "mxbai-embed", "all-minilm", "bge-", "snowflake-arctic-embed", "embed"}
        try:
            import requests
            res = requests.get(f"{OLLAMA_URL}/api/tags")
            all_models = [m['name'] for m in res.json().get('models', [])]
            ollama_models = [m for m in all_models if not any(tag in m.lower() for tag in EMBED_ONLY_TAGS)]
            if not ollama_models:
                ollama_models = all_models  # fallback: show everything if filter removes all
        except:
            st.warning("Could not fetch models. Is Ollama running?")
            ollama_models = ["llama3"]

        st.selectbox("LLM Model (Text Generation)", options=ollama_models, key="ui_model_name", on_change=sync_prefs)
        st.caption(f"Currently using: `{st.session_state.get('ui_model_name', 'None')}`")

    with st.expander("🧠 VRAM & Engine Settings", expanded=False):
        st.selectbox("AI Archetype", 
                     options=["Senior Architect", "Bug Hunter", "Code Optimizer", "Teacher"], 
                     key="ui_persona", on_change=sync_prefs)
        # Set defaults only if not already loaded from prefs
        if "ui_detail_threshold" not in st.session_state:
            st.session_state.ui_detail_threshold = 50
        if "ui_context_limit" not in st.session_state:
            st.session_state.ui_context_limit = 50
        if "ui_num_ctx" not in st.session_state:
            st.session_state.ui_num_ctx = 32768
        if "ui_max_tokens" not in st.session_state:
            st.session_state.ui_max_tokens = 8192
        st.slider("Detail Threshold (Full Code)", 1, 100, key="ui_detail_threshold", on_change=sync_prefs)
        st.slider("Context Depth", 5, 100, key="ui_context_limit", on_change=sync_prefs)
        st.select_slider("Input Context Window", options=[4096, 8192, 16384, 32768], key="ui_num_ctx", on_change=sync_prefs)
        st.slider("Max Output Tokens", 512, 8192, step=512, key="ui_max_tokens", on_change=sync_prefs)

    st.divider()

    # 📋 PROMPT VAULT (Fixed with Popovers)
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []

    with st.expander("📋 Prompt Vault (Last 5)", expanded=False):
        st.caption("Copy surgical prompts for external Cloud AIs.")
        if not st.session_state.prompt_history:
            st.info("No prompts generated yet.")
        else:
            for i, p_data in enumerate(st.session_state.prompt_history):
                v_col1, v_col2 = st.columns([0.7, 0.3])
                v_col1.markdown(f"**{i+1}.** `{p_data['query'][:35]}...`")
                with v_col2:
                    with st.popover("📄 View"):
                        st.markdown("### Surgical Prompt")
                        st.code(p_data['prompt'], language="markdown")
                if i < len(st.session_state.prompt_history) - 1:
                    st.divider()

    # 📊 HARDWARE
    with st.expander("📊 System & Hardware", expanded=True):
        used, total = get_gpu_usage()
        compute_unit = st.session_state.get('compute_unit', 'GPU')
        st.metric(f"{compute_unit} VRAM", f"{used:.1f}GB / {total:.1f}GB")
        if total > 0: st.progress(min(used/total, 1.0))
        if st.button("🌙 Purge VRAM", use_container_width=True):
            nuke_vram(st.session_state.get('ui_model_name', LLM_MODEL))
            st.rerun()

# ==========================================
# MAIN INTERFACE TABS
# ==========================================
tab_chat, tab_index, tab_qdrant, tab_graph, tab_roadmap,tab_export = st.tabs([
    "💬 Terminal", "⚙️ Indexer", "🗄️ Vector DB", "🕸️ Graph Map", "🗺️ Roadmap", "☁️ Cloud Export"
])# --- TAB 1: CHAT ---
with tab_chat:
    if not st.session_state.active_project: st.info("👋 Select a project folder in the sidebar to begin.")
    else:
        # Auto-detect the correct embedding model from the live Qdrant collection.
        # This is the ONLY reliable source — the saved project config can be stale
        # if the user changed the dropdown without re-indexing.
        active_model_path = detect_collection_model(
            st.session_state.active_collection,
        )
        st.caption(f"🔗 Collection `{st.session_state.active_collection}` → **{active_model_path}** ({get_model_dim(active_model_path)}d)")

        st.subheader(f"Session: `{st.session_state.active_session}`")
        chat_container = st.container(height=600, border=False)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if query := st.chat_input("Analyze architecture or generate code..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with chat_container:
                with st.chat_message("user"): st.markdown(query)
            with chat_container:
                with st.chat_message("assistant"):
                    embed_model = get_embedder(active_model_path)
                    
                    with st.spinner("Compiling Architectural Prompt..."):

                        _pipeline_log = PipelineLog()
                        surgical_prompt, _prompt_meta = build_surgical_prompt(
                                query=query,
                                active_project=st.session_state.active_project,
                                collection_name=st.session_state.active_collection,
                                embed_model=embed_model,
                                host=st.session_state.qdrant_host,
                                port=st.session_state.qdrant_port,
                                context_limit=st.session_state.get('ui_context_limit', 50),
                                detail_threshold=st.session_state.get('ui_detail_threshold', 50),
                                persona_key=st.session_state.get('ui_persona', 'Senior Architect'),
                                pipeline_log=_pipeline_log,
                                qdrant_mode=st.session_state.qdrant_mode,
                            )

                        st.session_state.prompt_history.insert(0, {"query": query, "prompt": surgical_prompt})
                        st.session_state.prompt_history = st.session_state.prompt_history[:5]

                        with st.expander("☁️ Cloud AI Prompt", expanded=False):
                            st.code(surgical_prompt, language="markdown")
                        if _pipeline_log.has_issues:
                            with st.expander(f"⚠️ {_pipeline_log.summary()}", expanded=False):
                                st.code(_pipeline_log.full_report(), language="bash")
                    response_placeholder = st.empty()
                    with st.spinner(f"Local {st.session_state.compute_unit} is thinking..."):
                        prior_history = st.session_state.messages[:-1]
                        response = ask_local_llm(
                            surgical_prompt,
                            model_name=st.session_state.ui_model_name,
                            max_tokens=st.session_state.ui_max_tokens,
                            num_ctx=st.session_state.ui_num_ctx,
                            chat_history=prior_history,
                        )
                        response_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    # --- Phase 3: Hallucination Verification ---
                    if not response.startswith("❌"):
                        try:
                            from core.graph_db import ScoutGraph
                            _vg = ScoutGraph(
                                st.session_state.active_project["id"],
                                st.session_state.active_project["scout_dir"],
                                st.session_state.active_collection,
                                read_only=True,
                            )
                            _vresult = verify_response(
                                response, _vg,
                                context_files=_prompt_meta.get("context_files"),
                                pipeline_log=_pipeline_log,
                            )
                            _vg.close()

                            # Prompt efficiency metrics
                            _efficiency = compute_prompt_efficiency(
                                surgical_prompt, response,
                                token_budget=_prompt_meta.get("token_budget", 24000),
                            )

                            # Display grounding score inline
                            _gscore = _vresult.grounding_score
                            if _gscore >= 0.9:
                                _icon, _color = "✅", "green"
                            elif _gscore >= 0.7:
                                _icon, _color = "⚠️", "orange"
                            else:
                                _icon, _color = "🔴", "red"
                            st.caption(f"{_icon} {_vresult.summary()}")

                            with st.expander("📊 Verification Report", expanded=False):
                                st.code(_vresult.full_report(), language="text")
                                _tused = _prompt_meta.get("tokens_used", 0)
                                _tbudget = _prompt_meta.get("token_budget", 24000)
                                st.markdown("**Prompt Efficiency**")
                                st.markdown(
                                    f"- Context budget: ~{_tused:,} / {_tbudget:,} tokens ({_tused/_tbudget:.0%} used)\n"
                                    f"- Budget utilization: {_efficiency['budget_utilization']:.0%}\n"
                                    f"- Response density: {_efficiency['response_density']:.4f} entities/token\n"
                                    f"- Unique entities cited: {_efficiency['unique_entities_cited']}\n"
                                    f"- Prompt tokens (est): {_efficiency['prompt_tokens_est']:,}\n"
                                    f"- Response tokens (est): {_efficiency['response_tokens_est']:,}"
                                )
                        except Exception as _ve:
                            st.caption(f"⚠️ Verification skipped: {_ve}")

            save_chat(st.session_state.active_project['scout_dir'], st.session_state.active_session, st.session_state.messages)

# --- TAB 2: INDEXER & MANAGEMENT ---

with tab_index:
    project = st.session_state.active_project

    
    if not project:
        st.info("👋 Select a project folder in the sidebar to begin.")
    else:
        # User Controls
        col1, col2, col3 = st.columns(3)
        with col1: b_size = st.slider("Batch Size", 64, 1024, 256, step=64)
        with col2: workers = st.slider("Threads", 1, 16, 4)
        
        with col3:
            selected_label = st.selectbox("Embedding Engine", list(SUPPORTED_MODELS.keys()))
            actual_model_name = SUPPORTED_MODELS[selected_label]

        # --- Proactive dimension mismatch check (runs on every render, not inside Run handler) ---
        _sel_dim = get_model_dim(actual_model_name)
        try:
            _qc_probe = _get_qc()
            _ci_probe = _qc_probe.get_collection(st.session_state.active_collection)
            _vc = _ci_probe.config.params.vectors
            _existing_dim = _vc["dense"].size if isinstance(_vc, dict) else getattr(_vc, 'size', None)
            if _existing_dim and _existing_dim != _sel_dim:
                st.error(
                    f"⚠️ Dimension mismatch: collection **'{st.session_state.active_collection}'** is "
                    f"**{_existing_dim}d** but **'{actual_model_name}'** outputs **{_sel_dim}d**. "
                    f"Reset the DB before indexing."
                )
                if st.button("🔥 Fix: Nuke & Reset to match model", use_container_width=True, key="mismatch_nuke_persistent"):
                    with st.spinner(f"Rebuilding for {_sel_dim}d..."):
                        graph_ok, vector_ok, dim, errors = nuke_and_reset(
                            project['scout_dir'],
                            st.session_state.active_collection,
                            actual_model_name,
                            qdrant_mode=st.session_state.qdrant_mode,
                            host=st.session_state.qdrant_host,
                            port=st.session_state.qdrant_port,
                        )
                        if graph_ok: st.success("✅ Graph DB wiped.")
                        if vector_ok: st.success(f"✅ Vector DB rebuilt ({dim}d)")
                        for _err in errors: st.error(_err)
                        if graph_ok and vector_ok: st.rerun()
        except Exception:
            pass  # Collection doesn't exist yet — fine

        if st.button("⚡ Run Indexing", type="primary", use_container_width=True):
            scout_dir = project['scout_dir']
            project_path = project['path']
            project_id = project['id']
            active_coll = st.session_state.active_collection
            q_host = st.session_state.qdrant_host
            q_port = st.session_state.qdrant_port

            os.makedirs(scout_dir, exist_ok=True)
            
            # Load existing roadmap data so we don't wipe it
            current_project_data = load_project_data(scout_dir)
            current_project_data['embedding_model'] = actual_model_name
            save_project_data(scout_dir, current_project_data) 
            st.session_state.active_project['embedding_model'] = actual_model_name

            try:
                with st.status("Indexing...", expanded=True) as status:
                    terminal_ui = st.empty()

                    # Clear the log UI at the start of a new run
                    terminal_ui.empty()
                    if 'log_history' in st.session_state:
                        st.session_state.log_history = []

                    # Ensure collection exists before indexing
                    _qc_pre = _get_qc()
                    selected_dim = get_model_dim(actual_model_name)
                    if ensure_collection(_qc_pre, active_coll, selected_dim):
                        push_log(f"Created collection '{active_coll}' ({selected_dim}d)", terminal_ui)
                    _qc_pre.close()

                    embed_model = get_embedder(actual_model_name)
                    ok, msg = embed_model.ensure_model(
                        progress_cb=lambda s: push_log(f"Ollama: {s}", terminal_ui)
                    )
                    if not ok:
                        st.error(f"Ollama pre-flight failed: {msg}")
                        st.stop()
                    push_log(f"Ollama: {msg}", terminal_ui)

                    # Dimension mismatch guard: abort if the collection was built with a
                    # different embedding dimension. The fix button lives ABOVE the run button.
                    selected_dim = get_model_dim(actual_model_name)
                    try:
                        _qc_check = _get_qc()
                        _coll_info = _qc_check.get_collection(active_coll)
                        _vc2 = _coll_info.config.params.vectors
                        _existing_dim = _vc2["dense"].size if isinstance(_vc2, dict) else getattr(_vc2, 'size', None)
                        if _existing_dim and _existing_dim != selected_dim:
                            st.error(
                                f"⚠️ Dimension mismatch: '{active_coll}' is {_existing_dim}d but "
                                f"'{actual_model_name}' outputs {selected_dim}d. "
                                f"Use the 'Fix: Nuke & Reset' button above the Run button."
                            )
                            st.stop()
                    except Exception:
                        # Collection doesn't exist yet — fine, it will be created on first upsert.
                        pass

                    graph = ScoutGraph(project_id, scout_dir, active_coll)
                    
                    # 2. Gather Files (skip non-project directories)
                    _SKIP_DIRS = {".scout", ".git", ".claude", "python_env", "node_modules",
                                  "__pycache__", ".godot", ".import", "venv", ".venv"}
                    files = []
                    for r, d, fs in os.walk(project_path):
                        d[:] = [sub for sub in d if sub not in _SKIP_DIRS]
                        for f in fs:
                            if f.endswith(('.gd', '.md', '.tscn', '.json')):
                                files.append(os.path.join(r, f))
                    
                    all_chunks = []
                    progress = st.progress(0)

                    def process_file_threadsafe(fpath, base_path):
                        rel = os.path.relpath(fpath, base_path)
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f: 
                            code = f.read()
                        return rel, parse_file(rel, code)

                    # ==========================================
                    # PHASE 1A: CONCURRENT DISK I/O & PARSING
                    # ==========================================
                    push_log("Phase 1: Parsing files...", terminal_ui)

                    parsed_results = []  # (rel_path, parsed)
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(process_file_threadsafe, f, project_path): f for f in files}
                        for i, future in enumerate(concurrent.futures.as_completed(futures)):
                            try:
                                rel_path, parsed = future.result()
                                if parsed and isinstance(parsed, dict) and 'chunks' in parsed:
                                    parsed_results.append((rel_path, parsed))
                            except Exception as file_error:
                                push_log(f"[ERROR] Error parsing file: {file_error}", terminal_ui)
                            progress.progress((i + 1) / len(files))

                    # ==========================================
                    # PHASE 1B: BULK GRAPH SYNC + CHUNK GATHER
                    # ==========================================
                    push_log(f"Building graph for {len(parsed_results)} files...", terminal_ui)
                    progress.progress(0)

                    # Collect parsed_map for bulk load + gather chunks
                    parsed_map = {}
                    for i, (rel_path, parsed) in enumerate(parsed_results):
                        safe_file = rel_path.replace("\\", "/")
                        parsed_map[safe_file] = parsed  # forward-slash key so Kuzu + Qdrant paths match
                        identity = parsed.get('identity', '')
                        global_state = parsed.get('global_state', '')
                        topic = parsed.get('topic', 'Unknown')
                        for c in parsed['chunks']:
                            c['file'] = safe_file
                            c['identity'] = identity
                            c['global_state'] = global_state
                            c['topic'] = topic
                            all_chunks.append(c)
                        progress.progress((i + 1) / len(parsed_results))

                    # Bulk load entire graph via CSV COPY FROM
                    try:
                        graph.bulk_sync(parsed_map)
                        push_log("Graph built.", terminal_ui)

                        # Propagate system tags through signal chains in the graph
                        enriched_systems = graph.propagate_systems(parsed_map)
                        for chunk in all_chunks:
                            fp = chunk.get('file', '')
                            if fp in enriched_systems:
                                existing = set(chunk.get('systems', []))
                                existing.update(enriched_systems[fp])
                                chunk['systems'] = sorted(existing)
                        push_log(f"System tags propagated via signal chains.", terminal_ui)
                    except Exception as graph_err:
                        push_log(f"[WARN] Graph build failed: {graph_err}. Continuing with vector indexing...", terminal_ui)

                    graph.close() # Graph is done, unlock the database!

                    # ==========================================
                    # PHASE 2: EMBED ALL + VECTOR DB UPSERT
                    # ==========================================
                    if all_chunks:
                        import uuid
                        import json as _json
                        from qdrant_client import QdrantClient
                        from qdrant_client.models import PointStruct, SparseVector
                        from core.sparse import build_bm25_vocab, encode_sparse

                        texts = [c['content'] for c in all_chunks]

                        # Step 1a: Dense embeddings via Ollama
                        push_log(f"Phase 2a: Generating dense embeddings for {len(all_chunks)} chunks...", terminal_ui)
                        progress.progress(0)
                        dense_vectors = embed_model.encode(texts)

                        if not dense_vectors or len(dense_vectors) != len(all_chunks):
                            push_log(f"[ERROR] Embedding failed: got {len(dense_vectors) if dense_vectors else 0} vectors for {len(all_chunks)} chunks.", terminal_ui)
                        else:
                            # Step 1b: BM25 sparse vectors (CPU-only, fast)
                            push_log(f"Phase 2b: Building BM25 sparse index...", terminal_ui)
                            bm25_vocab = build_bm25_vocab(texts)
                            sparse_vectors = [encode_sparse(t, bm25_vocab) for t in texts]

                            # Save BM25 vocab to disk so query-time can load it
                            vocab_path = os.path.join(scout_dir, "bm25_vocab.json")
                            with open(vocab_path, "w", encoding="utf-8") as _vf:
                                _json.dump({
                                    "idf": bm25_vocab["idf"],
                                    "avgdl": bm25_vocab["avgdl"],
                                    "k1": bm25_vocab["k1"],
                                    "b": bm25_vocab["b"],
                                }, _vf)
                            push_log(f"BM25 vocab saved ({len(bm25_vocab['idf'])} tokens).", terminal_ui)

                            progress.progress(0.5)
                            push_log(f"Embeddings ready. Uploading to Qdrant...", terminal_ui)

                            # Step 2: Build points with named dense + sparse vectors
                            points = []
                            for i, chunk in enumerate(all_chunks):
                                raw_file = chunk.get('file', 'unknown')
                                safe_file = raw_file.replace("\\", "/")
                                func_name = chunk.get('name', 'unknown')
                                det_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{safe_file}::{func_name}"))

                                sp_indices, sp_values = sparse_vectors[i]
                                point_vectors = {"dense": dense_vectors[i]}
                                if sp_indices:
                                    point_vectors["bm25"] = SparseVector(
                                        indices=sp_indices, values=sp_values
                                    )

                                # Auto-assign systems: use parser result, or infer from
                                # combined content + global_state + topic if empty
                                chunk_systems = chunk.get('systems', [])
                                if not chunk_systems:
                                    from core.parser import infer_systems
                                    combined = chunk.get('content', '') + " " + chunk.get('global_state', '') + " " + chunk.get('topic', '')
                                    chunk_systems = infer_systems(combined, chunk.get('topic', ''))

                                points.append(PointStruct(
                                    id=det_id,
                                    vector=point_vectors,
                                    payload={
                                        "file": safe_file,
                                        "content": chunk.get('content', ""),
                                        "line_start": chunk.get('line_start', 0),
                                        "type": chunk.get('type', 'logic'),
                                        "name": func_name,
                                        "identity": chunk.get('identity', ''),
                                        "global_state": chunk.get('global_state', ''),
                                        "topic": chunk.get('topic', 'Unknown'),
                                        "systems": chunk_systems,
                                    }
                                ))

                            # Step 3: Upsert in batches to Qdrant
                            qc = _get_qc()
                            total_saved = 0
                            qdrant_batches = [points[i:i + b_size] for i in range(0, len(points), b_size)]
                            for idx, batch in enumerate(qdrant_batches):
                                try:
                                    qc.upsert(collection_name=active_coll, points=batch)
                                    total_saved += len(batch)
                                    push_log(f"Injected batch {idx+1}/{len(qdrant_batches)} ({len(batch)} vectors)", terminal_ui)
                                except Exception as e:
                                    push_log(f"[ERROR] Qdrant upsert failed: {e}", terminal_ui)
                                progress.progress((idx + 1) / len(qdrant_batches))
                            push_log(f"Total: {total_saved} vectors (dense+sparse) stored.", terminal_ui)

                    status.update(label=f"✅ Indexing Complete! Processed {len(all_chunks)} chunks.", state="complete")
                    st.success(f"Project '{project_id}' is ready.")
            except Exception as e:
                st.error(f"Critical Failure: {e}")


        st.divider()
        st.subheader("⚠️ Database Management")
        st.markdown("Clear old data when updating parsers or switching code branches. **Requires re-indexing afterward.**")
        
        st.divider()
        with st.expander("🧨 Factory Reset Databases", expanded=False):
            st.warning("This will permanently delete all indexed Vectors and Graph relationships for this project.")
            
            target_dim = get_model_dim(actual_model_name)
            st.caption(f"Will reset to **{actual_model_name}** ({target_dim}d)")

            if st.button("🔥 Nuke & Reset Project Context", use_container_width=True):
                with st.spinner(f"Rebuilding for {target_dim}d..."):
                    graph_ok, vector_ok, dim, errors = nuke_and_reset(
                        st.session_state.active_project['scout_dir'],
                        st.session_state.active_collection,
                        actual_model_name,
                        qdrant_mode=st.session_state.qdrant_mode,
                        host=st.session_state.qdrant_host,
                        port=st.session_state.qdrant_port,
                    )
                    if graph_ok:
                        st.success("✅ Graph DB wiped.")
                    if vector_ok:
                        st.success(f"✅ Qdrant rebuilt for {actual_model_name} ({dim}d)")
                        st.session_state.active_project['embedding_model'] = actual_model_name
                    for err in errors:
                        st.error(err)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧨 Wipe Graph DB (Kùzu)", use_container_width=True):
                    try:
                        wipe_graph_db(st.session_state.active_project['scout_dir'], st.session_state.active_collection)
                        st.success("Graph wiped!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

            with col2:
                if st.button("🧨 Wipe Vector DB (Qdrant)", use_container_width=True):
                    try:
                        from core.qdrant_client import wipe_vector_db as _wipe_vdb
                        _wipe_qc = _get_qc()
                        dim = _wipe_vdb(_wipe_qc, st.session_state.active_collection, get_model_dim(actual_model_name))
                        _wipe_qc.close()
                        st.success(f"Wiped and reset to {actual_model_name} ({dim}d)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        show_hardware_stats()
        

# --- TAB 3: GRAPH MAP ---
with tab_graph:
    if not st.session_state.active_project: st.info("👋 Select a project folder in the sidebar to begin.")
    else:
        safe_name = "".join([c for c in st.session_state.active_collection if c.isalnum() or c in ('_', '-')])
        db_path = os.path.join(st.session_state.active_project['scout_dir'], f"graph_{safe_name}")
        
        # DEBUG: This will print the exact path to the UI so you can verify it matches your folder
        st.caption(f"🔍 Checking Graph Path: `{db_path}`")
        
        # THE FIX: Removed the os.listdir() check. Just verify the folder exists.
        if not os.path.exists(db_path):
            st.warning(f"🕸️ Graph Database '{st.session_state.active_collection}' does not exist yet. Please run the Indexer in Tab 2 to build the map.")
        else:
            st.markdown("""<style>iframe[title*="agraph"]{height:920px !important; min-height:920px !important; border-radius:8px; border:1px solid #333;} div[data-testid="stCustomComponentV1"]:has(iframe[title*="agraph"]){height:920px !important; min-height:920px !important; border-radius:10px; border:1px solid rgba(128,128,128,0.2); background-color:transparent; background-image:radial-gradient(circle, rgba(128,128,128,0.4) 1px, transparent 1px); background-size:25px 25px; position:absolute;}</style>""", unsafe_allow_html=True)
            
            head_col, space_col, btn_col, set_col = st.columns([0.6, 0.2, 0.1, 0.1])
            with head_col: st.subheader("🕸️ Architecture Map")
            with btn_col:
                if st.button("🔄", help="Refresh map", use_container_width=True): st.rerun()
            with set_col:
                with st.popover("⚙️", help="Map Settings", use_container_width=True):
                    layout_style = st.radio("Layout", ["Physics", "Tree"], label_visibility="collapsed")
                    node_limit = st.slider("Max Nodes", 50, 1500, 300, 50)
                    zoom_speed = st.slider("🖱️ Scroll Sensitivity", 0.01, 0.5, 0.17, 0.01) 
                    search_filter = st.text_input("🔍 Filter", placeholder="player.gd")

            st.divider()
            map_col, inspector_col = st.columns([0.75, 0.25], gap="large")

            with map_col:
                with st.spinner("Compiling Graph..."):
                    nodes, edges, added_nodes = [], [], set()
                    
                    try:
                        graph_db = ScoutGraph(st.session_state.active_project['id'], st.session_state.active_project['scout_dir'], st.session_state.active_collection, read_only=True)                    
                        
                        def add_node(n_id, n_label, n_title, n_size, n_color, n_shape):
                            if n_id not in added_nodes:
                                nodes.append(Node(id=n_id, label=n_label, title=n_title, size=n_size, color=n_color, shape=n_shape, font={"color": "white", "size": 14, "face": "monospace"}, shadow={"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 5}))
                                added_nodes.add(n_id)

                        edge_font = {"color": "#000000", "size": 8, "align": "middle", "strokeWidth": 3, "strokeColor": "#ffffff"}

                        # QUERY 1: ALL Scripts
                        q1 = "MATCH (s:Script) "
                        params = {}
                        if search_filter:
                            q1 += "WHERE s.path CONTAINS $search_str "
                            params['search_str'] = search_filter
                        q1 += f"RETURN s.path LIMIT {node_limit}"
                        res1 = graph_db.conn.execute(q1, params)
                        while res1.has_next():
                            script_path = res1.get_next()[0]
                            add_node(script_path, os.path.basename(script_path), script_path, 30, "#478CBF", "box")

                        # QUERY 2: Topics
                        res_t = graph_db.conn.execute(f"MATCH (s:Script)-[:IMPLEMENTS]->(t:Topic) RETURN s.path, t.name LIMIT 500")
                        while res_t.has_next():
                            script_path, topic_name = res_t.get_next()
                            if script_path in added_nodes:
                                topic_id = f"TOPIC::{topic_name}"
                                add_node(topic_id, topic_name, f"Topic: {topic_name}", 25, "#2ECC71", "hexagon")
                                edges.append(Edge(source=script_path, target=topic_id, label="IMPLEMENTS", color="#2ECC71", width=2.0, font=edge_font))

                        # QUERY 3: Functions
                        res2 = graph_db.conn.execute(f"MATCH (s:Script)-[:OWNS]->(f:Function) RETURN s.path, f.id, f.name LIMIT {node_limit}")
                        while res2.has_next():
                            script_path, func_id, func_name = res2.get_next()
                            if script_path in added_nodes:
                                is_built_in = func_name.startswith('_')
                                f_color = "#9B59B6" if is_built_in else "#E0E0E0" 
                                add_node(func_id, func_name, f"Function: {func_name}", 15, f_color, "dot")
                                edges.append(Edge(source=script_path, target=func_id, label="OWNS", color="#888888", width=1.5, font=edge_font))

                        # QUERY 4: Fired Signals
                        res3 = graph_db.conn.execute("MATCH (f:Function)-[:FIRES]->(sig:Signal) RETURN f.id, sig.name LIMIT 500")
                        while res3.has_next():
                            f_id, sig_name = res3.get_next()
                            if f_id in added_nodes: 
                                sig_id = f"SIG::{sig_name}"
                                add_node(sig_id, sig_name, f"Signal: {sig_name}", 25, "#F1C40F", "diamond")
                                edges.append(Edge(source=f_id, target=sig_id, label="FIRES", color="#F1C40F", width=2.5, font=edge_font))

                        # QUERY 5: Documents (.md, .json)
                        res_doc = graph_db.conn.execute("MATCH (d:Document)-[:CONTAINS_DOC]->(c:DataChunk) RETURN d.path, c.id, c.name LIMIT 200")
                        while res_doc.has_next():
                            doc_path, chunk_id, chunk_name = res_doc.get_next()
                            add_node(doc_path, os.path.basename(doc_path), doc_path, 30, "#E67E22", "box")
                            add_node(chunk_id, chunk_name, f"Chunk: {chunk_name}", 15, "#F39C12", "dot")
                            edges.append(Edge(source=doc_path, target=chunk_id, label="CONTAINS_DOC", color="#F39C12", width=1.5, font=edge_font))

                        # QUERY 6: Scenes (.tscn)
                        res_scn = graph_db.conn.execute("MATCH (sc:Scene)-[:CONTAINS_SCENE]->(c:DataChunk) RETURN sc.path, c.id, c.name LIMIT 200")
                        while res_scn.has_next():
                            scn_path, chunk_id, chunk_name = res_scn.get_next()
                            add_node(scn_path, os.path.basename(scn_path), scn_path, 30, "#E74C3C", "box")
                            add_node(chunk_id, "Hierarchy", f"Scene Data", 15, "#FD79A8", "dot")
                            edges.append(Edge(source=scn_path, target=chunk_id, label="CONTAINS_SCENE", color="#FD79A8", width=1.5, font=edge_font))

                        graph_db.close()
                        
                    except Exception as e: 
                        st.error(f"Graph error: {e}")
                        try: graph_db.close()
                        except: pass

                    selected_node = None
                    if nodes:
                        is_hierarchical = (layout_style == "Tree")
                        # Tweaked config slightly to ensure edge labels don't overlap too much
                        config = Config(width="100%", height=920, directed=True, physics=not is_hierarchical, hierarchical=is_hierarchical, nodeHighlightBehavior=True, highlightColor="#FF4B4B", interaction={"navigationButtons": False, "keyboard": True, "hover": False, "zoomView": True, "dragView": True, "selectConnectedEdges": True, "zoomSpeed": zoom_speed}, physics_layout={"solver": "barnesHut", "barnesHut": {"gravitationalConstant": -2000, "springLength": 150}})
                        selected_node = agraph(nodes=nodes, edges=edges, config=config)
                    else: 
                        st.info("No matching data found.")

            with inspector_col:
                st.subheader("🔍 Node Inspector")
                if not selected_node: 
                    st.info("👈 Click on a node to inspect source code.")
                else:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue

                    # Identify the type of node clicked
                    is_signal_or_topic = selected_node.startswith("SIG::") or selected_node.startswith("TOPIC::")
                    
                    if is_signal_or_topic:
                        # Signals and Topics are conceptual, they don't have code bodies in Qdrant
                        st.markdown(f"**Concept Node:** `{os.path.basename(selected_node)}`")
                        st.caption("This is an architectural link, not a direct file.")
                    
                    else:
                        # 1. Extract Target Identifiers
                        if "::" in selected_node:
                            # It's a Function, Chunk, or Scene Data node
                            script_name, func_name = selected_node.split("::", 1)
                            st.markdown(f"**Target:** `{func_name}`")
                            st.caption(f"**File:** `{script_name}`")

                            # Query Qdrant for this specific function/chunk
                            try:
                                local_q_client = _get_qc()
                                safe_script = script_name.replace("\\", "/")
                                hits = local_q_client.scroll(
                                    collection_name=st.session_state.active_collection,
                                    scroll_filter=Filter(must=[
                                        FieldCondition(key="file", match=MatchValue(value=safe_script)),
                                        FieldCondition(key="name", match=MatchValue(value=func_name))
                                    ]), limit=1
                                )[0]

                                if hits:
                                    point_id = hits[0].id
                                    payload = hits[0].payload

                                    # Metadata badges (compact)
                                    chunk_type = payload.get('type', 'unknown')
                                    topic = payload.get('topic', 'Unknown')
                                    line_start = payload.get('line_start', 0)
                                    st.markdown(
                                        f'<div style="display:flex;gap:12px;font-size:12px;margin:4px 0 8px 0;">'
                                        f'<span style="color:#888;">Type: <b style="color:#fff;">{chunk_type}</b></span>'
                                        f'<span style="color:#888;">Line: <b style="color:#fff;">{line_start}</b></span>'
                                        f'<span style="color:#888;">Topic: <b style="color:#fff;">{topic}</b></span>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                                    # System tags — always shown, always editable
                                    current_systems = payload.get('systems', [])
                                    _sys_key = f"sys_{selected_node.replace(':','_').replace('/','_')}"
                                    new_systems = st.multiselect(
                                        "Systems",
                                        options=sorted(SYSTEM_KEYWORDS.keys()),
                                        default=current_systems,
                                        key=_sys_key,
                                        help="Assign this function to one or more game systems. Controls how the AI groups context.",
                                    )
                                    if st.button("💾 Save Systems", key=f"save_{_sys_key}", use_container_width=True):
                                        try:
                                            local_q_client.set_payload(
                                                collection_name=st.session_state.active_collection,
                                                payload={"systems": new_systems},
                                                points=[point_id],
                                            )
                                            st.success("Systems saved.")
                                            st.rerun()
                                        except Exception as _se:
                                            st.error(f"Save failed: {_se}")

                                    # Code content
                                    content = payload.get('content', '')
                                    if content.strip():
                                        lang = 'gdscript' if script_name.endswith('.gd') else 'markdown'
                                        with st.container(height=300, border=True):
                                            st.code(content, language=lang)

                                    # Global state / variables (if present)
                                    g_state = payload.get('global_state', '')
                                    if g_state and g_state.strip():
                                        with st.expander("Class State / Variables", expanded=False):
                                            st.code(g_state, language='gdscript')

                                    # Identity / inheritance
                                    identity = payload.get('identity', '')
                                    if identity and identity.strip():
                                        with st.expander("Inheritance / Identity", expanded=False):
                                            st.code(identity, language='gdscript')
                                else:
                                    st.warning(f"No vector data found for `{func_name}` in `{script_name}`.\nThis chunk may not have been indexed yet.")
                            except Exception as e:
                                st.error(f"Qdrant query failed: {e}")

                        else:
                            # It's a full Script or Document Node
                            script_name = selected_node
                            st.markdown(f"**File:** `{os.path.basename(script_name)}`")
                            st.caption(f"**Path:** `{script_name}`")
                            
                            # 1. READ THE FULL FILE FROM DISK
                            full_path = os.path.join(st.session_state.active_project['path'], script_name)
                            try:
                                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                    full_code = f.read()
                                
                                st.markdown("**Full Source Code:**")
                                # Use st.container to make it scrollable if it's massive
                                with st.container(height=400, border=True):
                                    st.code(full_code, language='gdscript' if script_name.endswith('.gd') else 'markdown')
                            except Exception as e:
                                st.warning(f"Could not read local file: {e}")

                            # 2. GRAB ARCHITECTURAL SUMMARY FROM QDRANT
                            try:
                                local_q_client = _get_qc()
                                safe_script = script_name.replace("\\", "/")
                                
                                # We just need ONE chunk from this file to grab the global_state/identity
                                hits = local_q_client.scroll(
                                    collection_name=st.session_state.active_collection, 
                                    scroll_filter=Filter(must=[
                                        FieldCondition(key="file", match=MatchValue(value=safe_script))
                                    ]), limit=1
                                )[0]
                                
                                if hits:
                                    payload = hits[0].payload
                                    
                                    # Global State (Constants, Enums, Variables)
                                    g_state = payload.get('global_state', '')
                                    if g_state and g_state.strip():
                                        with st.expander("Show Class State / Variables", expanded=True):
                                            st.code(g_state, language='gdscript')
                                            
                                    # Identity (Parent class info)
                                    identity = payload.get('identity', '')
                                    if identity and identity.strip():
                                        with st.expander("Show Inheritance / Identity"):
                                            st.code(identity, language='gdscript')
                            except Exception as e:
                                pass

        # -------------------------------------------------------
        # SYSTEM CLUSTER MAP — Overlapping circles per game system
        # -------------------------------------------------------
        with st.expander("🔮 System Cluster Map", expanded=False):
            st.caption("Scripts grouped by inferred game system. Circles overlap for multi-system scripts.")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                from matplotlib.patches import Circle
                import networkx as nx
                import numpy as np

                _cluster_qc = _get_qc()

                # Scroll ALL points, collect file→systems
                _file_systems = {}
                _offset = None
                while True:
                    _batch, _next = _cluster_qc.scroll(
                        collection_name=st.session_state.active_collection,
                        with_payload=["file", "systems"],
                        with_vectors=False,
                        limit=500,
                        offset=_offset,
                    )
                    for _pt in _batch:
                        _f = _pt.payload.get("file", "")
                        _sys = _pt.payload.get("systems", [])
                        if _f:
                            _file_systems.setdefault(_f, set()).update(_sys)
                    if _next is None:
                        break
                    _offset = _next

                _scripts = [f for f, s in _file_systems.items() if s]  # only files that have systems
                _all_systems = sorted({s for sys_set in _file_systems.values() for s in sys_set})

                if not _scripts or not _all_systems:
                    st.info("No system tags found. Re-index with the updated parser, or manually assign systems via the Node Inspector.")
                else:
                    # Build graph: scripts connected by shared systems (for layout)
                    _G = nx.Graph()
                    _G.add_nodes_from(_scripts)
                    for _i, _s1 in enumerate(_scripts):
                        for _s2 in _scripts[_i+1:]:
                            if _file_systems[_s1] & _file_systems[_s2]:
                                _G.add_edge(_s1, _s2)

                    _pos = nx.spring_layout(_G, k=2.5, iterations=80, seed=42)
                    # Add isolated scripts (no edges) with random positions
                    for _s in _scripts:
                        if _s not in _pos:
                            _pos[_s] = np.random.uniform(-1, 1, 2)

                    _SYS_COLORS = {
                        "movement": "#3498DB", "combat": "#E74C3C", "animation": "#9B59B6",
                        "ui": "#2ECC71", "inventory": "#F39C12", "audio": "#1ABC9C",
                        "ai": "#E67E22", "camera": "#95A5A6", "save": "#F1C40F",
                        "input": "#D35400", "network": "#8E44AD",
                    }

                    _fig, _ax = plt.subplots(figsize=(14, 9))
                    _ax.set_facecolor("#0E1117")
                    _fig.patch.set_facecolor("#0E1117")
                    _ax.axis("off")

                    # Draw system circles (semi-transparent, behind nodes)
                    for _sys in _all_systems:
                        _members = [_s for _s in _scripts if _sys in _file_systems[_s]]
                        if not _members:
                            continue
                        _pts = np.array([_pos[_s] for _s in _members])
                        _cx, _cy = _pts.mean(axis=0)
                        if len(_members) == 1:
                            _r = 0.22
                        else:
                            _r = float(np.sqrt((((_pts - [_cx, _cy])**2).sum(axis=1))).max()) + 0.22
                        _col = _SYS_COLORS.get(_sys, "#aaaaaa")
                        _circ = Circle((_cx, _cy), _r, linewidth=2.0,
                                       edgecolor=_col, facecolor=_col, alpha=0.07)
                        _ax.add_patch(_circ)
                        # System label near top of circle
                        _ax.text(_cx, _cy + _r - 0.04, _sys.upper(),
                                 ha="center", va="top", color=_col,
                                 fontsize=8, fontweight="bold", alpha=0.95)

                    # Draw faint edges
                    for _e1, _e2 in _G.edges():
                        _x1, _y1 = _pos[_e1]; _x2, _y2 = _pos[_e2]
                        _ax.plot([_x1, _x2], [_y1, _y2], "-", color="white", alpha=0.07, linewidth=0.7)

                    # Draw nodes
                    for _s in _scripts:
                        _x, _y = _pos[_s]
                        _sys_list = sorted(_file_systems[_s])
                        _dot_col = _SYS_COLORS.get(_sys_list[0], "#607D8B") if _sys_list else "#607D8B"
                        _ax.scatter(_x, _y, s=100, c=_dot_col, zorder=5, alpha=0.95, edgecolors="white", linewidths=0.5)
                        _ax.text(_x, _y + 0.06, os.path.basename(_s),
                                 ha="center", va="bottom", color="white",
                                 fontsize=6.5, alpha=0.85)

                    # Legend
                    _legend_handles = [
                        mpatches.Patch(color=_SYS_COLORS.get(_sys, "#aaa"), label=_sys)
                        for _sys in _all_systems
                    ]
                    _ax.legend(handles=_legend_handles, loc="upper right",
                               framealpha=0.25, facecolor="#1a1a2e",
                               labelcolor="white", fontsize=8, edgecolor="#444")

                    plt.tight_layout(pad=0.5)
                    st.pyplot(_fig, use_container_width=True)
                    plt.close(_fig)

            except ImportError:
                st.warning("matplotlib not installed. Run: `pip install matplotlib>=3.7.0`")
            except Exception as _ce:
                st.error(f"Cluster map error: {_ce}")

# --- TAB 4: QDRANT DASHBOARD ---
with tab_qdrant:
    if not st.session_state.active_project: st.info("👋 Select a project folder in the sidebar to begin.")
    else:
        st.subheader("🗄️ Vector Database Connection")

        mode_col, host_col, port_col = st.columns([1, 1.5, 0.8])
        with mode_col:
            st.session_state.qdrant_mode = st.selectbox(
                "Storage Mode",
                options=["local", "server"],
                index=0 if st.session_state.qdrant_mode == "local" else 1,
                help="**Local**: embedded DB in .scout/vectors/ (no server needed). **Server**: remote Qdrant instance.",
            )
        with host_col:
            st.session_state.qdrant_host = st.text_input(
                "Qdrant Host IP", value=st.session_state.qdrant_host,
                disabled=(st.session_state.qdrant_mode == "local"),
            )
        with port_col:
            st.session_state.qdrant_port = st.number_input(
                "Qdrant Port", value=st.session_state.qdrant_port,
                disabled=(st.session_state.qdrant_mode == "local"),
            )
        if st.session_state.qdrant_mode == "local":
            scout_dir = st.session_state.active_project.get("scout_dir", "") if st.session_state.active_project else ""
            st.caption(f"Vectors stored in: `{scout_dir}/vectors/`")
        
        st.divider()
        head_col, btn_col = st.columns([0.8, 0.2])
        with head_col: st.subheader("📊 Collection Explorer")
        with btn_col: 
            if st.button("🔄 Refresh Stats", use_container_width=True): st.rerun()
        
        try:
            local_q_client = _get_qc()
            collections = local_q_client.get_collections().collections
            col_names = [c.name for c in collections]

            with st.expander("✨ Create New Collection", expanded=not bool(col_names)):
                st.markdown("Create a hybrid vector database schema (Dense + Sparse).")
                cc_col1, cc_col2 = st.columns([0.6, 0.4])

                with cc_col1:
                    new_col_name = st.text_input("Collection Name", value=st.session_state.active_project['id'] if st.session_state.active_project else "new_collection")
                with cc_col2:
                    dim_preset = st.selectbox("Semantic Dimension", options=[
                        "768 (nomic-embed-text / mxbai)",
                        "1024 (BAAI/bge-large)",
                        "384 (all-MiniLM-L6-v2)",
                        "1536 (OpenAI ada-002)",
                        "Custom"
                    ])
                    new_col_dim = st.number_input("Custom", min_value=1, value=EMBED_DIM) if dim_preset == "Custom" else int(dim_preset.split(" ")[0])

                if st.button("➕ Initialize Collection", use_container_width=True, type="primary"):
                    if new_col_name in col_names:
                        st.error(f"⚠️ A collection named '{new_col_name}' already exists!")
                    else:
                        try:
                            create_hybrid_coll(local_q_client, new_col_name, new_col_dim)
                            st.session_state.active_collection = new_col_name
                            st.success(f"✅ Collection '{new_col_name}' created!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Creation Error: {e}")

            if not col_names: 
                st.warning("No vector collections found in Qdrant. Use the menu above to initialize one.")
            else:
                def_idx = col_names.index(st.session_state.active_collection) if st.session_state.active_collection in col_names else 0
                st.session_state.active_collection = st.selectbox("🔍 Select Active Collection", col_names, index=def_idx)
                
                col_info = local_q_client.get_collection(st.session_state.active_collection)
                m_col1, m_col2, m_col3 = st.columns(3)
                
                # Read dimensions from named "dense" vector or fallback to unnamed/legacy
                v_conf = col_info.config.params.vectors
                if isinstance(v_conf, dict) and "dense" in v_conf:
                    dim_val = v_conf["dense"].size
                    dist_val = str(v_conf["dense"].distance).split('.')[-1]
                elif isinstance(v_conf, dict):
                    # Try first available named vector
                    _first = next(iter(v_conf.values()), None)
                    dim_val = getattr(_first, 'size', 'N/A') if _first else 'N/A'
                    dist_val = str(getattr(_first, 'distance', 'N/A')).split('.')[-1] if _first else 'N/A'
                else:
                    dim_val = getattr(v_conf, 'size', 'N/A')
                    dist_val = str(getattr(v_conf, 'distance', 'N/A')).split('.')[-1]

                m_col1.metric("Total Points", col_info.points_count)
                m_col2.metric("Semantic Dimension", dim_val)
                m_col3.metric("Distance Metric", dist_val)
                
                # Display Hybrid Status
                # is_hybrid = bool(col_info.config.params.sparse_vectors)
                # st.caption(f"🧬 **Hybrid Search Enabled:** `{'Yes' if is_hybrid else 'No (Dense Only)'}`")

        except Exception as e:
            st.error(f"Qdrant Connection Error: {e}")

        

        qdrant_url = f"http://{st.session_state.qdrant_host}:{st.session_state.qdrant_port}/dashboard"
        st.markdown(f"**Web UI:** [Open Qdrant Dashboard]({qdrant_url}) ↗️")


with tab_roadmap:
    if not st.session_state.active_project:
        st.info("Select a project to manage milestones.")
    else:
        scout_dir = st.session_state.active_project['scout_dir']
        data = load_project_data(scout_dir)
        
        # 1. Global Goal Management
        st.subheader("🎯 Global Project Goal")
        new_goal = st.text_area("What are we building?", value=data.get('global_goal', ''), height=100)
        if st.button("Update Goal"):
            data['global_goal'] = new_goal
            save_project_data(scout_dir, data)
            st.success("Goal Saved!")

        st.divider()

        # 2. Milestones CRUD (Your existing logic)
        st.subheader("🏁 Project Milestones")
        new_ms = st.text_input("Add Milestone", placeholder="e.g., Inventory System Alpha")
        if st.button("Add Milestone") and new_ms:
            if 'milestones' not in data: data['milestones'] = []
            data['milestones'].append({"name": new_ms, "status": "pending"})
            save_project_data(scout_dir, data)
            st.rerun()
            
        for i, ms in enumerate(data.get('milestones', [])):
            m_col1, m_col2, m_col3 = st.columns([0.6, 0.2, 0.2])
            m_col1.write(f"{'✅' if ms['status'] == 'completed' else '⏳'} **{ms['name']}**")
            if ms['status'] == 'pending' and m_col2.button("Done", key=f"ms_d_{i}"):
                ms['status'] = 'completed'
                save_project_data(scout_dir, data)
                st.rerun()
            if m_col3.button("🗑️", key=f"ms_x_{i}"):
                data['milestones'].pop(i)
                save_project_data(scout_dir, data)
                st.rerun()

        st.divider()
        
        # 3. AI Health Check
        if st.button("🧠 Run AI Progress Audit", use_container_width=True, type="primary"):
            with st.spinner("Analyzing codebase vs. roadmap..."):
                # We pull the current context to give the AI real data to audit
                # (You can use build_surgical_prompt here with a specific 'audit' query)
                check_prompt = f"Global Goal: {data['global_goal']}\nMilestones: {data['milestones']}\n\nTask: Based on the current code, estimate completion % and suggest the next 3 technical tasks."
                analysis = ask_local_llm(
                            check_prompt,
                            model_name=st.session_state.ui_model_name,
                            max_tokens=st.session_state.ui_max_tokens,
                            num_ctx=st.session_state.ui_num_ctx,
                        )
                st.info(analysis)


with tab_export:
    if not st.session_state.active_project: 
        st.info("👋 Select a project folder in the sidebar to begin.")
    else:
        st.subheader("☁️ Cloud AI Handshake")
        st.markdown("""
        Use this for **Heavy Reasoning**. Local models are fast, but **Claude 3.5 Sonnet** or **Gemini 1.5 Pro** excel at complex refactoring and cross-script logic.
        """)
        st.divider()

        # --- PROMPT INSPIRATION GALLERY ---
# --- PROMPT INSPIRATION GALLERY ---
        st.caption("💡 Inspiration: Copy these into the box below")
        
        cols = st.columns(2)
        
        with cols[0]:
            if st.button("🏗️ Architectural Audit", use_container_width=True):
                st.session_state.cloud_goal_input = "Conduct a comprehensive audit of my signal routing and dependency graph. Focus specifically on the relationships between Autoloads (Singletons) and UI components. Identify any circular dependencies, tight coupling, or race conditions. Design a highly decoupled 'Event Bus' or 'Signal Broker' architecture, detailing the exact refactoring steps to migrate the current state without breaking the game loop."
            
            if st.button("⚔️ Combat System Refactor", use_container_width=True):
                st.session_state.cloud_goal_input = "Analyze the current coupling between my combat managers and hitbox systems. I need to architect a highly modular Status Effect pipeline (e.g., Burn, Freeze, Stun). Propose a Godot-idiomatic design—such as using custom `Resource` scripts for effect data and a generic `Component` node pattern for effect application. Detail how this integrates with my existing combat loops and scales to dozens of enemy types."
                
            if st.button("🧭 Project Evolution & Roadmap", use_container_width=True):
                st.session_state.cloud_goal_input = "Act as a Lead Game Designer and Technical Architect. Evaluate my current codebase and active milestones to propose the next logical phase of development. Identify any 'technical debt' or legacy systems that must be refactored before scaling. Pitch three novel feature ideas that seamlessly synergize with my existing mechanics, and synthesize these recommendations into a concrete, step-by-step development roadmap."

        with cols[1]:
            if st.button("💾 Save/Load Infrastructure", use_container_width=True):
                st.session_state.cloud_goal_input = "Evaluate my current state-management and data structures to design a robust, scalable Save/Load infrastructure. Suggest a migration to a `Resource`-based serialization approach. Provide the architecture for a central SaveManager Autoload, detail how it should dynamically serialize/deserialize active scene nodes, and include a structural strategy for handling save-file versioning and backwards compatibility."
            
            if st.button("⚡ Performance Deep-Dive", use_container_width=True):
                st.session_state.cloud_goal_input = "Perform a rigorous performance analysis on the provided codebase. Focus heavily on `_process`, `_physics_process`, collision logic, and memory allocation. Identify execution bottlenecks or expensive node tree operations. Propose high-impact, Godot-specific optimizations such as implementing an Object Pool, transitioning heavy logic to the PhysicsServer, or restructuring scene hierarchies."

            if st.button("🧠 State Machine & Entity AI", use_container_width=True):
                st.session_state.cloud_goal_input = "Evaluate the logic currently controlling my game entities (Player, Enemies, NPCs). Design a highly scalable, decoupled Finite State Machine (FSM) or Behavior Tree architecture. Detail how states should communicate seamlessly with the AnimationPlayer, physics loop, and external managers without hardcoding dependencies. Provide the base class structures and a step-by-step guide for migrating my current monolithic logic into this modular pattern."
        
        # Text Area bound to the buttons above
        cloud_goal = st.text_area(
            "What do you want the Cloud AI to do?", 
            key="cloud_goal_input", 
            placeholder="e.g., Refactor the inventory system...", 
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_btn = st.button("💎 Generate Master Export", use_container_width=True, type="primary")
            
        if generate_btn and cloud_goal:
            with st.status("Compiling Full Project Identity...", expanded=True) as status:
                
                # 🛡️ THE UPGRADE: We bypass Qdrant and Ollama completely for instant compilation!
                massive_prompt = build_cloud_master_prompt(
                    query=cloud_goal,
                    active_project=st.session_state.active_project,
                    persona_key=st.session_state.get('ui_persona', 'Senior Architect'),
                    collection_name=st.session_state.active_collection,
                )
                
                status.update(label="✅ Master Prompt Ready!", state="complete")                
                
                st.markdown("**Click the copy icon in the top right of this box, then paste it into your browser:**")
                st.code(massive_prompt, language="markdown")
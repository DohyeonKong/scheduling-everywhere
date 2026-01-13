import streamlit as st
import openai
import re
import json
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="Scheduling, Everywhere",
    page_icon="�icing",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 15px;
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stChatMessage {
        padding: 10px;
    }
    /* Right column code blocks */
    div[data-testid="stCodeBlock"] {
        max-height: 400px;
        overflow-y: auto;
    }
    /* Loading modal overlay */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    .loading-modal {
        background: white;
        padding: 40px 60px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .loading-modal h3 {
        margin: 0 0 15px 0;
        color: #1e3a5f;
    }
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #1e3a5f;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h2>Scheduling, Everywhere [DEMO]</h2><p>Schedule Optimization Modeling Agent (Powered by GPT-5.2)</p></div>', unsafe_allow_html=True)

# Paths
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROMPT_DIR = PROJECT_DIR / "prompt"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt.txt"
USER_PROMPT_FILE = PROMPT_DIR / "user_prompt.txt"
CHAT_HISTORY_DIR = APP_DIR / "chat_histories"
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Get client IP
def get_client_ip():
    try:
        # Try to get from Streamlit headers (works with Cloudflare)
        headers = st.context.headers
        # Cloudflare passes real IP in these headers
        for header in ["Cf-Connecting-Ip", "X-Forwarded-For", "X-Real-Ip"]:
            if header in headers:
                return headers[header].split(",")[0].strip()
        return "unknown"
    except:
        return "local"

# List saved chat histories
def list_chat_histories():
    histories = []
    for f in sorted(CHAT_HISTORY_DIR.glob("*.json"), reverse=True):
        histories.append(f.stem)
    return histories

# Load system prompt
def load_system_prompt():
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text()
    return """You are an optimization modeling assistant..."""

# Load user prompt (default input)
def load_user_prompt():
    if USER_PROMPT_FILE.exists():
        return USER_PROMPT_FILE.read_text()
    return ""

# Load data files from data/ folder
def load_data_files():
    data_files = {}
    if DATA_DIR.exists():
        for csv_file in DATA_DIR.glob("*.csv"):
            data_files[csv_file.name] = csv_file.read_text()
    return data_files

SYSTEM_PROMPT = load_system_prompt()

# Session state initialization - load files fresh on first run
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.mzn_code = "% model.mzn will appear here after generation..."
    st.session_state.dzn_code = "% data.dzn will appear here after generation..."
    # Auto-load data files from data/ folder
    st.session_state.uploaded_files_content = load_data_files()
    st.session_state.user_input = load_user_prompt()
    # Session tracking
    st.session_state.client_ip = get_client_ip()
    st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save/Load chat history
def save_chat_history(name=None):
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = {
        "client_ip": st.session_state.client_ip,
        "session_start": st.session_state.session_start,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages,
        "mzn_code": st.session_state.mzn_code,
        "dzn_code": st.session_state.dzn_code
    }
    filepath = CHAT_HISTORY_DIR / f"{name}.json"
    filepath.write_text(json.dumps(history, ensure_ascii=False, indent=2))
    return name

def load_chat_history(name):
    filepath = CHAT_HISTORY_DIR / f"{name}.json"
    if filepath.exists():
        try:
            history = json.loads(filepath.read_text())
            st.session_state.messages = history.get("messages", [])
            st.session_state.mzn_code = history.get("mzn_code", "")
            st.session_state.dzn_code = history.get("dzn_code", "")
            st.session_state.client_ip = history.get("client_ip", "unknown")
            st.session_state.session_start = history.get("session_start", "")
            return True
        except:
            return False
    return False

def auto_save():
    """Auto-save current session"""
    session_id = st.session_state.session_start.replace("-", "").replace(":", "").replace(" ", "_")
    save_chat_history(f"session_{session_id}")

# Extract code blocks from response
def extract_code_blocks(text):
    mzn_match = re.search(r'```(?:mzn|minizinc)\n(.*?)```', text, re.DOTALL)
    dzn_match = re.search(r'```dzn\n(.*?)```', text, re.DOTALL)

    mzn_code = mzn_match.group(1).strip() if mzn_match else None
    dzn_code = dzn_match.group(1).strip() if dzn_match else None

    return mzn_code, dzn_code

# Build messages for API call
def build_api_messages():
    api_messages = []
    for msg in st.session_state.messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return api_messages

# Chat message colors CSS
st.html("""
<style>
    /* User chat messages - light sky blue */
    [class*="st-key-user"] {
        background-color: #e0f2fe;
        border-radius: 10px;
        padding: 5px;
    }
    [class*="st-key-user"] .stChatMessage,
    [class*="st-key-user"] pre,
    [class*="st-key-user"] code {
        background-color: transparent !important;
    }

    /* Assistant chat messages - deeper sky blue */
    [class*="st-key-assistant"] {
        background-color: #dbeafe;
        border-radius: 10px;
        padding: 5px;
    }
    [class*="st-key-assistant"] .stChatMessage,
    [class*="st-key-assistant"] pre,
    [class*="st-key-assistant"] code {
        background-color: transparent !important;
    }

    /* Chat input textarea - light sky blue */
    .stTextArea textarea {
        background-color: #e0f2fe !important;
        border-radius: 10px;
    }

    /* Right column code blocks (inside tabs) - deeper sky blue */
    div[data-testid="stTabs"] pre,
    div[data-testid="stTabs"] code,
    .stTabs pre,
    .stTabs code {
        background-color: #dbeafe !important;
        border-radius: 10px;
    }
</style>
""")

# Helper function for colored chat messages
def chat_message(name):
    return st.container(key=f"{name}-{uuid.uuid4()}").chat_message(name=name)

# Two columns layout
left_col, right_col = st.columns([1, 1], gap="medium")

# ============ LEFT COLUMN: Chat Interface ============
with left_col:
    st.subheader("Chat")

    # Chat container
    chat_container = st.container(height=550)

    with chat_container:
        for message in st.session_state.messages:
            with chat_message(name=message["role"]):
                # Display full message including code blocks
                st.markdown(message["content"])

    # Chat input with pre-filled prompt
    user_input = st.text_area(
        "Message",
        value=st.session_state.user_input,
        height=150,
        label_visibility="collapsed"
    )

    if st.button("Send", type="primary", use_container_width=True):
        if user_input.strip():
            # Update session state
            st.session_state.user_input = ""

            # Build user message with file contents if first message
            user_content = user_input

            if st.session_state.uploaded_files_content and len(st.session_state.messages) == 0:
                user_content += "\n\n--- Uploaded Files ---\n"
                for name, content in st.session_state.uploaded_files_content.items():
                    user_content += f"\n### {name}\n```csv\n{content}\n```\n"

            # Add user message with timestamp
            st.session_state.messages.append({
                "role": "user",
                "content": user_content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Show loading modal
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""
                <div class="loading-overlay">
                    <div class="loading-modal">
                        <div class="loading-spinner"></div>
                        <h3>Thinking...</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Call OpenAI API
            try:
                client = openai.OpenAI()

                # Build messages with system prompt
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                api_messages.extend(build_api_messages())

                response = client.chat.completions.create(
                    model="gpt-5.2",
                    max_completion_tokens=8192,
                    messages=api_messages
                )

                assistant_response = response.choices[0].message.content

                # Extract code blocks
                mzn_code, dzn_code = extract_code_blocks(assistant_response)

                if mzn_code:
                    st.session_state.mzn_code = mzn_code
                if dzn_code:
                    st.session_state.dzn_code = dzn_code

                # Add assistant message with timestamp
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Auto-save after each response
                auto_save()

            except openai.APIConnectionError:
                st.error("Connection error. Check your internet connection.")
            except openai.AuthenticationError:
                st.error("Authentication error. Check your OPENAI_API_KEY.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                loading_placeholder.empty()

            st.rerun()

    st.divider()

    # Data files section (auto-loaded)
    with st.expander(f"📁 Data Files ({len(st.session_state.uploaded_files_content)} loaded)", expanded=False):
        if st.session_state.uploaded_files_content:
            st.success(f"✓ {len(st.session_state.uploaded_files_content)} file(s) ready")
            for name, content in st.session_state.uploaded_files_content.items():
                with st.expander(f"Preview: {name}"):
                    lines = content.split('\n')[:6]
                    st.code('\n'.join(lines), language="csv")

    # System prompt viewer
    with st.popover("📋 View System Prompt", use_container_width=True):
        st.markdown("**System Prompt:**")
        st.code(SYSTEM_PROMPT, language="markdown")

    # Chat history controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            name = save_chat_history()
            st.toast(f"Saved: {name}")
    with col2:
        histories = list_chat_histories()
        if histories:
            selected = st.selectbox("📂 Load", histories, label_visibility="collapsed")
            if st.button("Load", use_container_width=True):
                if load_chat_history(selected):
                    st.toast(f"Loaded: {selected}")
                    st.rerun()
        else:
            st.button("📂 Load", disabled=True, use_container_width=True)
    with col3:
        if st.button("🗑️ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.mzn_code = "% model.mzn will appear here..."
            st.session_state.dzn_code = "% data.dzn will appear here..."
            st.session_state.user_input = load_user_prompt()
            st.session_state.uploaded_files_content = load_data_files()
            st.session_state.client_ip = get_client_ip()
            st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

# ============ RIGHT COLUMN: MiniZinc Output ============
with right_col:
    st.subheader("MiniZinc Output")

    # Tabs for .mzn and .dzn
    tab1, tab2 = st.tabs(["model.mzn", "data.dzn"])

    with tab1:
        st.code(st.session_state.mzn_code, language="minizinc", line_numbers=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download .mzn",
                data=st.session_state.mzn_code,
                file_name="model.mzn",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("📋 Copy .mzn", use_container_width=True):
                st.toast("Use Ctrl+C to copy from code block above")

    with tab2:
        st.code(st.session_state.dzn_code, language="minizinc", line_numbers=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download .dzn",
                data=st.session_state.dzn_code,
                file_name="data.dzn",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("📋 Copy .dzn", use_container_width=True):
                st.toast("Use Ctrl+C to copy from code block above")


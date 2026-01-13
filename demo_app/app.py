import streamlit as st
import openai
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (local) or Streamlit secrets (cloud)
load_dotenv()

# Support Streamlit Cloud secrets
import os
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    pass  # No secrets file (running locally)

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
st.markdown('<div class="main-header"><h2>🏭 Scheduling, Everywhere [DEMO]</h2><p>Schedule Optimization Modeling Agent (Powered by GPT-5.2)</p></div>', unsafe_allow_html=True)

# Paths
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROMPT_DIR = PROJECT_DIR / "prompt"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt.txt"
USER_PROMPT_FILE = PROMPT_DIR / "user_prompt.txt"
CHAT_HISTORY_FILE = APP_DIR / "chat_history.json"

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

# Save/Load chat history
def save_chat_history():
    history = {
        "messages": st.session_state.messages,
        "mzn_code": st.session_state.mzn_code,
        "dzn_code": st.session_state.dzn_code
    }
    CHAT_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2))

def load_chat_history():
    if CHAT_HISTORY_FILE.exists():
        try:
            history = json.loads(CHAT_HISTORY_FILE.read_text())
            st.session_state.messages = history.get("messages", [])
            st.session_state.mzn_code = history.get("mzn_code", "")
            st.session_state.dzn_code = history.get("dzn_code", "")
            return True
        except:
            return False
    return False

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

# Two columns layout
left_col, right_col = st.columns([1, 1], gap="medium")

# ============ LEFT COLUMN: Chat Interface ============
with left_col:
    st.subheader("💬 Chat")

    # Chat container
    chat_container = st.container(height=550)

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
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

            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_content
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

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })

                # Auto-save after each response
                save_chat_history()

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

    # Chat history controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Chat", use_container_width=True):
            save_chat_history()
            st.toast("Chat saved!")
    with col2:
        if st.button("📂 Load Chat", use_container_width=True):
            if load_chat_history():
                st.toast("Chat loaded!")
                st.rerun()
            else:
                st.toast("No saved chat found")
    with col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.mzn_code = "% model.mzn will appear here..."
            st.session_state.dzn_code = "% data.dzn will appear here..."
            st.session_state.user_input = load_user_prompt()
            st.session_state.uploaded_files_content = load_data_files()
            st.rerun()

# ============ RIGHT COLUMN: MiniZinc Output ============
with right_col:
    st.subheader("📄 MiniZinc Output")

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


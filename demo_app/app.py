import streamlit as st
import openai
import json
import uuid
import tempfile
import os
import pandas as pd

# Set Gurobi library path for MiniZinc
os.environ.setdefault("GUROBI_HOME", "/Library/gurobi1300/macos_universal2")
os.environ.setdefault("DYLD_LIBRARY_PATH", f"{os.environ.get('GUROBI_HOME', '')}/lib")
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import minizinc

# Import tools for agent loop
from tools import ModelBuilder, TOOL_SCHEMAS, execute_tool, get_product_order

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
    /* Top-level tab labels (Model/Solver) - larger font like subheader */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
        font-weight: 600;
    }
    /* Nested tabs (sub-tabs) - reset to normal size */
    .stTabs .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h2>Scheduling, Everywhere [DEMO]</h2><p>Scheduling Problem Modeling Agent (Powered by GPT-5.2)</p></div>', unsafe_allow_html=True)

# Paths
APP_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
PROMPT_DIR = PROJECT_DIR / "prompt"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt.txt"
USER_PROMPT_FILE = PROMPT_DIR / "user_prompt.txt"
TEMPLATE_FILE = PROMPT_DIR / "template.mzn"
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

# Load system prompt (with template injection)
def load_system_prompt():
    if SYSTEM_PROMPT_FILE.exists():
        prompt = SYSTEM_PROMPT_FILE.read_text()
        # Inject template.mzn content if placeholder exists
        if "{{TEMPLATE}}" in prompt and TEMPLATE_FILE.exists():
            template_content = TEMPLATE_FILE.read_text()
            prompt = prompt.replace("{{TEMPLATE}}", template_content)
        return prompt
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

# Parse products from demand.csv (aggregate by MOLD, SIZE, COLOR)
def parse_products_from_demand(demand_csv_content):
    """Extract unique (MOLD, SIZE, COLOR) products from demand.csv"""
    products = []
    seen = set()

    lines = demand_csv_content.strip().split('\n')
    if len(lines) < 2:
        return products

    # Parse header
    header = lines[0].split(',')
    try:
        mold_idx = header.index('MOLD')
        size_idx = header.index('SIZE')
        color_idx = header.index('COLOR')
    except ValueError:
        return products

    # Parse data rows - maintain order of first appearance
    for line in lines[1:]:
        cols = line.split(',')
        if len(cols) > max(mold_idx, size_idx, color_idx):
            mold = cols[mold_idx].strip()
            size = cols[size_idx].strip()
            color = cols[color_idx].strip()
            key = (mold, size, color)
            if key not in seen:
                seen.add(key)
                # Simplify color name (e.g., BIP024_WHITE -> WHITE)
                color_short = color.split('_')[-1] if '_' in color else color
                products.append({"mold": mold, "size": size, "color": color_short})

    return products

SYSTEM_PROMPT = load_system_prompt()

# Session state initialization - load files fresh on first run
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.math_model = "Mathematical formulation will appear here after generation..."
    st.session_state.explanation = ""
    st.session_state.mzn_code = "% model.mzn will appear here after generation..."
    st.session_state.dzn_code = "% data.dzn will appear here after generation..."
    # Auto-load data files from data/ folder
    st.session_state.uploaded_files_content = load_data_files()
    st.session_state.user_input = load_user_prompt()
    # Parse product list from demand.csv for display
    if "demand.csv" in st.session_state.uploaded_files_content:
        st.session_state.product_list = parse_products_from_demand(
            st.session_state.uploaded_files_content["demand.csv"]
        )
    else:
        st.session_state.product_list = []
    # Session tracking
    st.session_state.client_ip = get_client_ip()
    st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Solver state
    st.session_state.solver_result = None
    st.session_state.solver_status = None
    st.session_state.solver_time = None
    st.session_state.schedule_table = []  # Mapped schedule table

# Save/Load chat history
def save_chat_history(name=None):
    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = {
        "client_ip": st.session_state.client_ip,
        "session_start": st.session_state.session_start,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages,
        "math_model": st.session_state.math_model,
        "explanation": st.session_state.get("explanation", ""),
        "mzn_code": st.session_state.mzn_code,
        "dzn_code": st.session_state.dzn_code,
        "solver_status": st.session_state.solver_status,
        "solver_time": st.session_state.solver_time,
        "solver_result": st.session_state.solver_result,
        "schedule_table": st.session_state.get("schedule_table", [])
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
            st.session_state.math_model = history.get("math_model", "")
            st.session_state.explanation = history.get("explanation", "")
            st.session_state.mzn_code = history.get("mzn_code", "")
            st.session_state.dzn_code = history.get("dzn_code", "")
            st.session_state.client_ip = history.get("client_ip", "unknown")
            st.session_state.session_start = history.get("session_start", "")
            st.session_state.solver_status = history.get("solver_status", None)
            st.session_state.solver_time = history.get("solver_time", None)
            st.session_state.solver_result = history.get("solver_result", None)
            st.session_state.schedule_table = history.get("schedule_table", [])
            return True
        except:
            return False
    return False

def auto_save():
    """Auto-save current session"""
    session_id = st.session_state.session_start.replace("-", "").replace(":", "").replace(" ", "")
    save_chat_history(f"{session_id}_session")

# Build messages for API call
def build_api_messages():
    api_messages = []
    for msg in st.session_state.messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return api_messages

# ============================================================
# Schedule Extraction via Function Calling
# ============================================================

EXTRACT_SCHEDULE_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_schedule",
        "description": "Define how to extract schedule from solver result. Specify the quantity variable and index meanings, plus lookup arrays for product info.",
        "parameters": {
            "type": "object",
            "properties": {
                "var": {
                    "type": "string",
                    "description": "Variable name containing production quantities (e.g., 'x', 'q', 'production')"
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["machine", "side", "shift", "product", "day", "size", "color"]},
                    "description": "What each dimension represents, in order. E.g., ['machine', 'side', 'shift', 'product']"
                },
                "size_lookup": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "string", "description": "dzn array name containing size values"},
                        "index_array": {"type": "string", "description": "dzn array mapping product index to size index (optional)"}
                    },
                    "description": "How to lookup size for each product"
                },
                "color_lookup": {
                    "type": "object",
                    "properties": {
                        "array": {"type": "string", "description": "dzn array name containing color values"},
                        "index_array": {"type": "string", "description": "dzn array mapping product index to color index (optional)"}
                    },
                    "description": "How to lookup color for each product"
                },
                "mold_value": {
                    "type": "string",
                    "description": "Constant mold code value (e.g., 'MS252801-1') or dzn variable name"
                }
            },
            "required": ["var", "indices"]
        }
    }
}


def parse_dzn_arrays(dzn_code: str) -> dict:
    """Parse dzn code to extract array values for lookups."""
    import re
    arrays = {}

    # Pattern: name = [val1, val2, ...];
    pattern = r'(\w+)\s*=\s*\[(.*?)\];'
    for match in re.finditer(pattern, dzn_code, re.DOTALL):
        name = match.group(1)
        content = match.group(2).strip()
        if not content:
            arrays[name] = []
            continue

        # Parse values
        if '"' in content:
            # String array
            values = re.findall(r'"([^"]*)"', content)
        else:
            # Numeric array
            values = [v.strip() for v in content.split(',')]
            # Try to convert to int/float
            try:
                values = [int(v) for v in values]
            except:
                try:
                    values = [float(v) for v in values]
                except:
                    pass
        arrays[name] = values

    # Also parse scalar values: name = value;
    scalar_pattern = r'(\w+)\s*=\s*([^;\[\]]+);'
    for match in re.finditer(scalar_pattern, dzn_code):
        name = match.group(1)
        if name not in arrays:
            value = match.group(2).strip().strip('"')
            arrays[name] = value

    return arrays


def execute_extract_schedule(var: str, indices: list, solver_result: dict, dzn_arrays: dict,
                              size_lookup: dict = None, color_lookup: dict = None,
                              mold_value: str = None) -> list:
    """Execute the schedule extraction based on LLM-provided mapping.

    Args:
        var: Variable name containing production quantities
        indices: What each dimension represents
        solver_result: Solver output dict
        dzn_arrays: Parsed dzn arrays
        size_lookup: {'array': 'size_values', 'index_array': 'product_size_idx'} or None
        color_lookup: {'array': 'color_values', 'index_array': 'product_color_idx'} or None
        mold_value: Constant mold code or dzn array name
    """
    if var not in solver_result:
        print(f"[ERROR] Variable '{var}' not found in solver result")
        return []

    data = solver_result[var]
    schedule = []
    shifts_per_day = 3

    # Resolve lookup arrays using LLM-specified names
    size_values = None
    size_index_map = None
    color_values = None
    color_index_map = None
    mold_resolved = None

    if size_lookup:
        arr_name = size_lookup.get("array")
        idx_arr_name = size_lookup.get("index_array")
        if arr_name and arr_name in dzn_arrays:
            size_values = dzn_arrays[arr_name]
            print(f"[DEBUG] Using size_lookup array '{arr_name}' with {len(size_values)} values")
        if idx_arr_name and idx_arr_name in dzn_arrays:
            size_index_map = dzn_arrays[idx_arr_name]
            print(f"[DEBUG] Using size index_array '{idx_arr_name}'")

    if color_lookup:
        arr_name = color_lookup.get("array")
        idx_arr_name = color_lookup.get("index_array")
        if arr_name and arr_name in dzn_arrays:
            color_values = dzn_arrays[arr_name]
            print(f"[DEBUG] Using color_lookup array '{arr_name}' with {len(color_values)} values")
        if idx_arr_name and idx_arr_name in dzn_arrays:
            color_index_map = dzn_arrays[idx_arr_name]
            print(f"[DEBUG] Using color index_array '{idx_arr_name}'")

    if mold_value:
        if mold_value in dzn_arrays:
            mold_resolved = dzn_arrays[mold_value]
            print(f"[DEBUG] Using mold from dzn array '{mold_value}'")
        else:
            mold_resolved = mold_value  # Use as literal value
            print(f"[DEBUG] Using mold literal value '{mold_value}'")

    def get_size(product_idx: int) -> str:
        """Get size for a product index."""
        if size_index_map and size_values:
            # Use index mapping: product_idx -> size_idx -> size_value
            if product_idx < len(size_index_map):
                size_idx = size_index_map[product_idx]
                # MiniZinc is 1-indexed, Python is 0-indexed
                if isinstance(size_idx, int) and size_idx >= 1 and size_idx <= len(size_values):
                    return str(size_values[size_idx - 1])
        elif size_values:
            # Direct mapping: product_idx -> size_value
            if product_idx < len(size_values):
                return str(size_values[product_idx])
        return f"P{product_idx + 1}"

    def get_color(product_idx: int) -> str:
        """Get color for a product index."""
        if color_index_map and color_values:
            # Use index mapping: product_idx -> color_idx -> color_value
            if product_idx < len(color_index_map):
                color_idx = color_index_map[product_idx]
                if isinstance(color_idx, int) and color_idx >= 1 and color_idx <= len(color_values):
                    color = color_values[color_idx - 1]
                    # Simplify color name (e.g., BIP024_WHITE -> WHITE)
                    if '_' in str(color):
                        color = str(color).split('_')[-1]
                    return color
        elif color_values:
            # Direct mapping: product_idx -> color_value
            if product_idx < len(color_values):
                color = color_values[product_idx]
                if '_' in str(color):
                    color = str(color).split('_')[-1]
                return color
        return ""

    def get_mold(product_idx: int) -> str:
        """Get mold for a product index."""
        if mold_resolved:
            if isinstance(mold_resolved, list):
                if product_idx < len(mold_resolved):
                    return str(mold_resolved[product_idx])
            else:
                return str(mold_resolved)
        return ""

    def iterate_array(arr, idx_list, current_indices):
        """Recursively iterate through nested array."""
        if not isinstance(arr, list):
            # Leaf value (qty)
            qty = arr
            if qty > 0:
                row = build_row(current_indices, qty)
                if row:
                    schedule.append(row)
            return

        for i, sub in enumerate(arr):
            iterate_array(sub, idx_list, current_indices + [i])

    def build_row(idx_values, qty):
        """Build a schedule row from index values."""
        row = {"Qty": qty}
        product_idx = None

        for i, idx_type in enumerate(indices):
            if i >= len(idx_values):
                break
            val = idx_values[i]

            if idx_type == "machine":
                row["Machine"] = val + 1
            elif idx_type == "side":
                row["Side"] = "L" if val == 0 else "R"
            elif idx_type == "shift":
                # Convert to Day + Shift within day
                day = val // shifts_per_day + 1
                shift_in_day = val % shifts_per_day + 1
                row["Date"] = f"Day {day}"
                row["Shift"] = shift_in_day
            elif idx_type == "day":
                row["Date"] = f"Day {val + 1}"
            elif idx_type == "product":
                product_idx = val
                row["Size"] = get_size(val)
                row["Color"] = get_color(val)
                row["MoldCd"] = get_mold(val)
            elif idx_type == "size":
                if size_values and val < len(size_values):
                    row["Size"] = str(size_values[val])
                else:
                    row["Size"] = str(val + 1)
            elif idx_type == "color":
                if color_values and val < len(color_values):
                    color = color_values[val]
                    if '_' in str(color):
                        color = str(color).split('_')[-1]
                    row["Color"] = color
                else:
                    row["Color"] = str(val + 1)

        # Fill missing fields
        row.setdefault("Date", "")
        row.setdefault("Shift", 1)
        row.setdefault("Machine", 1)
        row.setdefault("Side", "")
        row.setdefault("MoldCd", "")
        row.setdefault("Color", "")
        row.setdefault("Size", "")

        return row

    iterate_array(data, indices, [])
    print(f"[DEBUG] Extracted {len(schedule)} schedule rows")
    return schedule


def map_solver_result_with_llm(solver_result: dict, mzn_code: str, dzn_code: str) -> list:
    """Use LLM with function calling to map solver result to schedule table."""
    if not solver_result or not isinstance(solver_result, dict):
        return []

    # Get variable names and shapes from result
    var_info = {}
    for name, val in solver_result.items():
        if isinstance(val, list):
            # Get dimensions
            dims = []
            v = val
            while isinstance(v, list):
                dims.append(len(v))
                v = v[0] if v else None
            var_info[name] = f"array with shape {dims}"
        else:
            var_info[name] = type(val).__name__

    # Parse dzn arrays first to show available arrays
    dzn_arrays = parse_dzn_arrays(dzn_code)
    dzn_array_info = {}
    for name, val in dzn_arrays.items():
        if isinstance(val, list):
            # Show first few values for context
            preview = val[:5]
            dzn_array_info[name] = f"array[{len(val)}] = {preview}..."
        else:
            dzn_array_info[name] = f"scalar = {val}"

    prompt = f"""You created a MiniZinc optimization model. Now map the solver output to a schedule table.

## Your Model (.mzn):
```minizinc
{mzn_code[:4000]}
```

## Your Data (.dzn) - Available Arrays:
{json.dumps(dzn_array_info, indent=2, ensure_ascii=False)}

## Solver Output Variables:
{json.dumps(var_info, indent=2)}

## Task:
Call the `extract_schedule` tool with:

1. **var**: The decision variable containing production quantities (e.g., 'x', 'q', 'production')

2. **indices**: What each array dimension means, in order.
   - Example: If `array[PRODUCTS, SHIFTS, SIDES] of var int: q` → indices=["product", "shift", "side"]
   - Match YOUR model's declaration order exactly

3. **size_lookup**: How to get SIZE for each product
   - Look at your dzn for arrays containing size values (e.g., "size", "sizes", "product_size")
   - If products are indexed and there's a size array aligned with product index, use:
     {{"array": "size_array_name"}}
   - If there's an index mapping array (product → size index), use:
     {{"array": "unique_sizes", "index_array": "product_size_idx"}}

4. **color_lookup**: How to get COLOR for each product
   - Similar to size_lookup, find arrays containing color values
   - Use {{"array": "color_array_name"}} or {{"array": "unique_colors", "index_array": "product_color_idx"}}

5. **mold_value**: The mold code
   - If all products have the same mold, use the literal value (e.g., "MS252801-1")
   - If mold varies, use the dzn array name containing mold codes

IMPORTANT:
- Look at the dzn arrays above to find the correct array names
- The schedule table needs: Date, Shift, Machine, Side, MoldCd, Color, Size, Qty
- Product index dimension will be used to look up Size, Color, MoldCd via the lookup arrays"""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=[EXTRACT_SCHEDULE_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_schedule"}},
            temperature=0
        )

        # Extract tool call
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        print(f"[DEBUG] LLM extract_schedule call:")
        print(f"  var={args.get('var')}")
        print(f"  indices={args.get('indices')}")
        print(f"  size_lookup={args.get('size_lookup')}")
        print(f"  color_lookup={args.get('color_lookup')}")
        print(f"  mold_value={args.get('mold_value')}")

        # Execute extraction with all parameters
        return execute_extract_schedule(
            var=args["var"],
            indices=args["indices"],
            solver_result=solver_result,
            dzn_arrays=dzn_arrays,
            size_lookup=args.get("size_lookup"),
            color_lookup=args.get("color_lookup"),
            mold_value=args.get("mold_value")
        )

    except Exception as e:
        print(f"[ERROR] Failed to map solver result: {e}")
        import traceback
        traceback.print_exc()
        return []


# Build solver result context for LLM
def build_solver_context():
    if not st.session_state.solver_result:
        return None
    if not isinstance(st.session_state.solver_result, dict):
        return None

    result = st.session_state.solver_result
    context_parts = ["--- Current Solver Result ---"]
    context_parts.append(f"Status: {st.session_state.solver_status}")

    if "total_machines" in result:
        context_parts.append(f"Machines Used: {result['total_machines']}")
    if "total_shifts" in result:
        context_parts.append(f"Shifts Used: {result['total_shifts']}")

    # Parse schedule from q variable
    if "q" in result:
        q_data = result["q"]
        product_list = st.session_state.get("product_list", [])

        schedule_lines = ["\nSchedule (Mold, Size, Color, Machine, Side, Shift, Qty):"]
        for p_idx, shifts in enumerate(q_data):
            for t_idx, sides in enumerate(shifts):
                for s_idx, qty in enumerate(sides):
                    if qty > 0:
                        if p_idx < len(product_list):
                            mold_label = product_list[p_idx]["mold"]
                            size_label = product_list[p_idx]["size"]
                            color_label = product_list[p_idx]["color"]
                        else:
                            mold_label = "?"
                            size_label = f"P{p_idx+1}"
                            color_label = ""
                        machine = (s_idx // 2) + 1
                        side_lr = "L" if s_idx % 2 == 0 else "R"
                        schedule_lines.append(f"  {mold_label}, Size {size_label}, {color_label}, Machine {machine}-{side_lr}, Shift {t_idx+1}: {qty} units")
        if len(schedule_lines) > 1:
            context_parts.extend(schedule_lines)
        else:
            context_parts.append("\nNo production scheduled.")

    context_parts.append("--- End Solver Result ---")
    return "\n".join(context_parts)

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

    /* Chat input textarea - grey */
    .stTextArea textarea {
        background-color: #f3f4f6 !important;
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

# Two columns layout (2:3 ratio - chat:output)
left_col, right_col = st.columns([2, 3], gap="medium")

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

    if st.button("Send", type="primary", width='stretch'):
        if user_input.strip():
            # Clear input for next run
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

            # Call OpenAI API with AGENT LOOP (function calling)
            try:
                client = openai.OpenAI()

                # Build messages with system prompt
                system_content = SYSTEM_PROMPT
                solver_context = build_solver_context()
                if solver_context:
                    system_content += f"\n\n{solver_context}"

                messages = [{"role": "system", "content": system_content}]
                messages.extend(build_api_messages())

                # Create model builder for this session
                builder = ModelBuilder(str(DATA_DIR))

                # ============================================================
                # AGENT LOOP - Process tool calls until done
                # ============================================================
                max_iterations = 30  # Safety limit
                iteration = 0
                assistant_response = ""

                print(f"[DEBUG] Starting agent loop...")

                while iteration < max_iterations:
                    iteration += 1
                    print(f"[DEBUG] === Iteration {iteration} ===")

                    # Call LLM with tools
                    response = client.chat.completions.create(
                        model="gpt-5.2",
                        messages=messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="auto"
                    )

                    response_message = response.choices[0].message
                    messages.append(response_message)

                    # Check if LLM wants to call tools
                    if response_message.tool_calls:
                        print(f"[DEBUG] Tool calls: {len(response_message.tool_calls)}")

                        for tool_call in response_message.tool_calls:
                            tool_name = tool_call.function.name
                            arguments = json.loads(tool_call.function.arguments)

                            print(f"[DEBUG] Calling: {tool_name}({list(arguments.keys())})")

                            try:
                                result = execute_tool(tool_name, arguments, builder)
                                result_str = json.dumps(result, default=str, ensure_ascii=False)

                                # Truncate large results for logging
                                if len(result_str) > 500:
                                    print(f"[DEBUG] Result: {result_str[:500]}...")
                                else:
                                    print(f"[DEBUG] Result: {result_str}")

                            except Exception as e:
                                print(f"[DEBUG] Tool error: {e}")
                                result_str = json.dumps({"error": str(e)})

                            # Add tool result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result_str
                            })

                            # Check if finalize was called and successful
                            if tool_name == "finalize" and isinstance(result, dict):
                                if result.get("status") == "complete":
                                    print(f"[DEBUG] Model complete!")
                                    st.session_state.mzn_code = result["mzn_code"]
                                    st.session_state.dzn_code = result["dzn_code"]
                                    if result.get("math_model"):
                                        st.session_state.math_model = result["math_model"]
                                    if result.get("explanation"):
                                        st.session_state.explanation = result["explanation"]

                                    # Update product_list for display
                                    if builder.product_group_by:
                                        demand_file = str(DATA_DIR / "demand.csv")
                                        st.session_state.product_list = get_product_order(
                                            demand_file, builder.product_group_by
                                        )
                                        print(f"[DEBUG] Updated product_list: {len(st.session_state.product_list)} products")

                    else:
                        # No tool calls - LLM is done, get final response
                        assistant_response = response_message.content or ""
                        print(f"[DEBUG] Final response received ({len(assistant_response)} chars)")
                        break

                if iteration >= max_iterations:
                    print(f"[DEBUG] WARNING: Hit max iterations!")
                    assistant_response = "(Agent loop reached maximum iterations)"

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
    with st.popover("📋 View System Prompt", width='stretch'):
        st.markdown("**System Prompt:**")
        st.code(SYSTEM_PROMPT, language="markdown")

    # Chat history controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save", width='stretch'):
            name = save_chat_history()
            st.toast(f"Saved: {name}")
    with col2:
        histories = list_chat_histories()
        if histories:
            selected = st.selectbox("📂 Load", histories, label_visibility="collapsed")
            if st.button("Load", width='stretch'):
                if load_chat_history(selected):
                    st.toast(f"Loaded: {selected}")
                    st.rerun()
        else:
            st.button("📂 Load", disabled=True, width='stretch')
    with col3:
        if st.button("🗑️ New Chat", width='stretch'):
            st.session_state.messages = []
            st.session_state.math_model = "Mathematical formulation will appear here..."
            st.session_state.explanation = ""
            st.session_state.mzn_code = "% model.mzn will appear here..."
            st.session_state.dzn_code = "% data.dzn will appear here..."
            st.session_state.user_input = load_user_prompt()
            st.session_state.uploaded_files_content = load_data_files()
            # Refresh product list from demand.csv
            if "demand.csv" in st.session_state.uploaded_files_content:
                st.session_state.product_list = parse_products_from_demand(
                    st.session_state.uploaded_files_content["demand.csv"]
                )
            else:
                st.session_state.product_list = []
            st.session_state.client_ip = get_client_ip()
            st.session_state.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Clear solver state
            st.session_state.solver_result = None
            st.session_state.solver_status = None
            st.session_state.solver_time = None
            st.session_state.schedule_table = []
            st.rerun()

# ============ RIGHT COLUMN: Output ============
with right_col:
    # Top-level tabs: Model and Solver
    tab_model, tab_solver = st.tabs(["Model", "Solver"])

    # ---- MODEL TAB ----
    with tab_model:
        # Sub-tabs: Math Model and MiniZinc
        sub_math, sub_mzn = st.tabs(["Math Model", "MiniZinc"])

        with sub_math:
            math_container = st.container(height=500)
            with math_container:
                st.markdown(st.session_state.math_model)

        with sub_mzn:
            # Sub-sub-tabs for .mzn and .dzn
            mzn_tab, dzn_tab = st.tabs(["model.mzn", "data.dzn"])

            with mzn_tab:
                st.code(st.session_state.mzn_code, language="minizinc", line_numbers=True, height=450)
                st.download_button(
                    label="📥 Download .mzn",
                    data=st.session_state.mzn_code,
                    file_name="model.mzn",
                    mime="text/plain",
                    width='stretch'
                )

            with dzn_tab:
                st.code(st.session_state.dzn_code, language="minizinc", line_numbers=True, height=450)
                st.download_button(
                    label="📥 Download .dzn",
                    data=st.session_state.dzn_code,
                    file_name="data.dzn",
                    mime="text/plain",
                    width='stretch'
                )

    # ---- SOLVER TAB ----
    with tab_solver:
        # Solver controls
        solver_col1, solver_col2 = st.columns([2, 1])
        with solver_col1:
            # Get available solvers
            # Available solvers (Gurobi first as default - requires valid license)
            available_solvers = ["gurobi", "cbc", "highs", "cplex"]
            default_idx = 0  # Gurobi as default

            selected_solver = st.selectbox("Solver", available_solvers, index=default_idx)

        with solver_col2:
            timeout_sec = st.number_input("Timeout (sec)", min_value=10, max_value=600, value=120)

        # Solve button
        if st.button("▶ Solve", type="primary", width='stretch'):
            mzn_code = st.session_state.mzn_code
            dzn_code = st.session_state.dzn_code

            if mzn_code.startswith("%") and "will appear" in mzn_code:
                st.warning("Please generate a model first using the chat.")
            else:
                with st.spinner("Solving..."):
                    try:
                        # Write to temporary files for reliable loading
                        with tempfile.TemporaryDirectory() as tmpdir:
                            mzn_path = os.path.join(tmpdir, "model.mzn")
                            dzn_path = os.path.join(tmpdir, "data.dzn")

                            # Write .mzn file
                            with open(mzn_path, "w", encoding="utf-8") as f:
                                f.write(mzn_code)

                            # Write .dzn file if available
                            has_dzn = dzn_code and "will appear" not in dzn_code
                            if has_dzn:
                                with open(dzn_path, "w", encoding="utf-8") as f:
                                    f.write(dzn_code)

                            # Gurobi requires subprocess (Python lib can't pass --gurobi-dll)
                            if selected_solver == "gurobi":
                                import subprocess
                                import json as json_lib

                                cmd = [
                                    "minizinc",
                                    "--solver", "gurobi",
                                    "--gurobi-dll", "/Library/gurobi1300/macos_universal2/lib/libgurobi130.dylib",
                                    "--output-mode", "json",
                                    "--json-stream",
                                    "--time-limit", str(timeout_sec * 1000),
                                    "-a",  # all solutions / intermediate
                                    mzn_path
                                ]
                                if has_dzn:
                                    cmd.append(dzn_path)

                                print(f"[DEBUG] Running solver command: {' '.join(cmd)}")
                                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 30)
                                print(f"[DEBUG] Solver return code: {proc.returncode}")

                                # Log stderr if any
                                if proc.stderr:
                                    print(f"[DEBUG] Solver stderr:\n{proc.stderr}")

                                # Parse JSON output
                                solution_dict = {}
                                status = "UNKNOWN"
                                solve_time = "N/A"
                                error_msg = None

                                for line in proc.stdout.strip().split('\n'):
                                    line = line.strip()
                                    if not line or not line.startswith('{'):
                                        continue
                                    try:
                                        obj = json_lib.loads(line)
                                        if obj.get("type") == "solution":
                                            solution_dict = obj.get("output", {}).get("json", {})
                                        elif obj.get("type") == "status":
                                            status = obj.get("status", "UNKNOWN")
                                        elif obj.get("type") == "statistics":
                                            solve_time = obj.get("statistics", {}).get("solveTime", "N/A")
                                        elif obj.get("type") == "error":
                                            error_msg = obj.get("message", "Unknown error")
                                            print(f"[DEBUG] Solver error: {error_msg}")
                                    except json_lib.JSONDecodeError:
                                        pass

                                if proc.returncode != 0 and not solution_dict:
                                    error_detail = error_msg or proc.stderr or "Solver failed"
                                    print(f"[ERROR] Solver failed: {error_detail}")
                                    raise Exception(error_detail)

                                st.session_state.solver_status = status
                                st.session_state.solver_time = str(solve_time)
                                st.session_state.solver_result = solution_dict if solution_dict else None

                            else:
                                # Other solvers: use minizinc-python
                                model = minizinc.Model(mzn_path)
                                if has_dzn:
                                    model.add_file(dzn_path)

                                solver = minizinc.Solver.lookup(selected_solver)
                                instance = minizinc.Instance(solver, model)
                                result = instance.solve(timeout=timedelta(seconds=timeout_sec))

                                st.session_state.solver_status = str(result.status)
                                st.session_state.solver_time = f"{result.statistics.get('solveTime', 'N/A')}"

                                if result.solution is not None:
                                    solution_dict = {}
                                    for var in dir(result.solution):
                                        if not var.startswith('_'):
                                            try:
                                                solution_dict[var] = getattr(result.solution, var)
                                            except:
                                                pass
                                    st.session_state.solver_result = solution_dict
                                else:
                                    st.session_state.solver_result = None

                    except minizinc.MiniZincError as e:
                        print(f"[ERROR] MiniZincError: {e}")
                        import traceback
                        traceback.print_exc()
                        st.session_state.solver_status = "Error"
                        st.session_state.solver_result = str(e)
                        st.session_state.solver_time = None
                    except Exception as e:
                        print(f"[ERROR] Solver exception: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        st.session_state.solver_status = "Error"
                        st.session_state.solver_result = str(e)
                        st.session_state.solver_time = None

                # Map solver result to schedule table (LLM function calling)
                if (st.session_state.solver_result and
                    isinstance(st.session_state.solver_result, dict) and
                    st.session_state.solver_status not in ["Error", None]):
                    with st.spinner("Preparing schedule..."):
                        st.session_state.schedule_table = map_solver_result_with_llm(
                            st.session_state.solver_result,
                            st.session_state.mzn_code,
                            st.session_state.dzn_code
                        )

                # Auto-save after solving
                auto_save()
                st.rerun()

        st.divider()

        # Display results
        if st.session_state.solver_status:
            st.markdown(f"**Status:** `{st.session_state.solver_status}`")
            if st.session_state.solver_time:
                st.markdown(f"**Solve Time:** `{st.session_state.solver_time}`")

            if st.session_state.solver_result:
                if isinstance(st.session_state.solver_result, dict):
                    result = st.session_state.solver_result

                    # Show schedule table (mapped by LLM)
                    schedule_table = st.session_state.get("schedule_table", [])
                    if schedule_table:
                        df_schedule = pd.DataFrame(schedule_table)

                        # Ensure column order: Date, Shift, Machine, Side, MoldCd, Color, Size, Qty
                        column_order = ["Date", "Shift", "Machine", "Side", "MoldCd", "Color", "Size", "Qty"]
                        # Only include columns that exist
                        existing_cols = [c for c in column_order if c in df_schedule.columns]
                        df_schedule = df_schedule[existing_cols]

                        # Sort by: Date, Shift, Machine, Side, MoldCd, Color, Size
                        sort_cols = [c for c in ["Date", "Shift", "Machine", "Side", "MoldCd", "Color", "Size"] if c in df_schedule.columns]
                        if sort_cols:
                            df_schedule = df_schedule.sort_values(by=sort_cols).reset_index(drop=True)

                        # Visualization tabs
                        viz_tab1, viz_tab2 = st.tabs(["Schedule Table", "Raw Data"])

                        with viz_tab1:
                            st.dataframe(df_schedule, width='stretch', height=300)

                            # CSV Export
                            csv_data = df_schedule.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Schedule CSV",
                                data=csv_data,
                                file_name="schedule.csv",
                                mime="text/csv",
                                width='stretch'
                            )

                        with viz_tab2:
                            result_container = st.container(height=300)
                            with result_container:
                                for var, val in result.items():
                                    st.code(f"{var} = {val}")
                    else:
                        # No schedule table - show raw results only
                        st.info("No schedule table available")
                        result_container = st.container(height=350)
                        with result_container:
                            for var, val in result.items():
                                st.code(f"{var} = {val}")
                else:
                    st.error(st.session_state.solver_result)
        else:
            st.info("Generate a model first, then click Solve to run the solver.")

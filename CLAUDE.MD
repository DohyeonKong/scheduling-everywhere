# Scheduling, Everywhere - LLM-Based Optimization Modeling

## Research Goal

This project demonstrates a novel approach to optimization modeling:
**Domain experts can create optimization models using natural language,
without requiring optimization expertise.**

Traditional approach:
- Hire optimization expert
- Learn mathematical modeling
- Write hundreds of lines of code
- Weeks of development

Our approach:
- Domain expert describes the problem naturally
- LLM translates to optimization model (MiniZinc)
- Minutes instead of weeks

---

## Problem Domain: IP (Injection Phylon) Scheduling

### What is IP?
IP (Injection Phylon) is the sole/bottom part of shoes. IP production
happens before final assembly where upper and bottom parts are combined.

### Equipment Structure
```
┌─────────────────────────────────────────┐
│             IP Machine                  │
│  ┌─────────────────┬─────────────────┐  │
│  │   Left Side     │   Right Side    │  │
│  │  (Submachine)   │  (Submachine)   │  │
│  │                 │                 │  │
│  │  10 Stations    │  10 Stations    │  │
│  │  [Mold slots]   │  [Mold slots]   │  │
│  └─────────────────┴─────────────────┘  │
└─────────────────────────────────────────┘
```

- **Machine**: Has two sides (Left and Right)
- **Side/Submachine**: Works independently, 10 mold slots each
- **Mold**: Metal tool that shapes foam into specific shoe size
- **Shifts**: 3 per day (Morning, Afternoon, Night - 480 min each)

### Products
Each product identified by:
- Mold type (e.g., MS252801-1)
- Size (e.g., 6, 7, 8, 6T, 7T - "T" = half sizes)
- Color (e.g., BIP024_WHITE)

### Key Constraints
1. One color per side per shift
2. Color change = 120 min cleaning
3. Mold change = 15 min per mold (side stops)
4. Limited mold inventory
5. Max 10 molds per side
6. Meet all due dates

### Objective
Find minimum machines/side-shifts needed (compact operation)

---

## Project Structure

```
260126-SchedulingAgent/
├── CLAUDE.MD              # This file - project overview
├── DEVLOG.md              # Development log / research journey
├── test_prompt.py         # Standalone test script for prompt
├── prompt/
│   ├── system_prompt.txt  # LLM system instructions (structured output)
│   └── user_prompt.txt    # Domain expert's problem description
├── data/
│   ├── demand.csv         # 372 orders (MOLD, SIZE, COLOR, QTY, DUE_DATE)
│   ├── molds.csv          # 22 mold specs (MOLD, SIZE, INVENTORY, CT)
│   └── working_dates.csv  # 19 days calendar (THEDATE, HOLIDAY_YN)
├── output/                # Generated MiniZinc files (auto-created)
│   ├── response_*.md      # Full LLM responses
│   ├── model_*.mzn        # Generated model files
│   └── data_*.dzn         # Generated data files
└── demo_app/
    ├── app.py             # Streamlit demo application
    ├── requirements.txt   # Python dependencies
    ├── .env               # API key (edit with your key)
    ├── venv-macbook/      # Virtual environment for macOS
    └── chat_histories/    # Saved chat sessions (yyyymmddhhmmss_session.json)
```

### Data Summary

| File | Rows | Description |
|------|------|-------------|
| demand.csv | 372 | Orders for mold MS252801-1 |
| molds.csv | 22 | Mold inventory and cycle times |
| working_dates.csv | 19 | 2023-12-23 to 2024-01-10 calendar |

### Demo Data Details

**demand.csv**
- Single mold: MS252801-1
- 2 colors: BIP024_WHITE, BIP024_LTIRONORE
- 22 sizes: from 3T to 15
- Due dates: 2024-01-02 to 2024-01-06 (5 working days)

**molds.csv**
- 22 size variants
- Inventory: 2-7 molds per size
- Cycle time: 6.99 or 12.0 min/unit

**working_dates.csv**
- Starts from 2023-12-23 (5 working days before earliest due date)
- Ends at 2024-01-10 (buffer after latest due date)

### Data Scale Note

The demo data is intentionally reduced to ~372 orders across 5 due dates.
This allows the LLM to generate a complete `.dzn` file within token limits.
For production-scale problems (1000+ orders), use a preprocessing script
to generate the `.dzn` file separately.

---

## Demo App

**Title**: Scheduling, Everywhere [DEMO]
**Subtitle**: Scheduling Problem Modeling Agent (Powered by GPT-5.2)
**URL**: https://scheduling.idea-lab.tech (via Cloudflare Named Tunnel)

### App Features
- **Split-screen layout**: Chat on left, output tabs on right
- **Structured output**: LLM returns structured response (explanation, math_model, mzn_code, dzn_code)
- **Math Model tab**: Mathematical formulation with LaTeX rendering
- **Auto-loaded data**: CSV files from `data/` folder pre-loaded on startup
- **Pre-filled prompt**: User prompt from `prompt/user_prompt.txt` ready to send
- **Multiple chat histories**: Save/Load with timestamps (yyyymmddhhmmss_session)
- **IP tracking**: Records client IP for each session (via Cloudflare headers)
- **Auto-save**: Automatically saves after each LLM response
- **System prompt viewer**: Button to view the system prompt
- **Colored UI**: Sky blue theme for chat messages and code blocks
- **Download**: Export model.mzn and data.dzn files
- **Custom domain**: https://scheduling.idea-lab.tech via Cloudflare Named Tunnel

### UI Layout (Left Column)
1. Chat messages (colored: user=light sky blue, assistant=soft blue)
2. Text input (pre-filled with user_prompt.txt, grey background)
3. Send button
4. Data files expander (shows 3 auto-loaded CSVs)
5. System prompt viewer button
6. Save/Load/New Chat buttons

### UI Layout (Right Column)
1. **Math Model** tab - Mathematical formulation (LaTeX)
2. **MiniZinc** tab (with sub-tabs):
   - model.mzn - Constraint model code
   - data.dzn - Data file
3. Download buttons for each file

### Structured Output Format

The LLM returns a Pydantic model with four fields:

```python
class ModelingResponse(BaseModel):
    explanation: str           # Chat response (left column)
    math_model: Optional[str]  # Mathematical formulation (right > Math Model)
    mzn_code: Optional[str]    # MiniZinc model (right > MiniZinc > model.mzn)
    dzn_code: Optional[str]    # MiniZinc data (right > MiniZinc > data.dzn)
```

### Running the App

```bash
# 1. Navigate to demo_app folder
cd demo_app

# 2. Create virtual environment (macOS)
python3 -m venv venv-macbook

# 3. Activate virtual environment
source venv-macbook/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set API key in .env file
# Edit .env and replace with your key:
# OPENAI_API_KEY=sk-your-actual-key

# 6. Run the app
streamlit run app.py --server.port 8501
```

### Demo Flow (Streamlined)

1. **Start app** - Data files auto-loaded, prompt pre-filled
2. **Click Send** - LLM generates structured response
3. **View Math Model** - Mathematical formulation appears in right column
4. **View MiniZinc** - model.mzn and data.dzn in sub-tabs
5. **Follow-up** - Ask questions, refine model through conversation

### Cloudflare Named Tunnel (Custom Domain)

To serve the app via custom domain:

```bash
# One-time setup (already done)
cloudflared tunnel login
cloudflared tunnel create scheduling-demo
cloudflared tunnel route dns scheduling-demo scheduling.idea-lab.tech

# Run the tunnel
cloudflared tunnel run scheduling-demo
```

Config file at `~/.cloudflared/config.yml`:
```yaml
tunnel: <tunnel-id>
credentials-file: ~/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: scheduling.idea-lab.tech
    service: http://localhost:8501
  - service: http_status:404
```

---

## System Prompt Architecture

The system prompt uses **structured output** format:

### Output Fields
1. **explanation** - Conversational response (left column)
2. **math_model** - Mathematical formulation with LaTeX (right > Math Model)
3. **mzn_code** - MiniZinc model code (right > MiniZinc > model.mzn)
4. **dzn_code** - MiniZinc data file (right > MiniZinc > data.dzn)

### Math Model Format
Uses LaTeX with `$...$` (inline) and `$$...$$` (block):
- Sets, Parameters, Decision Variables
- Constraints with mathematical notation
- Objective function

---

## Test Script (CLI)

For quick testing without the Streamlit UI, use `test_prompt.py`:

```bash
# From project root
source demo_app/venv-macbook/bin/activate
python test_prompt.py
```

The script:
1. Loads prompts from `prompt/` folder
2. Loads data files from `data/` folder
3. Calls OpenAI API (gpt-5.2)
4. Extracts and saves `.mzn` and `.dzn` files to `output/` folder

---

## Key Messages for Audience

- Domain experts can now create optimization models
- Natural language → Mathematical model (DSL)
- Conversational refinement (not one-shot)
- Minutes instead of weeks
- Democratizing operations research

---

## Technical Notes

### Why MiniZinc?
- Declarative constraint modeling language
- Solver-agnostic (works with many solvers)
- Clean syntax suitable for LLM generation

### Model: GPT-5.2
- Used for both demo app and test script
- Structured output via Pydantic

### Why Structured Output?
- More reliable than regex extraction
- Separates explanation from code
- Enables Math Model tab alongside code

---

## Future Extensions

- Support other modeling languages (PuLP, OR-Tools, Pyomo)
- Add solution interpretation/visualization
- Multi-mold scheduling
- Integration with factory systems

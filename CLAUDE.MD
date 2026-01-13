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
├── test_prompt.py         # Standalone test script for prompt
├── prompt/
│   ├── system_prompt.txt  # LLM system instructions (UI-aware)
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
    └── chat_histories/    # Saved chat sessions with IP/timestamps (auto-generated)
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
**Subtitle**: Schedule Optimization Modeling Agent (Powered by GPT-5.2)

### App Features
- **Split-screen layout**: Chat on left, MiniZinc output on right
- **Auto-loaded data**: CSV files from `data/` folder pre-loaded on startup
- **Pre-filled prompt**: User prompt from `prompt/user_prompt.txt` ready to send
- **UI-aware LLM**: System prompt tells LLM about the split-screen layout
- **Code highlighting**: Syntax highlighting for MiniZinc
- **Multiple chat histories**: Save/Load multiple conversations with timestamps
- **IP tracking**: Records client IP for each session (via Cloudflare headers)
- **Auto-save**: Automatically saves after each LLM response
- **System prompt viewer**: Button to view the system prompt
- **Colored UI**: Sky blue theme for chat messages, input box, and code blocks
- **Download**: Export model.mzn and data.dzn files
- **Cloudflare Tunnel**: Share demo via public URL (quick tunnel)

### UI Layout (Left Column)
1. Chat messages (colored: user=light sky blue, assistant=soft blue)
2. Text input (pre-filled with user_prompt.txt, sky blue background)
3. Send button
4. Data files expander (shows 3 auto-loaded CSVs)
5. System prompt viewer button
6. Save/Load/New Chat buttons

### UI Layout (Right Column)
1. model.mzn tab with code display (soft blue background)
2. data.dzn tab with code display (soft blue background)
3. Download buttons for each file

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
streamlit run app.py
```

### Demo Flow (Streamlined)

1. **Start app** - Data files auto-loaded, prompt pre-filled
2. **Click Send** - LLM generates MiniZinc model
3. **View output** - model.mzn and data.dzn appear on right
4. **Follow-up** - Ask questions about the generated code
5. **Iterate** - Refine model through conversation

### Sharing via Cloudflare Tunnel

To share the demo with others (e.g., professors) without server setup:

```bash
# Install cloudflared (macOS)
brew install cloudflared

# Start app locally
streamlit run app.py --server.port 8501

# In another terminal, create quick tunnel
cloudflared tunnel --url http://localhost:8501
```

This generates a public URL like `https://xxx-xxx.trycloudflare.com` that anyone can access.

---

## System Prompt Architecture

The system prompt includes **UI Context** so the LLM understands:
- User sees chat on LEFT, generated code on RIGHT
- After code generation, user can see the .mzn/.dzn files
- Follow-up questions like "explain line 15" refer to visible code

### System Prompt Sections
1. **UI Context** - Split-screen layout awareness
2. **Output Format** - Code fence tags for extraction
3. **model.mzn structure** - Parameters, Variables, Constraints, Objective
4. **data.dzn format** - How to convert CSV data

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
- max_completion_tokens: 8192

---

## Future Extensions

- Support other modeling languages (PuLP, OR-Tools, Pyomo)
- Add solution interpretation/visualization
- Multi-mold scheduling
- Integration with factory systems

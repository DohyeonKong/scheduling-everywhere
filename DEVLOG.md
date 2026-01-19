# Development Log

## 2026-01-14 (Day 1)

### Done
- Built Streamlit demo app with split-screen layout (Chat + MiniZinc output)
- Auto-load CSV data files and pre-fill user prompt
- Integrated GPT-5.2 API for optimization modeling
- Added colored UI (sky blue theme for chat/code blocks)
- Multiple chat history support with IP tracking and timestamps
- System prompt viewer button
- Cloudflare tunnel setup for sharing demo
- Created GitHub repo: https://github.com/DohyeonKong/scheduling-everywhere

### To Do
- [x] Use structured output for LLM response (instead of regex extraction)
- [ ] UI improvements based on Prof. Cho's feedback
- [ ] Fix code panel update logic (prevent snippets from overwriting complete models)

---

## 2026-01-18 (Day 2)

### Done
- **Structured Output**: Implemented `client.responses.parse()` with Pydantic model for reliable JSON parsing
- **Gurobi Solver Integration**:
  - Configured Gurobi 13.0 with academic license
  - Used subprocess approach (minizinc-python can't pass --gurobi-dll flag)
  - Added `--json-stream` for proper output parsing
- **Solver UI**: Added solver selection dropdown (Gurobi default), timeout setting, result display
- **Schedule Display**: Parse product info from demand.csv (Mold, Size, Color) instead of "P1, P2..."
- **Data Simplification**:
  - Filtered demand.csv to first date only (103 rows → 32 products)
  - Simplified deadline: uniform `prod_deadline = num_shifts` for all products
  - Removed `output` block from templates (not needed with JSON mode)
- **Terminology Cleanup**:
  - "side-shifts" → "shifts"
  - Updated objective comments: "Compact plan - fewest machines, then fewest shifts"
- **Bug Analysis**:
  - Found `num_sides = 10` error (LLM confused slots with sides)
  - Found recurring array length mismatch errors (LLM can't count reliably)

### Key Issues Identified
1. **LLM Array Alignment Problem**: LLM consistently fails to generate arrays with matching lengths
   - Example: `prod_demand: 39`, `prod_to_moldsize: 70`, `prod_to_color: 42`
   - Root cause: LLMs are bad at precise counting

2. **Color Constraint Violations**: Model output showed multiple colors on same side/shift
   - Root cause: `num_sides = 10` (wrong) instead of `4` (2 machines × 2 sides)

### Decision: Python Preprocessing
- LLM cannot reliably count and align arrays
- Solution: Python generates .dzn (data), LLM generates .mzn (model structure)
- Benefits: Guaranteed correct array lengths, consistent data

### To Do (Next Session)
- [ ] Implement `generate_dzn_from_csvs()` function in Python
- [ ] Modify app.py to use Python-generated .dzn
- [ ] Update system prompt - LLM only generates .mzn
- [ ] Test complete flow with Gurobi

### Notes
- Demo strategy: Keep templates "secret" - show audience that LLM generates from natural language
- Templates ensure reliability but LLM is essentially a "template filler" not true modeler
- This is acceptable trade-off for demo reliability

---


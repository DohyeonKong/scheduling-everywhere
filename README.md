# Scheduling, Everywhere

**LLM-Based Optimization Modeling for Scheduling Problems**

> Domain experts can create optimization models using natural language, without requiring optimization expertise.

## Demo

**Live**: https://scheduling.idea-lab.tech

## Features

- Natural language → MiniZinc optimization model
- Split-screen UI: Chat (left) + Output (right)
- Structured output: Math Model + MiniZinc code
- LaTeX rendering for mathematical formulation

## Quick Start

```bash
cd demo_app
python3 -m venv venv-macbook
source venv-macbook/bin/activate
pip install -r requirements.txt

# Set your API key in .env
streamlit run app.py --server.port 8501
```

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: GPT-5.2 with structured output
- **DSL**: MiniZinc
- **Hosting**: Cloudflare Tunnel

## Documentation

See [CLAUDE.MD](CLAUDE.MD) for detailed documentation.

## License

MIT

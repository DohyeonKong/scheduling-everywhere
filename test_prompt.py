#!/usr/bin/env python3
"""
Test script for IP Scheduling Agent prompt.
Same configuration as demo_app/app.py but runs as a standalone script.
"""

import openai
import re
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Dict, Tuple

# Load environment variables from demo_app/.env
APP_DIR = Path(__file__).parent / "demo_app"
load_dotenv(APP_DIR / ".env")

# Paths
PROJECT_DIR = Path(__file__).parent
PROMPT_DIR = PROJECT_DIR / "prompt"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


def load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_file = PROMPT_DIR / "system_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text()
    raise FileNotFoundError(f"System prompt not found: {prompt_file}")


def load_user_prompt() -> str:
    """Load user prompt from file."""
    prompt_file = PROMPT_DIR / "user_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text()
    raise FileNotFoundError(f"User prompt not found: {prompt_file}")


def load_data_files() -> Dict[str, str]:
    """Load all CSV data files."""
    data_files = {}
    for csv_file in DATA_DIR.glob("*.csv"):
        data_files[csv_file.name] = csv_file.read_text()
    return data_files


def build_user_message(user_prompt: str, data_files: Dict[str, str]) -> str:
    """Build user message with attached data files."""
    message = user_prompt

    if data_files:
        message += "\n\n--- Uploaded Files ---\n"
        for name, content in data_files.items():
            message += f"\n### {name}\n```csv\n{content}\n```\n"

    return message


def extract_code_blocks(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract MiniZinc code blocks from response."""
    mzn_match = re.search(r'```(?:mzn|minizinc)\n(.*?)```', text, re.DOTALL)
    dzn_match = re.search(r'```dzn\n(.*?)```', text, re.DOTALL)

    mzn_code = mzn_match.group(1).strip() if mzn_match else None
    dzn_code = dzn_match.group(1).strip() if dzn_match else None

    return mzn_code, dzn_code


def save_output(response: str, mzn_code: Optional[str], dzn_code: Optional[str]):
    """Save output files with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full response
    response_file = OUTPUT_DIR / f"response_{timestamp}.md"
    response_file.write_text(response)
    print(f"Saved response to: {response_file}")

    # Save MiniZinc model
    if mzn_code:
        mzn_file = OUTPUT_DIR / f"model_{timestamp}.mzn"
        mzn_file.write_text(mzn_code)
        print(f"Saved model to: {mzn_file}")

    # Save data file
    if dzn_code:
        dzn_file = OUTPUT_DIR / f"data_{timestamp}.dzn"
        dzn_file.write_text(dzn_code)
        print(f"Saved data to: {dzn_file}")


def run_test():
    """Run the prompt test."""
    print("=" * 60)
    print("IP Scheduling Agent - Prompt Test")
    print("=" * 60)

    # Load prompts and data
    print("\nLoading prompts and data...")
    system_prompt = load_system_prompt()
    user_prompt = load_user_prompt()
    data_files = load_data_files()

    print(f"  - System prompt: {len(system_prompt)} chars")
    print(f"  - User prompt: {len(user_prompt)} chars")
    print(f"  - Data files: {list(data_files.keys())}")

    # Build messages
    user_message = build_user_message(user_prompt, data_files)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Call OpenAI API
    print("\nCalling OpenAI API (model: gpt-5.2)...")
    print("This may take a minute...")

    try:
        client = openai.OpenAI()

        response = client.chat.completions.create(
            model="gpt-5.2",
            max_completion_tokens=8192,
            messages=messages
        )

        assistant_response = response.choices[0].message.content

        # Extract code blocks
        mzn_code, dzn_code = extract_code_blocks(assistant_response)

        print("\n" + "=" * 60)
        print("RESPONSE")
        print("=" * 60)
        print(assistant_response)

        # Save outputs
        print("\n" + "=" * 60)
        print("SAVING OUTPUT FILES")
        print("=" * 60)
        save_output(assistant_response, mzn_code, dzn_code)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  - model.mzn generated: {'Yes' if mzn_code else 'No'}")
        print(f"  - data.dzn generated: {'Yes' if dzn_code else 'No'}")
        print(f"  - Usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} tokens")

    except openai.APIConnectionError:
        print("ERROR: Connection error. Check your internet connection.")
    except openai.AuthenticationError:
        print("ERROR: Authentication error. Check your OPENAI_API_KEY in demo_app/.env")
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    run_test()

"""
agent.py — the LLM data-analyst agent.

Given a chat history (list of user message strings, oldest -> newest), this
module runs a tool-calling loop against an OpenAI-compatible chat completions
endpoint (aipipe, OpenAI, etc.), lets the model use two tools:

  - run_python(code): execute pandas/numpy python and get back stdout + a
    `result` variable, for any inline data / calculations.
  - fetch_url(url): fetch a public URL (MOSPI page, CSV, JSON, HTML) and get
    back cleaned text, for pulling in public datasets.

...until the model returns a final JSON object of the shape {"answer": ...}.
Every step (user turn, LLM request/response, tool call, tool result, final
answer) is appended to an in-memory log, which is written out as JSONL and
pushed to a public GitHub Gist so we get a free, permanent, wget-able
log_url for every run.

Environment variables:
  OPENAI_API_KEY / AIPIPE_TOKEN   - API key for the LLM endpoint (either works)
  OPENAI_BASE_URL                 - defaults to aipipe's OpenAI-compatible route
  MODEL_NAME                      - defaults to "gpt-4o-mini"
  GITHUB_TOKEN                    - a GitHub PAT with "gist" scope, for log upload
  LOG_DIR                         - local dir to also save logs, default "./logs"
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone

import requests
from openai import OpenAI

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("AIPIPE_TOKEN")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "8"))
TOOL_TIMEOUT_SECONDS = int(os.environ.get("TOOL_TIMEOUT_SECONDS", "20"))

os.makedirs(LOG_DIR, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """You are a meticulous data analyst agent answering a single
data-analysis question sent to you over Telegram.

You have two tools:
- run_python: execute Python (pandas, numpy available) to parse inline data,
  compute statistics, filter/aggregate tables, etc. Print anything you want
  to inspect, and set a variable named `result` to whatever value you want
  returned to you.
- fetch_url: fetch a public URL (a MOSPI page, a CSV/JSON data file, a
  Wikipedia page, etc.) and get back its text content so you can read or
  parse it (combine with run_python to parse fetched CSV/JSON/HTML text).

Work step by step:
1. Read the question carefully, including the EXACT shape it wants the
   answer in (it usually shows a literal example JSON object).
2. Use fetch_url to pull any public dataset you need, and run_python to
   compute the answer. Do not guess or hallucinate numbers you could
   compute or look up.
3. When you are fully done, respond with ONLY a single JSON object of the
   form {"answer": <value>} where <value> is shaped EXACTLY as the question
   requested (same keys, same nesting, same types). Do not include any
   other keys, prose, markdown fences, or explanation in that final message.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code. pandas (pd), numpy (np), json, re, "
                "math, and io are pre-imported. Print intermediate values "
                "for your own inspection. Set a variable named `result` to "
                "the value you want returned to you as the tool output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a public URL and return its text content (HTML is "
                "stripped down to readable text; CSV/JSON/plain text is "
                "returned as-is, truncated if very large)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"],
            },
        },
    },
]


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------

class _Timeout:
    """Best-effort wall-clock timeout for run_python (Unix only)."""

    def __init__(self, seconds):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise TimeoutError(f"run_python exceeded {self.seconds}s")

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)

    def __exit__(self, *exc):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def run_python(code: str) -> dict:
    """Execute python code in a semi-sandboxed namespace, return stdout + result."""
    import pandas as pd
    import numpy as np

    safe_globals = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "json": json,
        "re": re,
        "io": io,
        "math": __import__("math"),
    }
    local_vars: dict = {}
    stdout_buf = io.StringIO()
    error = None
    try:
        with _Timeout(TOOL_TIMEOUT_SECONDS):
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, safe_globals, local_vars)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    result = local_vars.get("result", None)
    # Make sure it's JSON-serializable; fall back to str() if not.
    try:
        json.dumps(result, default=str)
    except TypeError:
        result = str(result)

    return {
        "stdout": stdout_buf.getvalue()[-4000:],
        "result": result,
        "error": error,
    }


def fetch_url(url: str) -> dict:
    """Fetch a URL and return cleaned text content."""
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; data-analyst-bot/1.0)"},
        )
        content_type = resp.headers.get("Content-Type", "")
        text = resp.text

        if "html" in content_type.lower() or text.strip().startswith("<"):
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n"))
            except Exception:  # noqa: BLE001
                pass

        return {
            "status_code": resp.status_code,
            "content_type": content_type,
            "text": text[:20000],
            "truncated": len(text) > 20000,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


TOOL_IMPLS = {"run_python": run_python, "fetch_url": fetch_url}


# --------------------------------------------------------------------------
# Logging + Gist upload
# --------------------------------------------------------------------------

class RunLogger:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.lines: list[dict] = []

    def log(self, event_type: str, data: dict):
        self.lines.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "type": event_type,
                "data": data,
            }
        )

    def as_jsonl(self) -> str:
        return "\n".join(json.dumps(line, default=str) for line in self.lines) + "\n"

    def save_local(self) -> str:
        path = os.path.join(LOG_DIR, f"{self.run_id}.jsonl")
        with open(path, "w") as f:
            f.write(self.as_jsonl())
        return path


def push_log_to_gist(logger: RunLogger) -> str:
    """Upload the run log as a public Gist, return the raw file URL.

    Falls back to a local file path (not public!) if GITHUB_TOKEN is unset
    or the upload fails, so the caller always gets *something* back.
    """
    local_path = logger.save_local()

    if not GITHUB_TOKEN:
        return f"file://{os.path.abspath(local_path)}"

    try:
        resp = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "description": f"data-analyst-bot run log {logger.run_id}",
                "public": True,
                "files": {"run.jsonl": {"content": logger.as_jsonl()}},
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw_url = resp.json()["files"]["run.jsonl"]["raw_url"]
        return raw_url
    except Exception as e:  # noqa: BLE001
        logger.log("gist_upload_error", {"error": str(e)})
        return f"file://{os.path.abspath(local_path)}"


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def _extract_json_object(text: str):
    """Pull the first top-level JSON object out of a model response."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: find the first balanced {...} span.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def answer_question(chat_history: list[str]) -> dict:
    """Run the agent over a chat history, return {"answer":..., "log_url":...}."""
    run_id = uuid.uuid4().hex[:12]
    logger = RunLogger(run_id)
    logger.log("chat_history", {"messages": chat_history})

    user_content = (
        "Conversation so far (oldest to newest):\n\n"
        + "\n---\n".join(chat_history[:-1])
        + ("\n===\nAnswer this final message:\n" if len(chat_history) > 1 else "")
        + chat_history[-1]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    final_answer = None
    error_note = None

    for step in range(MAX_ITERATIONS):
        logger.log("llm_request", {"step": step, "messages": messages[-1:]})
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            logger.log("llm_error", {"step": step, "error": str(e)})
            error_note = str(e)
            break

        msg = resp.choices[0].message
        logger.log(
            "llm_response",
            {
                "step": step,
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])],
            },
        )

        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = TOOL_IMPLS.get(fn_name)
                logger.log("tool_call", {"step": step, "name": fn_name, "args": args})
                if impl is None:
                    tool_result = {"error": f"unknown tool {fn_name}"}
                else:
                    tool_result = impl(**args)
                logger.log("tool_result", {"step": step, "name": fn_name, "result": tool_result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, default=str)[:8000],
                    }
                )
            continue

        # No tool calls -> model believes it's done.
        parsed = _extract_json_object(msg.content or "")
        if parsed is not None and "answer" in parsed:
            final_answer = parsed["answer"]
            break
        elif parsed is not None:
            # Model returned JSON without an "answer" wrapper; use as-is.
            final_answer = parsed
            break
        else:
            # Nudge the model to comply with the output contract.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Respond with ONLY a JSON object of the form "
                        '{"answer": <value>} and nothing else.'
                    ),
                }
            )

    if final_answer is None:
        final_answer = {"error": error_note or "agent did not produce a final answer"}

    logger.log("final_answer", {"answer": final_answer})
    log_url = push_log_to_gist(logger)
    logger.log("log_uploaded", {"log_url": log_url})
    # Re-upload once more so the log itself records its own URL (best effort).
    log_url = push_log_to_gist(logger)

    return {"answer": final_answer, "log_url": log_url}


if __name__ == "__main__":
    # Quick manual test:
    #   OPENAI_API_KEY=... GITHUB_TOKEN=... python agent.py
    q = (
        'What is the mean of [4, 8, 15, 16, 23, 42]? '
        'Reply with ONLY this JSON object and nothing else: '
        '{"answer": {"mean": <number>}, "log_url": "<url>"}'
    )
    out = answer_question([q])
    print(json.dumps(out, indent=2, default=str))

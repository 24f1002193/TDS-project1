# Telegram Data-Analyst Bot

An LLM agent, wired up to a Telegram bot, that answers data-analysis
questions (inline data or public datasets like MOSPI) and replies with a
single JSON object:

```json
{"answer": <shaped as the question asks>, "log_url": "https://gist.githubusercontent.com/.../raw/run.jsonl"}
```

## Architecture

```
Telegram message → bot.py (long-polling)
                       → agent.py: tool-calling loop against an
                         OpenAI-compatible LLM (aipipe/OpenAI)
                           ├─ tool: run_python   (pandas/numpy on inline data)
                           └─ tool: fetch_url    (pull public datasets/pages)
                       → every step logged to JSONL
                       → log pushed to a public GitHub Gist → log_url
                    → bot replies with {"answer": ..., "log_url": ...}
```

Polling (not webhooks) means no public domain/HTTPS setup is needed — the
process just has to stay running. Logs go to a Gist so `log_url` is free,
public, and `wget`-able with zero extra infra.

## Setup

1. Create the bot: message `@BotFather` on Telegram → `/newbot` → pick a
   username ending in `bot`. Save the token.
2. Create a GitHub PAT with the **gist** scope (Settings → Developer
   settings → Personal access tokens) so the agent can publish logs.
3. Get an LLM API key — either an aipipe token, or an OpenAI key.
4. Copy `.env.example` to `.env` and fill in the values.

```bash
git clone <this repo>
cd telegram-data-analyst-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env
export $(cat .env | xargs)   # or use a process manager that loads .env
python bot.py
```

Test the agent alone (no Telegram needed):

```bash
python agent.py
```


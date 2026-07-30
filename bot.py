"""
bot.py — Telegram polling bot for the data-analyst agent.

Uses long-polling (no public HTTPS endpoint / webhook needed), so all it
needs to do to satisfy grading is stay running somewhere. Keeps a short
per-chat message history in memory so multi-turn questions have context,
and replies to every incoming text message with exactly one JSON object
(no extra prose), as the grader requires.
"""

import json
import logging
import os
from collections import defaultdict

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

import agent

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger("bot")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))

# In-memory per-chat history: chat_id -> list[str] of user messages.
# Fine for a single-process bot; swap for Redis/SQLite if you need
# persistence across restarts.
chat_histories: dict[int, list[str]] = defaultdict(list)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or not message.text:
        return

    chat_id = message.chat_id
    text = message.text.strip()
    logger.info("chat=%s received: %s", chat_id, text[:200])

    history = chat_histories[chat_id]
    history.append(text)
    if len(history) > MAX_HISTORY:
        del history[: len(history) - MAX_HISTORY]

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        result = agent.answer_question(list(history))
    except Exception as e:  # noqa: BLE001
        logger.exception("agent failed")
        result = {"answer": None, "log_url": "", "error": str(e)}

    reply_text = json.dumps(result, default=str)
    logger.info("chat=%s replying: %s", chat_id, reply_text[:500])
    await message.reply_text(reply_text)

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # silence default request logging, keeps logs clean


def _start_health_server():
    port = int(os.environ.get("PORT", "10000"))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    logger.info(f"health server listening on :{port}")
    server.serve_forever()

def main() -> None:
    threading.Thread(target=_start_health_server, daemon=True).start()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("bot starting (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()

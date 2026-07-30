FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Polling bot — no port to expose, just needs to keep running.
CMD ["python", "bot.py"]

#!/bin/bash
# Sends one chat request to the server started by 06_serve.sh.
source "$(dirname "$0")/config.sh"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Сколько будет 17 * 23? Ответь одним числом."}],
  "max_tokens": 64, "temperature": 0
}' | python3 -m json.tool

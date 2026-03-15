import json
from typing import Dict


def format_sse(event: Dict) -> str:
    event_type = event.get("event_type", "message")
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"

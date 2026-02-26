import json
import os
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

LOG_FILE = "/app/logs/rag_events.jsonl"

class EventLogger:
    @staticmethod
    async def log_event(event_data: dict):
        """Writes evaluation telemetry to a JSONL file asynchronously."""
        def write_log():
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                event_data["timestamp"] = datetime.utcnow().isoformat()
                f.write(json.dumps(event_data) + "\n")
        
        try:
            await asyncio.to_thread(write_log)
        except Exception as e:
            logger.error(f"Failed to write event log: {e}")

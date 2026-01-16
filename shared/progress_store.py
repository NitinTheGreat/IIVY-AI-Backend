"""In-memory progress store for real-time polling."""
import time
from typing import Dict, Any, Optional

_STORE: Dict[str, Dict[str, Any]] = {}

def update_progress(project_id: str, step: str, percent: int, details: dict = None):
    _STORE[project_id] = {
        "project_id": project_id,
        "status": "completed" if percent >= 100 else "indexing",
        "percent": percent,
        "step": step,
        "details": details or {"thread_count": 0, "message_count": 0, "pdf_count": 0},
        "updated_at": time.time()
    }

def get_progress(project_id: str) -> Optional[Dict[str, Any]]:
    return _STORE.get(project_id)

def clear_progress(project_id: str):
    _STORE.pop(project_id, None)

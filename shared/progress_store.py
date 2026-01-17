"""In-memory progress store for real-time polling."""
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_STORE: Dict[str, Dict[str, Any]] = {}


def update_progress(project_id: str, phase: str, step: str, percent: int, details: dict = None):
    """
    Update progress for a project.
    
    Args:
        project_id: Unique project identifier (includes user hash)
        phase: Current phase name (e.g., "Searching", "Processing PDFs", "Creating Embeddings")
        step: User-friendly message describing current step
        percent: Progress percentage (0-100)
        details: Optional dict with thread_count, message_count, pdf_count
    """
    status = "completed" if percent >= 100 else ("indexing" if percent <= 80 else "vectorizing")
    
    _STORE[project_id] = {
        "project_id": project_id,
        "status": status,
        "phase": phase,
        "step": step,
        "percent": percent,
        "details": details or {"thread_count": 0, "message_count": 0, "pdf_count": 0},
        "updated_at": time.time()
    }
    
    # Debug log every update
    logger.info(f"PROGRESS_STORE: {project_id} -> {percent}% [{status}] {phase}: {step}")


def get_progress(project_id: str) -> Optional[Dict[str, Any]]:
    return _STORE.get(project_id)


def clear_progress(project_id: str):
    _STORE.pop(project_id, None)


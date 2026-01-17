import sys, os
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import azure.functions as func
from shared import create_response, create_error_response
from shared.progress_store import get_progress, _STORE

logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    project_id = req.params.get("project_id")
    if not project_id:
        return create_error_response("Missing project_id")
    
    # Debug: Log what's being requested vs what's in store
    store_keys = list(_STORE.keys())
    logger.info(f"GET_STATUS: Requested '{project_id}', Store has: {store_keys}")
    
    progress = get_progress(project_id)
    if not progress:
        # Check if there's a similar key (partial match)
        similar = [k for k in store_keys if project_id in k or k in project_id]
        if similar:
            logger.warning(f"GET_STATUS: '{project_id}' not found but similar keys exist: {similar}")
        
        return create_response({
            "project_id": project_id,
            "status": "not_found",
            "percent": 0,
            "phase": "",
            "step": "No active indexing",
            "details": {"thread_count": 0, "message_count": 0, "pdf_count": 0},
            "updated_at": 0,
            "_debug_available_ids": store_keys  # Help frontend debug
        })
    
    return create_response(progress)

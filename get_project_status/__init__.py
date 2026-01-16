import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import azure.functions as func
from shared import create_response, create_error_response
from shared.progress_store import get_progress

def main(req: func.HttpRequest) -> func.HttpResponse:
    project_id = req.params.get("project_id")
    if not project_id:
        return create_error_response("Missing project_id")
    
    progress = get_progress(project_id)
    if not progress:
        return create_response({
            "project_id": project_id,
            "status": "not_found",
            "percent": 0,
            "step": "No active indexing",
            "details": {"thread_count": 0, "message_count": 0, "pdf_count": 0},
            "updated_at": 0
        })
    return create_response(progress)

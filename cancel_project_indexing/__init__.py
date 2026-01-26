"""
Azure Function: cancel_project_indexing

HTTP Trigger to request cancellation of an in-progress indexing operation.
Sets a cancellation flag that the indexing/vectorization loops check periodically.
"""
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response
from shared.progress_store import request_cancel, get_progress, update_progress


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Request cancellation of a project indexing operation.
    
    POST /api/cancel_project_indexing?project_id=88_supermarket_a1b2c3d4
    
    Query Parameters:
        project_id (required): The project ID to cancel
    
    Returns:
    {
        "status": "cancel_requested",
        "project_id": "88_supermarket_a1b2c3d4",
        "message": "Cancellation requested. The indexing will stop shortly."
    }
    """
    logging.info('cancel_project_indexing: Processing request')
    
    # Get required query parameter
    project_id = req.params.get('project_id')
    if not project_id:
        return create_error_response("Missing required parameter: project_id")
    
    try:
        # Check if there's an active indexing operation for this project
        current_progress = get_progress(project_id)
        
        if not current_progress:
            return create_error_response(
                f"No active indexing found for project: {project_id}",
                status_code=404
            )
        
        # Check if already completed or cancelled
        current_status = current_progress.get("status", "")
        if current_status == "completed":
            return create_error_response(
                f"Project indexing already completed: {project_id}",
                status_code=400
            )
        
        # Set the cancellation flag
        request_cancel(project_id)
        
        # Update progress to show cancellation pending
        update_progress(
            project_id,
            "Cancelling",
            "Cancellation requested, stopping...",
            current_progress.get("percent", 0),
            current_progress.get("details", {})
        )
        
        logging.info(f'cancel_project_indexing: Cancellation requested for "{project_id}"')
        
        return create_response({
            "status": "cancel_requested",
            "project_id": project_id,
            "message": "Cancellation requested. The indexing will stop shortly."
        })
        
    except Exception as e:
        logging.error(f'cancel_project_indexing: Error - {str(e)}')
        return create_error_response(
            f"Failed to cancel project: {str(e)}",
            status_code=500
        )

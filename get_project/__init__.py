"""
Azure Function: get_project

HTTP Trigger to get details of a specific project.
"""
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response
from project_manager import ProjectManager


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get details of a specific project.
    
    GET /api/get_project?project_id=88_supermarket
    
    Query Parameters:
        project_id (required): The project ID to look up
    
    Returns:
    {
        "project_id": "88_supermarket",
        "project_name": "88 SuperMarket",
        "user_email": "harv@example.com",
        "indexed_at": "2024-01-15T10:30:00Z",
        "thread_count": 45,
        "message_count": 123,
        "vector_count": 450,
        "is_indexed": true
    }
    """
    logging.info('get_project: Processing request')
    
    # Get required query parameter
    project_id = req.params.get('project_id')
    if not project_id:
        return create_error_response("Missing required parameter: project_id")
    
    try:
        # Create manager instance
        manager = ProjectManager()
        
        # Get project info
        project = manager.get_project(project_id)
        
        if not project:
            return create_error_response(
                f"Project not found: {project_id}",
                status_code=404
            )
        
        return create_response({
            "project_id": project.project_id,
            "project_name": project.project_name,
            "user_email": project.user_email,
            "indexed_at": project.indexed_at,
            "last_email_timestamp": project.last_email_timestamp,
            "thread_count": project.thread_count,
            "message_count": project.message_count,
            "attachment_count": project.attachment_count,
            "vector_count": project.vector_count,
            "is_indexed": project.is_indexed
        })
        
    except Exception as e:
        logging.error(f'get_project: Error - {str(e)}')
        return create_error_response(
            f"Failed to get project: {str(e)}",
            status_code=500
        )

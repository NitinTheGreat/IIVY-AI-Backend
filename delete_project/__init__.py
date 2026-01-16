"""
Azure Function: delete_project

HTTP Trigger to delete a project (vectors from Pinecone + registry entry).
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
    Delete a project.
    
    DELETE /api/delete_project?project_id=88_supermarket
    
    Query Parameters:
        project_id (required): The project ID to delete
    
    Returns:
    {
        "status": "deleted",
        "project_id": "88_supermarket"
    }
    """
    logging.info('delete_project: Processing request')
    
    # Get required query parameter
    project_id = req.params.get('project_id')
    if not project_id:
        return create_error_response("Missing required parameter: project_id")
    
    try:
        # Create manager instance
        manager = ProjectManager()
        
        # Check if project exists
        project = manager.get_project(project_id)
        if not project:
            return create_error_response(
                f"Project not found: {project_id}",
                status_code=404
            )
        
        # Delete project
        logging.info(f'delete_project: Deleting project "{project_id}"')
        manager.delete_project(project_id)
        
        return create_response({
            "status": "deleted",
            "project_id": project_id
        })
        
    except Exception as e:
        logging.error(f'delete_project: Error - {str(e)}')
        return create_error_response(
            f"Failed to delete project: {str(e)}",
            status_code=500
        )

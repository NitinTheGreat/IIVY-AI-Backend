"""
Azure Function: list_projects

HTTP Trigger to list all indexed projects.
"""
import logging
import sys
import os
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response
from project_manager import ProjectManager


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    List all indexed projects.
    
    GET /api/list_projects?user_email=harv@example.com
    
    Query Parameters:
        user_email (optional): Filter projects by user email
    
    Returns:
    {
        "projects": [
            {
                "project_id": "88_supermarket",
                "project_name": "88 SuperMarket",
                "user_email": "harv@example.com",
                "indexed_at": "2024-01-15T10:30:00Z",
                "thread_count": 45,
                "message_count": 123,
                "vector_count": 450
            }
        ]
    }
    """
    logging.info('list_projects: Processing request')
    
    # Get optional query parameter
    user_email = req.params.get('user_email')
    
    try:
        # Create manager instance
        manager = ProjectManager()
        
        # List projects
        projects = manager.list_projects(user_email=user_email)
        
        # Convert to serializable format
        projects_list = []
        for p in projects:
            projects_list.append({
                "project_id": p.project_id,
                "project_name": p.project_name,
                "user_email": p.user_email,
                "indexed_at": p.indexed_at,
                "last_email_timestamp": p.last_email_timestamp,
                "thread_count": p.thread_count,
                "message_count": p.message_count,
                "attachment_count": p.attachment_count,
                "vector_count": p.vector_count,
                "is_indexed": p.is_indexed
            })
        
        return create_response({
            "projects": projects_list,
            "count": len(projects_list)
        })
        
    except Exception as e:
        logging.error(f'list_projects: Error - {str(e)}')
        return create_error_response(
            f"Failed to list projects: {str(e)}",
            status_code=500
        )

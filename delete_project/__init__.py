"""
Azure Function: delete_project

HTTP Trigger to delete a project's vectors from Pinecone.
This is the cleanup endpoint called by database backend when a project is deleted.

The AI backend owns Pinecone, so only it can delete vectors.
"""
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func
from pinecone import Pinecone

from shared import create_response, create_error_response

# Configuration
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "donna-email")


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Delete a project's vectors from Pinecone.
    
    DELETE /api/delete_project?project_id=88_supermarket_abc123
    
    Also accepts POST with JSON body:
    {
        "project_id": "88_supermarket_abc123",
        "user_email": "john@example.com"  // Optional: for Supabase storage cleanup
    }
    
    Query Parameters:
        project_id (required): The project ID (Pinecone namespace) to delete
    
    Returns:
    {
        "status": "deleted",
        "project_id": "88_supermarket_abc123",
        "vectors_deleted": true,
        "storage_deleted": true
    }
    """
    logging.info('delete_project: Processing request')
    
    # Get project_id from query params or body
    project_id = req.params.get('project_id')
    user_email = req.params.get('user_email')
    
    # Try to get from body if not in query params
    if not project_id:
        try:
            body = req.get_json()
            project_id = body.get('project_id')
            user_email = body.get('user_email') or user_email
        except:
            pass
    
    if not project_id:
        return create_error_response("Missing required parameter: project_id")
    
    result = {
        "status": "deleted",
        "project_id": project_id,
        "vectors_deleted": False,
        "storage_deleted": False
    }
    
    try:
        # ============================================================
        # Step 1: Delete from Pinecone
        # ============================================================
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        
        if pinecone_api_key:
            try:
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index(PINECONE_INDEX_NAME)
                
                # Delete all vectors in this namespace
                index.delete(delete_all=True, namespace=project_id)
                
                logging.info(f'delete_project: Deleted Pinecone namespace "{project_id}"')
                result["vectors_deleted"] = True
                
            except Exception as e:
                logging.warning(f'delete_project: Pinecone deletion warning - {str(e)}')
                # Don't fail the whole request if Pinecone deletion has issues
                # The namespace might not exist, which is fine
        else:
            logging.warning('delete_project: PINECONE_API_KEY not configured')
        
        # ============================================================
        # Step 2: Delete from Supabase Storage (if user_email provided)
        # ============================================================
        if user_email:
            try:
                from shared.supabase_storage import delete_project_data
                success = delete_project_data(user_email, project_id)
                result["storage_deleted"] = success
                
                if success:
                    logging.info(f'delete_project: Deleted Supabase storage for "{project_id}"')
                else:
                    logging.warning(f'delete_project: Could not delete Supabase storage for "{project_id}"')
                    
            except Exception as e:
                logging.warning(f'delete_project: Supabase deletion warning - {str(e)}')
        else:
            logging.info('delete_project: No user_email provided, skipping Supabase storage cleanup')
        
        return create_response(result)
        
    except Exception as e:
        logging.error(f'delete_project: Error - {str(e)}')
        return create_error_response(
            f"Failed to delete project: {str(e)}",
            status_code=500
        )

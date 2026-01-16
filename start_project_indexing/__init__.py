"""
Azure Function: start_project_indexing

HTTP Trigger that wraps ProjectIndexer to index emails and PDFs for a project.
"""
import logging
import sys
import os
import io
from dataclasses import asdict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response, parse_request_body, get_required_param
from project_indexer import ProjectIndexer


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Start indexing a project.
    
    POST Body:
    {
        "project_name": "88 SuperMarket",
        "user_email": "harv@example.com",
        "gmail_credentials": { "access_token": "...", "refresh_token": "...", ... },
        "max_threads": 50  // Optional: limit number of threads to index
    }
    
    Returns:
    {
        "status": "completed",
        "project_id": "88_supermarket",
        "stats": { ... },
        "threads_data": { ... },  // Full threads JSON for vectorization
        "attachments_data": { ... }  // Full attachments JSON
    }
    """
    logging.info('start_project_indexing: Processing request')
    
    # Parse request body
    body = parse_request_body(req)
    if not body:
        return create_error_response("Request body must be valid JSON")
    
    # Get required parameters
    project_name, err = get_required_param(body, "project_name")
    if err:
        return err
    
    user_email, err = get_required_param(body, "user_email")
    if err:
        return err
    
    gmail_credentials, err = get_required_param(body, "gmail_credentials")
    if err:
        return err
    
    # Optional parameters
    max_threads = body.get("max_threads")
    
    try:
        # Create the indexer
        indexer = ProjectIndexer(
            project_name=project_name,
            user_email=user_email,
            provider="gmail",
            credentials=gmail_credentials,
            max_threads=max_threads
        )
        
        # Run indexing
        logging.info(f'start_project_indexing: Starting index for project "{project_name}"')
        project_index, attachment_index = indexer.index_project()
        
        # Convert dataclasses to dicts for JSON serialization
        threads_data = asdict(project_index)
        attachments_data = asdict(attachment_index)
        
        # Return the full data (frontend can pass to vectorize_project)
        return create_response({
            "status": "completed",
            "project_id": project_index.project_id,
            "project_name": project_index.project_name,
            "stats": {
                "thread_count": project_index.thread_count,
                "message_count": project_index.message_count,
                "pdf_count": project_index.pdf_count,
                "indexed_at": project_index.indexed_at
            },
            "threads_data": threads_data,
            "attachments_data": attachments_data
        })
        
    except Exception as e:
        logging.error(f'start_project_indexing: Error - {str(e)}')
        return create_error_response(
            f"Indexing failed: {str(e)}",
            status_code=500
        )

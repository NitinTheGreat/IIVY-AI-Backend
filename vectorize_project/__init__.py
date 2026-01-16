"""
Azure Function: vectorize_project

HTTP Trigger that wraps ProjectVectorizer to create embeddings and store in Pinecone.
Supports fetching threads_data from Supabase Storage if not provided directly.
"""
import logging
import sys
import os
import json
import tempfile
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response, parse_request_body, get_required_param
from shared import download_threads_data
from project_vectorizer import ProjectVectorizer


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Vectorize a project's indexed data.
    
    POST Body (Option 1 - Direct data):
    {
        "project_id": "88_supermarket",
        "threads_data": { ... }  // The threads_data returned from start_project_indexing
    }
    
    POST Body (Option 2 - Fetch from Supabase):
    {
        "project_id": "88_supermarket",
        "user_email": "harv@example.com"  // Used to fetch from Supabase
    }
    
    Returns:
    {
        "status": "completed",
        "project_id": "88_supermarket",
        "vectors_created": 450,
        "message_chunks": 200,
        "attachment_chunks": 250,
        "duration_seconds": 12.5
    }
    """
    logging.info('vectorize_project: Processing request')
    
    # Parse request body
    body = parse_request_body(req)
    if not body:
        return create_error_response("Request body must be valid JSON")
    
    # Get required parameters
    project_id, err = get_required_param(body, "project_id")
    if err:
        return err
    
    # Get threads_data - either directly provided or fetch from Supabase
    threads_data = body.get("threads_data")
    
    if not threads_data:
        # Try to fetch from Supabase using user_email
        user_email = body.get("user_email")
        if not user_email:
            return create_error_response(
                "Either 'threads_data' or 'user_email' must be provided. "
                "user_email is used to fetch data from Supabase Storage."
            )
        
        logging.info(f'vectorize_project: Fetching threads_data from Supabase for {project_id}')
        try:
            threads_data = download_threads_data(user_email, project_id)
            logging.info(f'vectorize_project: Successfully fetched threads_data from Supabase')
        except FileNotFoundError as e:
            return create_error_response(
                f"threads_data not found in Supabase Storage for project '{project_id}'. "
                "Run start_project_indexing or index_and_vectorize first.",
                status_code=404
            )
        except Exception as e:
            return create_error_response(
                f"Failed to fetch threads_data from Supabase: {str(e)}",
                status_code=500
            )
    
    try:
        # Create vectorizer
        vectorizer = ProjectVectorizer()
        
        # Debug: Log the structure of threads_data
        logging.info(f"threads_data keys: {threads_data.keys() if isinstance(threads_data, dict) else 'NOT A DICT'}")
        if isinstance(threads_data, dict):
            logging.info(f"thread_docs count: {len(threads_data.get('thread_docs', []))}")
            logging.info(f"thread_count field: {threads_data.get('thread_count', 'MISSING')}")
        
        # The current vectorizer expects a file path, so we create a temp file
        # TODO: In future, modify ProjectVectorizer to accept dict directly
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(threads_data, f, default=str)
            temp_path = f.name
        
        try:
            # Run vectorization
            logging.info(f'vectorize_project: Starting vectorization for "{project_id}"')
            result = vectorizer.vectorize_project(temp_path)
            
            return create_response({
                "status": "completed",
                "project_id": result.project_id,
                "namespace": result.namespace,
                "vectors_created": result.total_vectors,
                "message_chunks": result.message_chunks,
                "attachment_chunks": result.attachment_chunks,
                "duration_seconds": result.duration_seconds
            })
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logging.error(f'vectorize_project: Error - {str(e)}')
        return create_error_response(
            f"Vectorization failed: {str(e)}",
            status_code=500
        )

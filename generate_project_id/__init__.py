"""
Azure Function: generate_project_id

Fast endpoint that returns the project_id immediately.
Frontend calls this FIRST to get the ID, then starts polling, 
then calls index_and_vectorize.

POST /api/generate_project_id
{
    "project_name": "Microsoft Azure",
    "user_email": "nitin.p@buildsmartr.com"
}

Returns:
{
    "project_id": "microsoft_azure_a7e1423f",
    "project_name": "Microsoft Azure"
}
"""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func
from shared import create_response, create_error_response, parse_request_body, get_required_param
from project_indexer import _generate_project_id

logger = logging.getLogger(__name__)


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Generate and return project_id immediately."""
    logging.info('generate_project_id: Processing request')
    
    # Parse request
    body = parse_request_body(req)
    if not body:
        return create_error_response("Request body must be valid JSON")
    
    # Get required params
    project_name, err = get_required_param(body, "project_name")
    if err:
        return err
    
    user_email, err = get_required_param(body, "user_email")
    if err:
        return err
    
    # Generate the project_id (same logic as index_and_vectorize uses)
    project_id = _generate_project_id(project_name, user_email)
    
    logger.info(f"generate_project_id: '{project_name}' + '{user_email}' -> '{project_id}'")
    
    return create_response({
        "project_id": project_id,
        "project_name": project_name
    })

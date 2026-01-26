"""
Shared utilities for Azure Functions.
"""
import json
import logging
import sys
import os

# Add parent directory to path so we can import from main project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class CancellationError(Exception):
    """
    Raised when indexing/vectorization is cancelled by user request.
    This allows graceful shutdown and cleanup of partial data.
    """
    pass


def create_response(data: dict, status_code: int = 200) -> func.HttpResponse:
    """Create a JSON HTTP response."""
    return func.HttpResponse(
        body=json.dumps(data, default=str),
        status_code=status_code,
        mimetype="application/json"
    )


def create_error_response(message: str, status_code: int = 400, details: dict = None) -> func.HttpResponse:
    """Create an error HTTP response."""
    error_body = {"error": message}
    if details:
        error_body["details"] = details
    return func.HttpResponse(
        body=json.dumps(error_body),
        status_code=status_code,
        mimetype="application/json"
    )


def parse_request_body(req: func.HttpRequest) -> dict:
    """
    Parse JSON body from request.
    Returns empty dict if no body or invalid JSON.
    """
    try:
        return req.get_json()
    except ValueError:
        return {}


def get_required_param(body: dict, param_name: str) -> tuple:
    """
    Get a required parameter from request body.
    Returns (value, None) if present, (None, error_response) if missing.
    """
    value = body.get(param_name)
    if not value:
        return None, create_error_response(f"Missing required parameter: {param_name}")
    return value, None


def setup_logging():
    """Configure logging for Azure Functions."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# Supabase Storage exports
from shared.supabase_storage import (
    upload_project_data,
    download_threads_data,
    download_attachments_data,
    delete_project_data
)

# Progress Store exports (including cancellation)
from shared.progress_store import (
    update_progress,
    get_progress,
    clear_progress,
    request_cancel,
    is_cancelled,
    clear_cancel
)

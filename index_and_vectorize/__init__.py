"""
Azure Function: index_and_vectorize

Combined HTTP Trigger that:
1. Indexes emails using ProjectIndexer (0-80% progress)
2. Stores JSON in Supabase Storage
3. Automatically vectorizes using ProjectVectorizer (80-100% progress)

Progress is stored in progress_store for real-time polling via get_project_status endpoint.
"""
import logging
import sys
import os
import json
import tempfile
import time
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import parse_request_body, get_required_param, upload_project_data, create_response, create_error_response
from shared.progress_store import update_progress, clear_progress
from project_indexer import ProjectIndexer
from project_vectorizer import ProjectVectorizer

logger = logging.getLogger(__name__)


# User-friendly message mappings for indexer progress
INDEXER_MESSAGES = {
    0: "Preparing to scan your emails...",
    5: "Searching for emails related to this project...",
    15: "Found your conversations, now fetching details...",
    20: "Reading through your email threads...",
    25: "Still fetching email contents, please wait...",
    30: "Almost done fetching emails...",
    35: "Processing email attachments...",
    40: "Analyzing email threads...",
    45: "Looking for PDF attachments...",
    50: "Found some documents to analyze...",
    55: "Reading through your PDF documents...",
    60: "Extracting information from PDFs...",
    65: "Still processing documents...",
    70: "Almost done with document analysis...",
    75: "Organizing all the information...",
    80: "Building your project knowledge base...",
    85: "Compiling all conversation data...",
    90: "Preparing final index...",
    95: "Almost ready...",
    98: "Finalizing indexing...",
    100: "Indexing complete!"
}


def get_user_friendly_message(percent: int, default_msg: str) -> str:
    """Get a user-friendly message for the given percentage."""
    # Find the closest percentage that has a message
    closest = min(INDEXER_MESSAGES.keys(), key=lambda x: abs(x - percent))
    if abs(closest - percent) <= 5:  # Within 5% range
        return INDEXER_MESSAGES[closest]
    return default_msg


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Index and vectorize a project.
    
    POST Body:
    {
        "project_name": "88 SuperMarket",
        "user_email": "harv@example.com",
        "gmail_credentials": { "access_token": "...", "refresh_token": "...", ... },
        "max_threads": 50  // Optional: limit number of threads to index
    }
    
    Returns: JSON with complete result and progress log
    """
    logger.info('index_and_vectorize: Processing request')
    
    # Parse request body
    body = parse_request_body(req)
    if not body:
        return create_error_response("Request body must be valid JSON")
    
    # Validate required parameters
    project_name, err = get_required_param(body, "project_name")
    if err:
        return err
    
    user_email, err = get_required_param(body, "user_email")
    if err:
        return err
    
    gmail_credentials, err = get_required_param(body, "gmail_credentials")
    if err:
        return err
    
    max_threads = body.get("max_threads")
    
    # Generate project_id early for progress tracking
    project_id = project_name.lower().replace(" ", "_").replace("-", "_")
    
    # Progress log to track all steps
    progress_log = []
    last_logged_percent = -1
    
    # Persistent stats that get updated during indexing
    current_stats = {
        "thread_count": 0,
        "message_count": 0,
        "pdf_count": 0
    }
    
    def log_progress(step: str, percent: int, stats_update: dict = None):
        nonlocal last_logged_percent, current_stats
        
        # Update stats if provided
        if stats_update:
            current_stats.update(stats_update)
        
        # Avoid duplicate entries for same percentage
        if percent == last_logged_percent and percent != 100:
            return
        last_logged_percent = percent
        
        # Always include current stats in details
        details = {
            "thread_count": current_stats["thread_count"],
            "message_count": current_stats["message_count"],
            "pdf_count": current_stats["pdf_count"]
        }
        
        event = {"step": step, "percent": percent, "timestamp": time.time(), "details": details}
        progress_log.append(event)
        logger.info(f"Progress: {step} ({percent}%) - threads:{details['thread_count']}, msgs:{details['message_count']}, pdfs:{details['pdf_count']}")
        
        # Update shared progress store for polling endpoint
        update_progress(project_id, step, percent, details)
    
    def indexer_progress_callback(msg: str, pct: int):
        """Callback from ProjectIndexer - maps to 0-80% range with friendly messages."""
        # Map indexer's 0-100% to our 0-80% range
        mapped_pct = int(pct * 0.8)
        friendly_msg = get_user_friendly_message(mapped_pct, msg)
        log_progress(friendly_msg, mapped_pct)
    
    def vectorizer_progress_callback(msg: str, pct: int):
        """Callback from ProjectVectorizer - maps to 80-100% range."""
        # Map vectorizer's 0-100% to our 80-100% range
        mapped_pct = 80 + int(pct * 0.2)
        
        # Friendly messages for vectorization
        if pct < 30:
            friendly_msg = "Teaching AI to understand your project..."
        elif pct < 60:
            friendly_msg = "Creating smart search capabilities..."
        elif pct < 90:
            friendly_msg = "Connecting all the pieces together..."
        else:
            friendly_msg = "Final touches on AI training..."
        
        log_progress(friendly_msg, mapped_pct)
    
    try:
        # ============================================================
        # PHASE 1: INDEXING (0-80%)
        # ============================================================
        log_progress("Waking up your AI assistant...", 1)
        
        indexer = ProjectIndexer(
            project_name=project_name,
            user_email=user_email,
            provider="gmail",
            credentials=gmail_credentials,
            max_threads=max_threads
        )
        
        log_progress("Connecting to your email account...", 3)
        
        # Run indexing WITH progress callback
        project_index, attachment_index = indexer.index_project(
            progress_callback=indexer_progress_callback
        )
        project_id = project_index.project_id
        
        # Update stats with final counts from indexing
        log_progress(f"Analyzed {project_index.thread_count} conversations and {project_index.pdf_count} documents", 78, {
            "thread_count": project_index.thread_count,
            "message_count": project_index.message_count,
            "pdf_count": project_index.pdf_count
        })
        
        # Convert to dicts
        threads_data = asdict(project_index)
        attachments_data = asdict(attachment_index)
        
        log_progress("Saving your project data securely...", 79)
        
        # ============================================================
        # PHASE 2: STORE IN SUPABASE
        # ============================================================
        threads_path, attachments_path = upload_project_data(
            user_email=user_email,
            project_id=project_id,
            threads_data=threads_data,
            attachments_data=attachments_data
        )
        
        log_progress("Project data saved successfully!", 80, {
            "threads_path": threads_path,
            "attachments_path": attachments_path
        })
        
        # ============================================================
        # PHASE 3: VECTORIZATION (80-100%)
        # ============================================================
        log_progress("Preparing AI for intelligent search...", 81)
        
        vectorizer = ProjectVectorizer()
        
        log_progress("Teaching AI to understand your project...", 83)
        
        # Create temp file for vectorizer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(threads_data, f, default=str)
            temp_path = f.name
        
        try:
            result = vectorizer.vectorize_project(
                temp_path, 
                progress_callback=vectorizer_progress_callback
            )
            
            log_progress("AI training complete!", 98)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # ============================================================
        # COMPLETE
        # ============================================================
        log_progress("All done! Your project is ready for intelligent search.", 100)
        
        final_result = {
            "status": "completed",
            "project_id": project_id,
            "project_name": project_name,
            "stats": {
                "thread_count": project_index.thread_count,
                "message_count": project_index.message_count,
                "pdf_count": project_index.pdf_count,
                "indexed_at": project_index.indexed_at
            },
            "vectorization": {
                "namespace": result.namespace,
                "vectors_created": result.total_vectors,
                "message_chunks": result.message_chunks,
                "attachment_chunks": result.attachment_chunks,
                "duration_seconds": result.duration_seconds
            },
            "storage_paths": {
                "threads_data": threads_path,
                "attachments_data": attachments_path
            },
            "progress_log": progress_log
        }
        
        return create_response(final_result)
        
    except Exception as e:
        logger.error(f"index_and_vectorize error: {str(e)}", exc_info=True)
        log_progress(f"Something went wrong: {str(e)}", -1)
        return create_error_response(str(e), status_code=500, details={"progress_log": progress_log})

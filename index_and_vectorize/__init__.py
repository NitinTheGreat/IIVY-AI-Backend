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

from pinecone import Pinecone

from shared import parse_request_body, get_required_param, upload_project_data, create_response, create_error_response, CancellationError, delete_project_data
from shared.progress_store import update_progress, clear_progress, is_cancelled, clear_cancel
from project_indexer import ProjectIndexer, _generate_project_id
from project_vectorizer import ProjectVectorizer

# Configuration
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "donna-email")

logger = logging.getLogger(__name__)


# User-friendly message mappings for indexing phase (0-80%)
INDEXING_MESSAGES = {
    0: ("Starting", "Waking up your AI assistant..."),
    3: ("Connecting", "Connecting to your email account..."),
    5: ("Searching", "Searching for project-related emails..."),
    15: ("Discovering", "Found your conversations, fetching details..."),
    25: ("Reading Emails", "Reading through your email threads..."),
    35: ("Fetching", "Downloading email contents..."),
    45: ("Scanning Attachments", "Looking for PDF attachments..."),
    55: ("Processing PDFs", "Extracting information from documents..."),
    65: ("Analyzing", "AI is analyzing your documents..."),
    72: ("Building Index", "Organizing all the information..."),
    78: ("Saving", "Saving your project data..."),
    80: ("Indexing Complete", "All emails and documents indexed!"),
}

# User-friendly message mappings for vectorization phase (80-100%)
VECTORIZATION_MESSAGES = {
    80: ("Starting Vectorization", "Preparing AI for intelligent search..."),
    83: ("Creating Embeddings", "Teaching AI to understand your project..."),
    88: ("Training", "Creating smart search capabilities..."),
    93: ("Uploading", "Storing AI knowledge in cloud..."),
    98: ("Finalizing", "Final touches on AI training..."),
    100: ("Complete", "Your project is ready for intelligent search!"),
}


def get_phase_and_message(percent: int) -> tuple:
    """Get the phase name and user-friendly message for the given percentage."""
    if percent <= 80:
        messages = INDEXING_MESSAGES
    else:
        messages = VECTORIZATION_MESSAGES
    
    # Find the closest percentage that has a message
    closest = min(messages.keys(), key=lambda x: abs(x - percent))
    if abs(closest - percent) <= 5:  # Within 5% range
        return messages[closest]
    
    # Fallback
    if percent <= 80:
        return ("Indexing", f"Processing... {percent}%")
    else:
        return ("Vectorizing", f"AI Training... {percent}%")


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
    
    # Generate project_id early for progress tracking (with user isolation)
    project_id = _generate_project_id(project_name, user_email)
    
    # Progress log to track all steps
    progress_log = []
    last_logged_percent = -1
    
    # Persistent stats that get updated during indexing
    current_stats = {
        "thread_count": 0,
        "message_count": 0,
        "pdf_count": 0
    }
    
    def log_progress(percent: int, stats_update: dict = None, custom_msg: str = None):
        nonlocal last_logged_percent, current_stats
        
        # Update stats if provided
        if stats_update:
            current_stats.update(stats_update)
        
        # Avoid duplicate entries for same percentage
        if percent == last_logged_percent and percent != 100:
            return
        last_logged_percent = percent
        
        # Get phase and message
        phase, message = get_phase_and_message(percent)
        if custom_msg:
            message = custom_msg
        
        # Always include current stats in details
        details = {
            "thread_count": current_stats["thread_count"],
            "message_count": current_stats["message_count"],
            "pdf_count": current_stats["pdf_count"]
        }
        
        event = {"phase": phase, "step": message, "percent": percent, "timestamp": time.time(), "details": details}
        progress_log.append(event)
        logger.info(f"Progress: [{phase}] {message} ({percent}%)")
        
        # Update shared progress store for polling endpoint
        update_progress(project_id, phase, message, percent, details)
    
    def indexer_progress_callback(msg: str, pct: int):
        """Callback from ProjectIndexer - maps to 0-80% range with friendly messages."""
        # Map indexer's 0-100% to our 0-80% range
        mapped_pct = int(pct * 0.8)
        log_progress(mapped_pct)
    
    def vectorizer_progress_callback(msg: str, pct: int):
        """Callback from ProjectVectorizer - maps to 80-100% range."""
        # Map vectorizer's 0-100% to our 80-100% range
        mapped_pct = 80 + int(pct * 0.2)
        log_progress(mapped_pct)
    
    try:
        # ============================================================
        # PHASE 1: INDEXING (0-80%)
        # ============================================================
        log_progress(1)
        
        indexer = ProjectIndexer(
            project_name=project_name,
            user_email=user_email,
            provider="gmail",
            credentials=gmail_credentials,
            max_threads=max_threads
        )
        
        log_progress(3)
        
        # Run indexing WITH progress callback
        project_index, attachment_index = indexer.index_project(
            progress_callback=indexer_progress_callback
        )
        # project_id already set correctly from _generate_project_id()
        
        # Update stats with final counts from indexing
        log_progress(78, {
            "thread_count": project_index.thread_count,
            "message_count": project_index.message_count,
            "pdf_count": project_index.pdf_count
        })
        
        # Convert to dicts
        threads_data = asdict(project_index)
        attachments_data = asdict(attachment_index)
        
        log_progress(79)
        
        # ============================================================
        # PHASE 2: STORE IN SUPABASE
        # ============================================================
        threads_path, attachments_path = upload_project_data(
            user_email=user_email,
            project_id=project_id,
            threads_data=threads_data,
            attachments_data=attachments_data
        )
        
        log_progress(80)
        
        # ============================================================
        # PHASE 3: VECTORIZATION (80-100%)
        # ============================================================
        log_progress(81)
        
        vectorizer = ProjectVectorizer()
        
        log_progress(83)
        
        # Create temp file for vectorizer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(threads_data, f, default=str)
            temp_path = f.name
        
        try:
            result = vectorizer.vectorize_project(
                temp_path, 
                progress_callback=vectorizer_progress_callback
            )
            
            log_progress(98)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Final completion update with all stats - ensure this persists for frontend polling
        update_progress(
            project_id,
            "Complete",
            "Your project is ready for intelligent search!",
            100,
            {
                "thread_count": project_index.thread_count,
                "message_count": project_index.message_count,
                "pdf_count": project_index.pdf_count
            }
        )
        logger.info(f"Progress FINAL: [Complete] 100% - project_id={project_id}")
        
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
        
    except CancellationError as e:
        # User requested cancellation - perform cleanup
        logger.info(f"index_and_vectorize: Cancellation detected for {project_id}: {str(e)}")
        
        # Cleanup Step 1: Delete from Supabase Storage (if anything was uploaded)
        try:
            delete_project_data(user_email, project_id)
            logger.info(f"Cleanup: Deleted Supabase storage for {project_id}")
        except Exception as cleanup_err:
            logger.warning(f"Cleanup: Could not delete Supabase storage: {cleanup_err}")
        
        # Cleanup Step 2: Delete from Pinecone (if any vectors were created)
        try:
            pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            if pinecone_api_key:
                pc = Pinecone(api_key=pinecone_api_key)
                index = pc.Index(PINECONE_INDEX_NAME)
                index.delete(delete_all=True, namespace=project_id)
                logger.info(f"Cleanup: Deleted Pinecone namespace for {project_id}")
        except Exception as cleanup_err:
            logger.warning(f"Cleanup: Could not delete Pinecone namespace: {cleanup_err}")
        
        # Cleanup Step 3: Clear cancellation flag and progress
        clear_cancel(project_id)
        
        # Update progress to show cancelled status
        update_progress(
            project_id,
            "Cancelled",
            "Project indexing was cancelled by user",
            -1,
            current_stats
        )
        
        return create_response({
            "status": "cancelled",
            "project_id": project_id,
            "message": "Project indexing was cancelled and partial data has been cleaned up",
            "progress_log": progress_log
        })
        
    except Exception as e:
        logger.error(f"index_and_vectorize error: {str(e)}", exc_info=True)
        update_progress(project_id, "Error", f"Something went wrong: {str(e)}", -1, current_stats)
        return create_error_response(str(e), status_code=500, details={"progress_log": progress_log})

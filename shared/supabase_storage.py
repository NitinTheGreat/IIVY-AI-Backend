"""
Supabase Storage client for storing and retrieving project JSON data.

Uses the `json-storage` bucket to store threads_data and attachments_data
after project indexing, enabling automatic retrieval during vectorization.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple

from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Bucket name for storing project JSON data
BUCKET_NAME = "json-storage"


def get_supabase_client() -> Client:
    """Get Supabase client using environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
    
    return create_client(url, key)


def _get_storage_path(user_email: str, project_id: str, filename: str) -> str:
    """Generate consistent storage path for project data files."""
    # Sanitize email for use in path (replace @ and . with safe chars)
    safe_email = user_email.replace("@", "_at_").replace(".", "_")
    return f"{safe_email}/{project_id}/{filename}"


def upload_project_data(
    user_email: str,
    project_id: str,
    threads_data: Dict[str, Any],
    attachments_data: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[str]]:
    """
    Upload project data (threads and attachments) to Supabase Storage.
    
    Args:
        user_email: User's email address
        project_id: Project identifier
        threads_data: Threads data dict from indexing
        attachments_data: Optional attachments data dict
        
    Returns:
        Tuple of (threads_path, attachments_path) for stored files
    """
    client = get_supabase_client()
    storage = client.storage.from_(BUCKET_NAME)
    
    # Upload threads_data
    threads_path = _get_storage_path(user_email, project_id, "threads_data.json")
    threads_json = json.dumps(threads_data, default=str).encode('utf-8')
    
    logger.info(f"Uploading threads_data to {threads_path}")
    
    # Try to remove existing file first (upsert behavior)
    try:
        storage.remove([threads_path])
    except Exception:
        pass  # File may not exist
    
    storage.upload(
        path=threads_path,
        file=threads_json,
        file_options={"content-type": "application/json"}
    )
    
    # Upload attachments_data if provided
    attachments_path = None
    if attachments_data:
        attachments_path = _get_storage_path(user_email, project_id, "attachments_data.json")
        attachments_json = json.dumps(attachments_data, default=str).encode('utf-8')
        
        logger.info(f"Uploading attachments_data to {attachments_path}")
        
        try:
            storage.remove([attachments_path])
        except Exception:
            pass
        
        storage.upload(
            path=attachments_path,
            file=attachments_json,
            file_options={"content-type": "application/json"}
        )
    
    logger.info(f"Successfully uploaded project data for {project_id}")
    return threads_path, attachments_path


def download_threads_data(user_email: str, project_id: str) -> Dict[str, Any]:
    """
    Download threads_data from Supabase Storage.
    
    Args:
        user_email: User's email address
        project_id: Project identifier
        
    Returns:
        Threads data dict
        
    Raises:
        FileNotFoundError: If file doesn't exist in storage
    """
    client = get_supabase_client()
    storage = client.storage.from_(BUCKET_NAME)
    
    threads_path = _get_storage_path(user_email, project_id, "threads_data.json")
    
    logger.info(f"Downloading threads_data from {threads_path}")
    
    try:
        response = storage.download(threads_path)
        return json.loads(response.decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to download threads_data: {e}")
        raise FileNotFoundError(f"threads_data not found for project {project_id}") from e


def download_attachments_data(user_email: str, project_id: str) -> Optional[Dict[str, Any]]:
    """
    Download attachments_data from Supabase Storage.
    
    Args:
        user_email: User's email address
        project_id: Project identifier
        
    Returns:
        Attachments data dict, or None if not found
    """
    client = get_supabase_client()
    storage = client.storage.from_(BUCKET_NAME)
    
    attachments_path = _get_storage_path(user_email, project_id, "attachments_data.json")
    
    logger.info(f"Downloading attachments_data from {attachments_path}")
    
    try:
        response = storage.download(attachments_path)
        return json.loads(response.decode('utf-8'))
    except Exception as e:
        logger.warning(f"attachments_data not found (this may be expected): {e}")
        return None


def delete_project_data(user_email: str, project_id: str) -> bool:
    """
    Delete all stored data for a project.
    
    Args:
        user_email: User's email address
        project_id: Project identifier
        
    Returns:
        True if deletion was successful
    """
    client = get_supabase_client()
    storage = client.storage.from_(BUCKET_NAME)
    
    threads_path = _get_storage_path(user_email, project_id, "threads_data.json")
    attachments_path = _get_storage_path(user_email, project_id, "attachments_data.json")
    
    logger.info(f"Deleting project data for {project_id}")
    
    try:
        storage.remove([threads_path, attachments_path])
        return True
    except Exception as e:
        logger.error(f"Failed to delete project data: {e}")
        return False

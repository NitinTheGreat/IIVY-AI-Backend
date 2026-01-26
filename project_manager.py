"""
Project Manager for Donna Email Research.

This module manages project indexing lifecycle:
- Lists indexed projects (from Pinecone + local registry)
- Tracks metadata (when indexed, last email, stats)
- Checks if projects need updates (new emails)
- Orchestrates indexing + vectorization workflow

Usage:
    from project_manager import ProjectManager
    
    manager = ProjectManager()
    projects = manager.list_projects(user_email)
    
    # Index a new project
    manager.index_new_project("88 SuperMarket", user_email, progress_callback)
    
    # Check if project needs update
    if manager.needs_update("88_supermarket"):
        manager.update_project("88_supermarket", user_email)
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PINECONE_INDEX_NAME = "donna-email"
PROJECT_INDEXES_DIR = "project_indexes"
REGISTRY_FILE = "index_registry.json"
IS_LOCAL_DEV = os.environ.get("IS_LOCAL_DEV", "true").lower() == "true"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ProjectInfo:
    """Information about an indexed project."""
    project_id: str
    project_name: str
    user_email: str
    indexed_at: str
    last_email_timestamp: str
    vector_count: int
    thread_count: int
    message_count: int
    attachment_count: int
    is_indexed: bool = True


@dataclass 
class IndexingProgress:
    """Progress update during indexing."""
    phase: str  # "indexing", "vectorizing", "complete"
    message: str
    percent: int
    details: Optional[Dict] = None


# ============================================================
# PROJECT MANAGER CLASS
# ============================================================

class ProjectManager:
    """
    Manages project indexing and search lifecycle.
    """
    
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        indexes_dir: str = PROJECT_INDEXES_DIR
    ):
        """
        Initialize the project manager.
        
        Args:
            pinecone_api_key: Pinecone API key (or from env)
            openai_api_key: OpenAI API key (or from env)
            indexes_dir: Directory for storing index files
        """
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.indexes_dir = indexes_dir
        self.registry_path = os.path.join(indexes_dir, REGISTRY_FILE)
        
        # Ensure directory exists (only on local - Azure filesystem is read-only)
        if IS_LOCAL_DEV:
            os.makedirs(indexes_dir, exist_ok=True)
        
        # Lazy-loaded clients
        self._pinecone_index = None
    
    @property
    def pinecone_index(self):
        """Lazy-load Pinecone index."""
        if self._pinecone_index is None and self.pinecone_api_key:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.pinecone_api_key)
                self._pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            except Exception as e:
                print(f"Warning: Could not connect to Pinecone: {e}")
        return self._pinecone_index
    
    # ============================================================
    # REGISTRY MANAGEMENT
    # ============================================================
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load the project registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
        return {}
    
    def _save_registry(self, registry: Dict[str, Dict]):
        """Save the project registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save registry: {e}")
    
    def _update_registry(self, project_id: str, info: Dict):
        """Update a single project in the registry."""
        registry = self._load_registry()
        registry[project_id] = info
        self._save_registry(registry)
    
    # ============================================================
    # PROJECT LISTING
    # ============================================================
    
    def list_projects(self, user_email: Optional[str] = None) -> List[ProjectInfo]:
        """
        List all indexed projects.
        
        Args:
            user_email: Filter by user email (optional)
        
        Returns:
            List of ProjectInfo objects
        """
        projects = []
        
        # Get vector counts from Pinecone
        pinecone_namespaces = {}
        if self.pinecone_index:
            try:
                stats = self.pinecone_index.describe_index_stats()
                pinecone_namespaces = {
                    ns: data.get("vector_count", 0) 
                    for ns, data in stats.get("namespaces", {}).items()
                }
            except Exception as e:
                print(f"Warning: Could not get Pinecone stats: {e}")
        
        # Load local registry
        registry = self._load_registry()
        
        # Combine: prefer Pinecone as source of truth for existence
        all_project_ids = set(pinecone_namespaces.keys()) | set(registry.keys())
        
        for project_id in all_project_ids:
            # Skip if no vectors in Pinecone AND not in registry
            vector_count = pinecone_namespaces.get(project_id, 0)
            if vector_count == 0 and project_id not in registry:
                continue
            
            # Get metadata from registry (if available)
            meta = registry.get(project_id, {})
            
            # Filter by user if specified (only if registry has user info)
            if user_email and meta.get("user_email") and meta.get("user_email") != user_email:
                continue
            
            # Fallback values if not in registry
            project_name = meta.get("project_name") or project_id.replace("_", " ").title()
            indexed_at = meta.get("indexed_at") or "Unknown"
            
            projects.append(ProjectInfo(
                project_id=project_id,
                project_name=project_name,
                user_email=meta.get("user_email", "Unknown"),
                indexed_at=indexed_at,
                last_email_timestamp=meta.get("last_email_timestamp", ""),
                vector_count=vector_count, # Always prefer Pinecone count
                thread_count=meta.get("thread_count", 0),
                message_count=meta.get("message_count", 0),
                attachment_count=meta.get("attachment_count", 0),
                is_indexed=vector_count > 0
            ))
        
        # Sort by indexed_at (newest first)
        projects.sort(key=lambda p: p.indexed_at or "", reverse=True)
        
        return projects
    
    def get_project(self, project_id: str) -> Optional[ProjectInfo]:
        """Get info for a specific project."""
        projects = self.list_projects()
        for p in projects:
            if p.project_id == project_id:
                return p
        return None
    
    def is_indexed(self, project_id: str) -> bool:
        """Check if a project is indexed and searchable."""
        if not self.pinecone_index:
            return False
        
        try:
            stats = self.pinecone_index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            return project_id in namespaces and namespaces[project_id].get("vector_count", 0) > 0
        except:
            return False
    
    # ============================================================
    # INDEXING WORKFLOW
    # ============================================================
    
    def index_new_project(
        self,
        project_name: str,
        user_email: str,
        provider: str = "gmail",
        credentials: Optional[Dict] = None,
        progress_callback: Optional[Callable[[IndexingProgress], None]] = None
    ) -> ProjectInfo:
        """
        Index a new project (full workflow: index → vectorize).
        
        Args:
            project_name: Name of the project (e.g., "88 SuperMarket")
            user_email: User's email address
            provider: Email provider ("gmail" or "outlook")
            credentials: Optional auth credentials
            progress_callback: Optional callback for progress updates
        
        Returns:
            ProjectInfo for the indexed project
        """
        from project_indexer import ProjectIndexer, _generate_project_id
        from project_vectorizer import ProjectVectorizer
        
        project_id = _generate_project_id(project_name, user_email)
        
        def _progress(phase: str, msg: str, pct: int, details: Dict = None):
            if progress_callback:
                progress_callback(IndexingProgress(
                    phase=phase,
                    message=msg,
                    percent=pct,
                    details=details
                ))
            print(f"[{phase}] {pct}% - {msg}")
        
        _progress("indexing", f"Starting indexing for '{project_name}'...", 0)
        
        # Phase 1: Index emails and PDFs
        indexer = ProjectIndexer(
            project_name=project_name,
            user_email=user_email,
            provider=provider,
            credentials=credentials
        )
        
        def indexer_progress(msg, pct):
            # Scale to 0-60% for indexing phase
            scaled_pct = int(pct * 0.6)
            _progress("indexing", msg, scaled_pct)
        
        project_index, attachment_index = indexer.index_project(progress_callback=indexer_progress)
        
        # Save JSON files
        threads_path = indexer.save_index(project_index, self.indexes_dir)
        attachments_path = indexer.save_attachments_index(attachment_index, self.indexes_dir)
        
        _progress("indexing", "Indexing complete. Starting vectorization...", 60)
        
        # Phase 2: Vectorize
        if not self.pinecone_api_key or not self.openai_api_key:
            _progress("error", "Missing API keys for vectorization", 60)
            raise ValueError("Pinecone and OpenAI API keys required for vectorization")
        
        vectorizer = ProjectVectorizer(
            pinecone_api_key=self.pinecone_api_key,
            openai_api_key=self.openai_api_key
        )
        
        def vectorizer_progress(msg, pct):
            # Scale to 60-95% for vectorization phase
            scaled_pct = 60 + int(pct * 0.35)
            _progress("vectorizing", msg, scaled_pct)
        
        result = vectorizer.vectorize_project(
            threads_path, 
            attachments_path,
            progress_callback=vectorizer_progress
        )
        
        # Phase 3: Update registry
        _progress("complete", "Updating registry...", 98)
        
        registry_entry = {
            "project_id": project_id,
            "project_name": project_name,
            "user_email": user_email,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
            "last_email_timestamp": project_index.last_email_timestamp,
            "vector_count": result.total_vectors,
            "thread_count": project_index.thread_count,
            "message_count": project_index.message_count,
            "attachment_count": attachment_index.total_attachments,
            "threads_json": threads_path,
            "attachments_json": attachments_path
        }
        
        self._update_registry(project_id, registry_entry)
        
        _progress("complete", f"Project '{project_name}' indexed successfully!", 100, {
            "vectors": result.total_vectors,
            "threads": project_index.thread_count,
            "attachments": attachment_index.total_attachments
        })
        
        return ProjectInfo(
            project_id=project_id,
            project_name=project_name,
            user_email=user_email,
            indexed_at=registry_entry["indexed_at"],
            last_email_timestamp=registry_entry["last_email_timestamp"],
            vector_count=result.total_vectors,
            thread_count=project_index.thread_count,
            message_count=project_index.message_count,
            attachment_count=attachment_index.total_attachments,
            is_indexed=True
        )
    
    def delete_project(self, project_id: str):
        """
        Delete a project (vectors + registry entry + Supabase storage).
        
        Args:
            project_id: Project ID to delete
        """
        # Get user_email from registry before deleting (needed for Supabase)
        registry = self._load_registry()
        user_email = registry.get(project_id, {}).get("user_email")
        
        # Delete from Pinecone
        if self.pinecone_index:
            try:
                self.pinecone_index.delete(delete_all=True, namespace=project_id)
                print(f"Deleted Pinecone namespace: {project_id}")
            except Exception as e:
                print(f"Warning: Could not delete Pinecone namespace: {e}")
        
        # Delete from Supabase Storage
        if user_email:
            try:
                from shared.supabase_storage import delete_project_data
                success = delete_project_data(user_email, project_id)
                if success:
                    print(f"Deleted Supabase storage: {project_id}")
                else:
                    print(f"Warning: Could not delete Supabase storage for {project_id}")
            except Exception as e:
                print(f"Warning: Could not delete Supabase storage: {e}")
        else:
            print(f"Warning: No user_email found in registry, skipping Supabase deletion")
        
        # Remove from registry
        if project_id in registry:
            del registry[project_id]
            self._save_registry(registry)
            print(f"Removed from registry: {project_id}")
    
    def needs_update(self, project_id: str, user_email: str, provider: str = "gmail", credentials: Dict = None) -> bool:
        """
        Check if a project has new emails since last index.
        
        Args:
            project_id: Project to check
            user_email: User's email
            provider: Email provider
            credentials: Auth credentials
        
        Returns:
            True if there are new emails to index
        """
        registry = self._load_registry()
        project_meta = registry.get(project_id)
        
        if not project_meta:
            return True  # Not indexed at all
        
        last_email = project_meta.get("last_email_timestamp")
        if not last_email:
            return True  # No timestamp recorded
        
        # Check Gmail for newer emails
        try:
            if provider == "gmail":
                from tools.gmail_tools import get_gmail_tools
                gmail = get_gmail_tools(credentials=credentials)
                
                project_name = project_meta.get("project_name", project_id)
                # Search for emails after last indexed timestamp
                result = gmail.search_messages(
                    query=f'"{project_name}" after:{last_email[:10]}',
                    top=1
                )
                
                emails = result.get("emails") or result.get("messages") or []
                return len(emails) > 0
        except Exception as e:
            print(f"Warning: Could not check for updates: {e}")
        
        return False


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_project_manager() -> ProjectManager:
    """Get a singleton ProjectManager instance."""
    return ProjectManager()


def list_indexed_projects(user_email: Optional[str] = None) -> List[ProjectInfo]:
    """List all indexed projects."""
    return get_project_manager().list_projects(user_email)


def index_project(
    project_name: str,
    user_email: str,
    provider: str = "gmail",
    credentials: Dict = None,
    progress_callback: Callable = None
) -> ProjectInfo:
    """Index a new project."""
    return get_project_manager().index_new_project(
        project_name=project_name,
        user_email=user_email,
        provider=provider,
        credentials=credentials,
        progress_callback=progress_callback
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    
    manager = ProjectManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python project_manager.py list")
        print("  python project_manager.py index 'Project Name' user@email.com")
        print("  python project_manager.py delete project_id")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "list":
        projects = manager.list_projects()
        print(f"\n📂 Indexed Projects ({len(projects)}):\n")
        for p in projects:
            status = "✓" if p.is_indexed else "⚠"
            print(f"  {status} {p.project_name} ({p.project_id})")
            print(f"      Vectors: {p.vector_count} | Threads: {p.thread_count}")
            print(f"      Indexed: {p.indexed_at[:10] if p.indexed_at else 'N/A'}")
            print()
    
    elif cmd == "index" and len(sys.argv) >= 4:
        project_name = sys.argv[2]
        user_email = sys.argv[3]
        print(f"\n🚀 Indexing '{project_name}' for {user_email}...\n")
        result = manager.index_new_project(project_name, user_email)
        print(f"\n✅ Done! {result.vector_count} vectors created.")
    
    elif cmd == "delete" and len(sys.argv) >= 3:
        project_id = sys.argv[2]
        print(f"\n🗑️ Deleting '{project_id}'...")
        manager.delete_project(project_id)
        print("✅ Deleted.")
    
    else:
        print("Unknown command. Use: list, index, delete")


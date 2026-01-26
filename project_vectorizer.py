"""
Project Vectorizer for Donna Email Research.

This module handles Phase 2 of project-based email search:
- Takes the JSON outputs from project_indexer.py
- Creates MESSAGE chunks (from thread messages)
- Creates ATTACHMENT chunks (from PDF AI analysis)
- Embeds all chunks using OpenAI
- Upserts to Pinecone with rich metadata

Usage:
    # Option 1: Pass JSON file paths directly
    python project_vectorizer.py threads.json attachments.json
    
    # Option 2: Pass project directory (finds latest files)
    python project_vectorizer.py --dir ./project_indexes/ --project 88_supermarket
    
    # Option 3: Import and use programmatically
    from project_vectorizer import ProjectVectorizer
    vectorizer = ProjectVectorizer(pinecone_api_key, openai_api_key)
    vectorizer.vectorize_project(threads_json_path, attachments_json_path)
"""

import json
import os
import re
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import cancellation utilities
from shared import CancellationError
from shared.progress_store import is_cancelled


# ============================================================
# CONFIGURATION
# ============================================================

# Pinecone settings
PINECONE_INDEX_NAME = "donna-email"  # Create this index in Pinecone dashboard
PINECONE_DIMENSION = 1536  # text-embedding-3-small dimension

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100  # Max texts per API call

# Chunk settings
MAX_CHUNK_CHARS = 4000  # Smaller chunks for sliding window (conservative limit)
CHUNK_OVERLAP = 400     # Overlap to preserve context at boundaries


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Chunk:
    """A chunk ready for vectorization."""
    chunk_id: str
    chunk_type: str  # "message" or "attachment"
    text: str
    metadata: Dict[str, Any]


@dataclass
class VectorizationResult:
    """Result of vectorization process."""
    project_id: str
    namespace: str
    message_chunks: int
    attachment_chunks: int
    total_vectors: int
    duration_seconds: float


# ============================================================
# PROJECT VECTORIZER CLASS
# ============================================================

class ProjectVectorizer:
    """
    Vectorizes project data from project_indexer.py output.
    Creates embeddings and stores in Pinecone.
    """
    
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        index_name: str = PINECONE_INDEX_NAME
    ):
        """
        Initialize the vectorizer.
        
        Args:
            pinecone_api_key: Pinecone API key (or from PINECONE_API_KEY env var)
            openai_api_key: OpenAI API key (or from OPENAI_API_KEY env var)
            index_name: Name of Pinecone index
        """
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.index_name = index_name
        
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key required (pass or set PINECONE_API_KEY)")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required (pass or set OPENAI_API_KEY)")
        
        # Initialize clients lazily
        self._pinecone_index = None
        self._openai_client = None
        
        # Project ID for cancellation checks (set during vectorize_project)
        self._current_project_id = None
    
    @property
    def pinecone_index(self):
        """Lazy-load Pinecone index."""
        if self._pinecone_index is None:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_api_key)
            self._pinecone_index = pc.Index(self.index_name)
        return self._pinecone_index
    
    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.openai_api_key)
        return self._openai_client
    
    def vectorize_project(
        self,
        threads_json_path: str,
        progress_callback=None
    ) -> VectorizationResult:
        """
        Main entry point: Vectorize a project's data.
        
        Args:
            threads_json_path: Path to threads JSON from project_indexer.py
            progress_callback: Optional function(message, percent) for progress updates
        
        Returns:
            VectorizationResult with stats
        """
        start_time = time.time()
        
        def _progress(msg: str, pct: int = 0):
            elapsed = time.time() - start_time
            print(f"📊 [{pct:3d}%] [{elapsed:6.1f}s] {msg}")
            if progress_callback:
                progress_callback(msg, pct)
        
        print("\n" + "="*60)
        print("🚀 PROJECT VECTORIZER - Starting")
        print("="*60)
        
        # Step 1: Load JSON file
        _progress("Loading JSON file...", 5)
        threads_data = self._load_json(threads_json_path)
        
        project_id = threads_data.get("project_id", "unknown")
        project_name = threads_data.get("project_name", "Unknown Project")
        
        # Store project_id for cancellation checks
        self._current_project_id = project_id
        
        print(f"   Project:     {project_name}")
        print(f"   Project ID:  {project_id}")
        print(f"   Threads:     {threads_data.get('thread_count', 0)}")
        print("="*60)
        
        # Step 2: Process Threads (Messages + Attachments)
        _progress("Processing threads and attachments...", 15)
        all_chunks = self._process_threads(threads_data)
        
        message_chunks_count = len([c for c in all_chunks if c.chunk_type == "message"])
        attachment_chunks_count = len([c for c in all_chunks if c.chunk_type == "attachment"])
        
        print(f"   ✓ Generated {len(all_chunks)} total chunks")
        print(f"     - Messages: {message_chunks_count}")
        print(f"     - Attachments: {attachment_chunks_count}")
        
        if not all_chunks:
            print("⚠️ No chunks to vectorize!")
            return VectorizationResult(
                project_id=project_id,
                namespace=project_id,
                message_chunks=0,
                attachment_chunks=0,
                total_vectors=0,
                duration_seconds=time.time() - start_time
            )
        
        # Step 3: Generate embeddings
        _progress(f"Generating embeddings for {len(all_chunks)} chunks...", 45)
        embedded_chunks = self._embed_chunks(all_chunks, _progress)
        print(f"   ✓ Generated {len(embedded_chunks)} embeddings")
        
        # Step 4: Upsert to Pinecone
        _progress("Upserting to Pinecone...", 80)
        self._upsert_to_pinecone(embedded_chunks, namespace=project_id, progress_cb=_progress)
        
        elapsed = time.time() - start_time
        _progress(f"Vectorization complete in {elapsed:.1f}s", 100)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"📈 VECTORIZATION STATS FOR: {project_name}")
        print(f"{'='*50}")
        print(f"Total chunks:      {len(all_chunks)}")
        print(f"  - Messages:      {message_chunks_count}")
        print(f"  - Attachments:   {attachment_chunks_count}")
        print(f"Pinecone namespace: {project_id}")
        print(f"Total time:        {elapsed:.1f}s")
        print(f"{'='*50}\n")
        
        return VectorizationResult(
            project_id=project_id,
            namespace=project_id,
            message_chunks=message_chunks_count,
            attachment_chunks=attachment_chunks_count,
            total_vectors=len(all_chunks),
            duration_seconds=elapsed
        )
    
    def _load_json(self, path: str) -> Dict:
        """Load a JSON file."""
        print(f"   📂 Loading: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_sliding_window_chunks(self, text: str, base_id: str, metadata: Dict) -> List[Chunk]:
        """
        Split text into overlapping chunks using a sliding window.
        Ensures 100% data retention for long documents.
        """
        if not text:
            return []
            
        chunks = []
        text_len = len(text)
        
        # If text is small enough, return as single chunk
        if text_len <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(
                chunk_id=f"{base_id}_0",
                chunk_type=metadata.get("chunk_type", "unknown"),
                text=text,
                metadata={
                    **metadata,
                    "chunk_index": 0,
                    "total_chunks": 1
                }
            ))
            return chunks
            
        # Sliding window logic
        start = 0
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + MAX_CHUNK_CHARS, text_len)
            
            # If we're not at the end, try to break at a newline for cleaner cuts
            if end < text_len:
                # Look for last newline in the last 10% of the chunk
                search_zone = text[end-int(MAX_CHUNK_CHARS*0.1):end]
                last_newline = search_zone.rfind('\n')
                if last_newline != -1:
                    end = (end - int(MAX_CHUNK_CHARS*0.1)) + last_newline
            
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                chunk_id=f"{base_id}_{chunk_idx}",
                chunk_type=metadata.get("chunk_type", "unknown"),
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": chunk_idx,
                    # We can't know total_chunks easily upfront with dynamic splitting, 
                    # but we can update it later if needed. For now, index is enough.
                }
            ))
            
            # Move window forward, respecting overlap
            start += (MAX_CHUNK_CHARS - CHUNK_OVERLAP)
            chunk_idx += 1
            
        return chunks
    
    def _process_threads(self, threads_data: Dict) -> List[Chunk]:
        """
        Process threads to create both MESSAGE and ATTACHMENT chunks.
        Single-pass approach for maximum context.
        """
        chunks = []
        project_id = threads_data.get("project_id", "unknown")
        
        for thread in threads_data.get("thread_docs", []):
            thread_id = thread.get("thread_id", "")
            thread_subject = thread.get("subject", "")
            
            for msg in thread.get("messages", []):
                message_id = msg.get("message_id", "")
                
                # --- PROCESS EMAIL MESSAGE ---
                
                # Build the chunk text
                from_email = msg.get("from_email", "")
                from_name = msg.get("from_name", "")
                to_list = msg.get("to", [])
                cc_list = msg.get("cc", [])
                timestamp = msg.get("timestamp", "")
                subject = msg.get("subject", "")
                body = msg.get("body", "")
                attachments = msg.get("attachments", [])
                
                # Format TO and CC
                to_str = ", ".join(to_list[:5]) if to_list else "(recipients in thread)"
                if len(to_list) > 5:
                    to_str += f" (+{len(to_list) - 5} more)"
                
                cc_str = ""
                if cc_list:
                    cc_str = f"\nCC: {', '.join(cc_list[:3])}"
                    if len(cc_list) > 3:
                        cc_str += f" (+{len(cc_list) - 3} more)"
                
                # Format attachment filenames for the email text
                att_str = ""
                if attachments:
                    filenames = [a.get("filename", "unknown") for a in attachments]
                    att_str = f"\n\nATTACHMENTS:\n- " + "\n- ".join(filenames)
                
                # Build full text for the email
                sender_display = f"{from_name} <{from_email}>" if from_name else from_email
                email_text = f"""FROM: {sender_display}
TO: {to_str}{cc_str}
DATE: {timestamp}
SUBJECT: {subject}

{body}{att_str}"""
                
                # Vectorize the email (using sliding window, though emails are usually short enough for 1 chunk)
                base_msg_id = f"{project_id}_msg_{message_id[:16]}"
                msg_metadata = {
                    "project_id": project_id,
                    "chunk_type": "message",
                    "thread_id": thread_id,
                    "thread_subject": thread_subject,
                    "message_id": message_id,
                    "sequence": msg.get("sequence", 0),
                    "timestamp": timestamp,
                    "sender_email": from_email,
                    "sender_name": from_name,
                    "has_attachments": len(attachments) > 0,
                    "attachment_count": len(attachments)
                }
                
                msg_chunks = self._create_sliding_window_chunks(email_text, base_msg_id, msg_metadata)
                chunks.extend(msg_chunks)
                
                # --- PROCESS ATTACHMENTS (NESTED) ---
                
                for att in attachments:
                    filename = att.get("filename", "")
                    ai_analysis = att.get("content", "") # Note: 'content' field in threads.json
                    attachment_id = att.get("attachment_id", "")
                    
                    if not ai_analysis:
                        continue
                        
                    # Create rich context header inherited from the email
                    att_header = f"""DOCUMENT: {filename}
SENT BY: {sender_display}
DATE: {timestamp}
EMAIL SUBJECT: {subject}
PARENT EMAIL ID: {message_id}

--- AI ANALYSIS ---

"""
                    # Combine header and content
                    full_att_text = att_header + ai_analysis
                    
                    # Create hash for ID
                    att_hash = hashlib.md5(attachment_id.encode()).hexdigest()[:12]
                    base_att_id = f"{project_id}_att_{att_hash}"
                    
                    att_metadata = {
                        "project_id": project_id,
                        "chunk_type": "attachment",
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "attachment_id": attachment_id,
                        "filename": filename,
                        "content_type": att.get("content_type", ""),
                        "extraction_method": att.get("extraction_method", ""),
                        "timestamp": timestamp,
                        "sender_email": from_email,
                        "sender_name": from_name,
                        "email_subject": subject
                    }
                    
                    # Vectorize attachment with sliding window (CRITICAL for long PDFs)
                    att_chunks = self._create_sliding_window_chunks(full_att_text, base_att_id, att_metadata)
                    chunks.extend(att_chunks)
        
        return chunks
    
    def _embed_chunks(self, chunks: List[Chunk], progress_cb) -> List[Dict]:
        """Generate embeddings for all chunks."""
        embedded = []
        texts = [c.text for c in chunks]
        
        # Process in batches
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            # Check for cancellation before each batch
            if self._current_project_id and is_cancelled(self._current_project_id):
                raise CancellationError(f"Vectorization cancelled during embedding for project {self._current_project_id}")
            
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            
            # Call OpenAI embedding API
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts
            )
            
            # Match embeddings with chunks
            for j, embedding_data in enumerate(response.data):
                chunk = batch_chunks[j]
                embedded.append({
                    "id": chunk.chunk_id,
                    "values": embedding_data.embedding,
                    "metadata": {
                        **chunk.metadata,
                        "text": chunk.text  # Store full text in metadata
                    }
                })
            
            # Progress update
            done = min(i + EMBEDDING_BATCH_SIZE, len(texts))
            pct = 45 + int((done / len(texts)) * 30)
            progress_cb(f"Embedded {done}/{len(texts)} chunks...", pct)
        
        return embedded
    
    def _upsert_to_pinecone(self, vectors: List[Dict], namespace: str, progress_cb):
        """Upsert vectors to Pinecone."""
        batch_size = 100  # Pinecone recommends 100-200 vectors per upsert
        
        for i in range(0, len(vectors), batch_size):
            # Check for cancellation before each batch
            if self._current_project_id and is_cancelled(self._current_project_id):
                raise CancellationError(f"Vectorization cancelled during Pinecone upsert for project {self._current_project_id}")
            
            batch = vectors[i:i + batch_size]
            self.pinecone_index.upsert(vectors=batch, namespace=namespace)
            
            done = min(i + batch_size, len(vectors))
            pct = 80 + int((done / len(vectors)) * 18)
            progress_cb(f"Upserted {done}/{len(vectors)} vectors...", pct)
        
        print(f"   ✓ Upserted {len(vectors)} vectors to namespace '{namespace}'")
    
    def delete_project(self, project_id: str):
        """Delete all vectors for a project (namespace)."""
        print(f"🗑️ Deleting all vectors in namespace '{project_id}'...")
        self.pinecone_index.delete(delete_all=True, namespace=project_id)
        print(f"   ✓ Deleted namespace '{project_id}'")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def vectorize_project(
    threads_json_path: str,
    pinecone_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> VectorizationResult:
    """
    Convenience function to vectorize a project.
    
    Args:
        threads_json_path: Path to threads JSON
        pinecone_api_key: Optional Pinecone API key
        openai_api_key: Optional OpenAI API key
    
    Returns:
        VectorizationResult
    """
    vectorizer = ProjectVectorizer(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key
    )
    return vectorizer.vectorize_project(threads_json_path)


def find_latest_project_files(project_dir: str, project_id: str) -> str:
    """
    Find the latest threads JSON file for a project.
    
    Args:
        project_dir: Directory containing project index files
        project_id: Project ID (e.g., "88_supermarket")
    
    Returns:
        Path to threads JSON file
    """
    import glob
    
    # Find threads files
    threads_pattern = os.path.join(project_dir, f"{project_id}_threads_*.json")
    threads_files = sorted(glob.glob(threads_pattern), reverse=True)
    
    if not threads_files:
        raise FileNotFoundError(f"No threads file found matching: {threads_pattern}")
    
    return threads_files[0]


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vectorize project data for Donna Email Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct file paths:
  python project_vectorizer.py threads.json
  
  # Find latest files in directory:
  python project_vectorizer.py --dir ./project_indexes/ --project 88_supermarket
  
  # Delete a project's vectors:
  python project_vectorizer.py --delete --project 88_supermarket
        """
    )
    
    parser.add_argument("threads_json", nargs="?", help="Path to threads JSON file")
    parser.add_argument("--dir", help="Directory containing project index files")
    parser.add_argument("--project", help="Project ID (used with --dir)")
    parser.add_argument("--delete", action="store_true", help="Delete project vectors")
    
    args = parser.parse_args()
    
    # Handle delete
    if args.delete:
        if not args.project:
            print("Error: --project required with --delete")
            sys.exit(1)
        vectorizer = ProjectVectorizer()
        vectorizer.delete_project(args.project)
        sys.exit(0)
    
    # Determine file paths
    if args.threads_json:
        threads_path = args.threads_json
    elif args.dir and args.project:
        try:
            threads_path = find_latest_project_files(args.dir, args.project)
            print(f"📂 Found threads file: {threads_path}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nError: Provide either threads_json or (--dir + --project)")
        sys.exit(1)
    
    # Run vectorization
    result = vectorize_project(threads_path)
    
    print(f"\n✅ Done! Vectorized {result.total_vectors} chunks to namespace '{result.namespace}'")


"""
Project Search for Donna Email Research.

This module handles Phase 3 of project-based email search:
- Takes a user question about a project
- Searches the vectorized project data in Pinecone
- Uses LLM to synthesize an answer from retrieved chunks
- Returns structured response with sources

Usage:
    # CLI
    python project_search.py "88_supermarket" "What's the pile length?"
    
    # Programmatic
    from project_search import ProjectSearch
    search = ProjectSearch()
    result = search.ask("88_supermarket", "What's the engineering cost?")
    print(result.answer)
"""

import json
import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def normalize_project_id(name: str) -> str:
    """Normalize project name to match Pinecone namespace format."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


# ============================================================
# CONFIGURATION
# ============================================================

# Pinecone settings
PINECONE_INDEX_NAME = "donna-email"

# OpenAI settings (for embeddings only)
EMBEDDING_MODEL = "text-embedding-3-small"

# Gemini Config (for LLM answers)
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_TEMPERATURE = 0.2

# Search settings
TOP_K = 90  # Number of chunks to retrieve from Pinecone


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SourceChunk:
    """A source chunk used to generate the answer."""
    chunk_id: str
    chunk_type: str  # "message" or "attachment"
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Result of a project search."""
    project_id: str
    question: str
    answer: Any # Changed from str to Any (Stream)
    sources: List[SourceChunk]
    chunks_retrieved: int
    search_time_ms: int
    llm_time_ms: int
    total_time_ms: int
    context_string: str = "" # Added context string for debugging


# ============================================================
# PROJECT SEARCH CLASS
# ============================================================

class ProjectSearch:
    """
    Searches vectorized project data and generates answers.
    """
    
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        index_name: str = PINECONE_INDEX_NAME
    ):
        """
        Initialize the search engine.
        
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
        self._gemini_model = None
    
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
    
    @property
    def gemini_model(self):
        """Lazy-load Gemini client (New SDK)."""
        if self._gemini_model is None:
            from google import genai
            
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
                
            self._gemini_model = genai.Client(api_key=api_key)
        return self._gemini_model
    
    def ask(
        self,
        project_id: str,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        include_sources: bool = True
    ) -> SearchResult:
        """
        Ask a question about a project.
        
        Args:
            project_id: Project namespace in Pinecone (e.g., "88_supermarket")
            question: The user's question
            top_k: Number of chunks to retrieve (auto-determined if None)
            filter_metadata: Additional Pinecone metadata filters
            include_sources: Whether to include source chunks in result
        
        Returns:
            SearchResult with answer and sources
        """
        total_start = time.time()
        
        # Normalize project_id to match Pinecone namespace format
        # e.g., "Load Minds" -> "load_minds", "Security" -> "security"
        project_id = normalize_project_id(project_id)
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = TOP_K
        
        print(f"\n🔍 Searching project '{project_id}' (top_k={top_k})")
        print(f"   Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        
        # Step 1: Embed the question
        search_start = time.time()
        question_embedding = self._embed_question(question)
        
        # Step 2: Search Pinecone
        raw_results = self._search_pinecone(
            embedding=question_embedding,
            namespace=project_id,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        search_time_ms = int((time.time() - search_start) * 1000)
        
        if not raw_results:
            print("   ⚠️ No results found")
            return SearchResult(
                project_id=project_id,
                question=question,
                answer="I couldn't find any relevant information for this question in the project data.",
                sources=[],
                chunks_retrieved=0,
                search_time_ms=search_time_ms,
                llm_time_ms=0,
                total_time_ms=int((time.time() - total_start) * 1000)
            )
        
        # Step 3: Sort by timestamp if "latest" query
        if self._is_latest_query(question):
            raw_results = self._sort_by_timestamp(raw_results)
        
        # Step 4: Build context for LLM
        context = self._build_context(raw_results)
        
        # Step 5: Generate answer with LLM
        llm_start = time.time()
        answer_stream = self._generate_answer(question, context, project_id)
        # Note: We can't calculate LLM time upfront anymore because it's a stream
        
        # Build source chunks
        sources = []
        if include_sources:
            for r in raw_results[:5]:  # Top 5 sources
                sources.append(SourceChunk(
                    chunk_id=r["id"],
                    chunk_type=r["metadata"].get("chunk_type", "unknown"),
                    text=r["metadata"].get("text", ""),
                    score=r["score"],
                    metadata=r["metadata"]
                ))
        
        total_time_ms = int((time.time() - total_start) * 1000)
        
        print(f"   ✅ Found {len(raw_results)} chunks")
        print(f"   ⏱️ Search: {search_time_ms}ms")
        
        return SearchResult(
            project_id=project_id,
            question=question,
            answer=answer_stream,  # Now returns the stream object
            sources=sources,
            chunks_retrieved=len(raw_results),
            search_time_ms=search_time_ms,
            llm_time_ms=0, # Placeholder
            total_time_ms=total_time_ms,
            context_string=context # Pass the raw context for debugging
        )
    
    def _is_latest_query(self, question: str) -> bool:
        """Check if this is a 'latest' type query that needs timestamp sorting."""
        q_lower = question.lower()
        return any(word in q_lower for word in ["latest", "recent", "newest", "last", "most recent"])
    
    def _embed_question(self, question: str) -> List[float]:
        """Generate embedding for the question."""
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        )
        return response.data[0].embedding
    
    def _search_pinecone(
        self,
        embedding: List[float],
        namespace: str,
        top_k: int,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Search Pinecone and return matches."""
        
        # --- DEBUG LOGGING ---
        logging.info(f"\n--- PINECONE QUERY DEBUG ---")
        logging.info(f"Index: {self.index_name}")
        logging.info(f"Namespace: {namespace}")
        logging.info(f"Top K: {top_k}")
        logging.info(f"Filter: {filter_metadata}")
        logging.info(f"Embedding length: {len(embedding)}")
        logging.info(f"Embedding sample (first 5): {embedding[:5]}")
        logging.info(f"----------------------------\n")
        # ---------------------
        
        query_params = {
            "vector": embedding,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True
        }
        
        if filter_metadata:
            query_params["filter"] = filter_metadata
        
        results = self.pinecone_index.query(**query_params)
        
        # Convert to list of dicts
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            })
        
        return matches
    
    def _sort_by_timestamp(self, results: List[Dict]) -> List[Dict]:
        """Sort results by timestamp (newest first)."""
        def get_timestamp(r):
            ts = r["metadata"].get("timestamp", "")
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                return datetime.min
        
        return sorted(results, key=get_timestamp, reverse=True)
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from search results for LLM."""
        context_parts = []
        
        for i, r in enumerate(results, 1):
            metadata = r["metadata"]
            chunk_type = metadata.get("chunk_type", "unknown")
            text = metadata.get("text", "")
            
            # Extract common metadata
            sender = metadata.get("sender_name") or metadata.get("sender_email", "Unknown")
            timestamp = metadata.get("timestamp", "")
            subject = metadata.get("thread_subject") or metadata.get("email_subject", "No Subject")
            
            # Build header based on chunk type
            if chunk_type == "message":
                header = f"[EMAIL from {sender} | {timestamp} | Thread: {subject}]"
            elif chunk_type == "attachment":
                filename = metadata.get("filename", "Unknown")
                header = f"[DOCUMENT: {filename} | Sent by {sender} on {timestamp} | Thread: {subject}]"
            else:
                header = f"[{chunk_type.upper()}]"
            
            context_parts.append(f"--- SOURCE {i} (relevance: {r['score']:.2f}) ---\n{header}\n\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, project_id: str):
        """Use LLM to generate answer from context (Streaming)."""
        
        # Get current date for context
        current_date = datetime.now().strftime("%B %d, %Y")
        
        system_prompt = f"""You are an expert assistant helping users find information in their project emails and documents.

PROJECT: {project_id.replace('_', ' ').title()}
TODAY'S DATE: {current_date}

Your job is to answer questions based ONLY on the provided context. The context contains:
- EMAIL messages (with sender, date, subject, body)
- DOCUMENT content (AI-extracted content from PDFs like invoices, drawings, reports)

**IMPORTANT - RECENCY PRIORITY:**
In construction projects, the LATEST information is usually the most accurate and relevant.
- When multiple sources contain similar information, PREFER the most recent one
- Look at the timestamps in the sources to identify which is newest
- If there are conflicting details (e.g., different prices, specs, quantities), use the LATEST source
- Explicitly mention the date when citing sources so the user knows how current the info is

Guidelines:
1. Answer based ONLY on the provided context - don't make up information
2. Be specific - include exact numbers, dates, names, and quotes when available
3. Cite your sources clearly:
   - When citing a DOCUMENT, explicitly mention who sent it and when (e.g., "According to Invoice.pdf sent by Bob on Jan 12th...")
   - When citing an EMAIL, mention the sender and date
4. If the context doesn't contain the answer, say so clearly
5. For financial questions, be precise with dollar amounts
6. For date questions, provide the exact dates found
7. Keep answers concise but complete
8. When information has changed over time, highlight the LATEST value and note any previous values if relevant

If asked about "latest" or "most recent", look at the timestamps in the sources to identify the newest."""

        user_prompt = f"""CONTEXT FROM PROJECT EMAILS AND DOCUMENTS:

{context}

---

QUESTION: {question}

Please provide a clear, specific answer based on the context above. Cite relevant sources."""

        # Generate answer using Gemini
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.gemini_model.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config={
                    "temperature": GEMINI_TEMPERATURE
                }
            )
            return response
            
        except ImportError:
            return "Error: `google-genai` package not installed. Run `pip install google-genai`."
        except Exception as e:
            return f"Gemini Error: {str(e)}"
    
    def list_projects(self) -> List[str]:
        """List all project namespaces in the index."""
        # Pinecone doesn't have a direct way to list namespaces
        # This is a workaround using index stats
        stats = self.pinecone_index.describe_index_stats()
        namespaces = list(stats.get("namespaces", {}).keys())
        return namespaces
    
    def get_project_stats(self, project_id: str) -> Dict:
        """Get stats for a project namespace."""
        # Normalize project_id to match Pinecone namespace format
        project_id = normalize_project_id(project_id)
        stats = self.pinecone_index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(project_id, {})
        return {
            "project_id": project_id,
            "vector_count": ns_stats.get("vector_count", 0)
        }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def ask_project(
    project_id: str,
    question: str,
    pinecone_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> SearchResult:
    """
    Convenience function to ask a question about a project.
    
    Args:
        project_id: Project namespace (e.g., "88_supermarket" or "88 SuperMarket")
        question: The question to ask
        pinecone_api_key: Optional Pinecone API key
        openai_api_key: Optional OpenAI API key
    
    Returns:
        SearchResult
    """
    # Note: project_id is normalized inside search.ask()
    search = ProjectSearch(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key
    )
    return search.ask(project_id, question)


def interactive_search(project_id: str):
    """
    Interactive search session for a project.
    
    Args:
        project_id: Project namespace to search
    """
    # Normalize project_id to match Pinecone namespace format
    project_id = normalize_project_id(project_id)
    
    search = ProjectSearch()
    
    print(f"\n{'='*60}")
    print(f"🔍 INTERACTIVE SEARCH: {project_id.replace('_', ' ').title()}")
    print(f"{'='*60}")
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            result = search.ask(project_id, question)
            
            print(f"\n{'─'*50}")
            print(f"📝 ANSWER:\n")
            
            # Stream answer
            full_answer = ""
            print("   (Generating answer...)\n") # Show visual indicator
            answer_start_time = time.time() # Start timing answer generation
            
            if isinstance(result.answer, str):
                print(result.answer)
            else:
                # Handle Gemini stream
                for chunk in result.answer:
                    text = chunk.text
                    if text:
                        print(text, end="", flush=True)
                        full_answer += text
                print()
            
            answer_time_ms = int((time.time() - answer_start_time) * 1000) # Calculate duration
            
            print(f"\n{'─'*50}")
            print(f"📊 Stats: {result.chunks_retrieved} chunks | Total: {result.total_time_ms}ms | Answer Gen: {answer_time_ms}ms")
            
            if result.sources:
                print(f"\n📎 Top Sources:")
                for i, src in enumerate(result.sources[:3], 1):
                    src_type = src.chunk_type
                    sender = src.metadata.get('sender_name') or src.metadata.get('sender_email', 'Unknown')
                    ts = src.metadata.get("timestamp", "")[:10]
                    
                    if src_type == "message":
                        subject = src.metadata.get('thread_subject') or src.metadata.get('email_subject') or "No Subject"
                        info = f"Email: '{subject}' from {sender} ({ts})"
                    else:
                        filename = src.metadata.get('filename', 'Unknown')
                        info = f"Document: {filename} (via {sender}, {ts})"
                    print(f"   {i}. [{src.score:.2f}] {info}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Search project data for Donna Email Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question:
  python project_search.py 88_supermarket "What's the pile length?"
  
  # Interactive mode:
  python project_search.py 88_supermarket --interactive
  
  # List all projects:
  python project_search.py --list
  
  # Get project stats:
  python project_search.py 88_supermarket --stats
        """
    )
    
    parser.add_argument("project_id", nargs="?", help="Project namespace (e.g., 88_supermarket)")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive search mode")
    parser.add_argument("--list", "-l", action="store_true", help="List all projects")
    parser.add_argument("--stats", "-s", action="store_true", help="Show project stats")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve")
    
    args = parser.parse_args()
    
    # List projects
    if args.list:
        search = ProjectSearch()
        projects = search.list_projects()
        print(f"\n📂 Projects in index '{PINECONE_INDEX_NAME}':")
        if projects:
            for p in projects:
                stats = search.get_project_stats(p)
                print(f"   • {p} ({stats['vector_count']} vectors)")
        else:
            print("   (no projects found)")
        sys.exit(0)
    
    # Require project_id for other operations
    if not args.project_id:
        parser.print_help()
        print("\nError: project_id required")
        sys.exit(1)
    
    # Show stats
    if args.stats:
        search = ProjectSearch()
        stats = search.get_project_stats(args.project_id)
        print(f"\n📊 Stats for '{args.project_id}':")
        print(f"   Vectors: {stats['vector_count']}")
        sys.exit(0)
    
    # Interactive mode
    if args.interactive:
        interactive_search(args.project_id)
        sys.exit(0)
    
    # Single question
    if not args.question:
        parser.print_help()
        print("\nError: question required (or use --interactive)")
        sys.exit(1)
    
    result = ask_project(args.project_id, args.question)
    
    print(f"\n{'='*60}")
    print(f"📝 ANSWER")
    print(f"{'='*60}\n")
    
    # Stream the answer to console
    full_answer = ""
    if isinstance(result.answer, str):
        print(result.answer)
    else:
        # Handle Gemini stream
        for chunk in result.answer:
            text = chunk.text
            if text:
                print(text, end="", flush=True)
                full_answer += text
        print()
        
    print(f"\n{'='*60}")
    print(f"📊 Retrieved {result.chunks_retrieved} chunks in {result.total_time_ms}ms")
    print(f"   Search: {result.search_time_ms}ms")
    
    if result.sources:
        print(f"\n📎 Sources used:")
        for i, src in enumerate(result.sources, 1):
            src_type = src.chunk_type
            ts = src.metadata.get("timestamp", "")[:10]
            sender = src.metadata.get('sender_name') or src.metadata.get('sender_email', 'Unknown')
            
            if src_type == "message":
                subject = src.metadata.get('thread_subject') or src.metadata.get('email_subject') or "No Subject"
                print(f"   {i}. [EMAIL] '{subject}' from {sender} ({ts}) - score: {src.score:.2f}")
            else:
                filename = src.metadata.get('filename', 'Unknown')
                print(f"   {i}. [DOC] {filename} (via {sender}, {ts}) - score: {src.score:.2f}")


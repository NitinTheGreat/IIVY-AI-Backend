"""
Azure Function: search_project_stream

HTTP Trigger with SSE streaming for real-time search responses.
Like ChatGPT/Gemini - sends "thinking" status, then streams answer tokens.

CONVERSATION-AWARE RAG:
- Accepts conversation context (summary + recent messages)
- Rewrites follow-up questions to standalone queries
- Passes conversation context to answer LLM for consistency

NOTE: Azure Functions v1 model buffers responses. For true streaming,
the frontend calls this endpoint DIRECTLY (bypassing Database Backend proxy)
which removes one buffering layer and speeds up response delivery.
"""
import logging
import sys
import os
import json
import time
from typing import Generator, Optional, List, Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from project_search import ProjectSearch, LLM_PROVIDER
from conversation import ConversationManager, RewriteResult


def format_sse(event: str, data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_search_generator(
    project_id: str,
    question: str,
    top_k: int = None,
    filter_metadata: dict = None,
    # NEW: Conversation context
    summary: Optional[str] = None,
    recent_messages: Optional[List[Dict]] = None,
    project_name: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Generator that yields SSE events for streaming search.
    
    Events:
    - thinking: Status updates during processing
    - rewrite: Original and standalone question (for debugging)
    - sources: Retrieved sources (sent before answer)
    - chunk: Answer text chunks
    - done: Final timing stats
    - error: Error information
    
    Args:
        project_id: Pinecone namespace for the project
        question: User's question (may be a follow-up)
        top_k: Number of chunks to retrieve
        filter_metadata: Optional Pinecone filters
        summary: Conversation summary for context
        recent_messages: Recent messages for context
        project_name: Project name for prompts
    """
    total_start = time.time()
    rewrite_result: Optional[RewriteResult] = None
    standalone_question = question  # Default to original
    
    try:
        # ================================================================
        # STEP 0: Rewrite + Classify question (if conversation context exists)
        # ================================================================
        question_type = "rag"  # Default to RAG
        
        if summary or recent_messages:
            yield format_sse("thinking", {"status": "🧠 Understanding your question..."})
            
            try:
                conv_manager = ConversationManager()
                rewrite_result = conv_manager.rewrite_question(
                    question=question,
                    summary=summary,
                    recent_messages=recent_messages or [],
                    project_name=project_name or project_id.replace('_', ' ').title()
                )
                standalone_question = rewrite_result.standalone_question
                question_type = rewrite_result.question_type
                
                # Send rewrite info to frontend (for debugging/transparency)
                yield format_sse("rewrite", {
                    "original": rewrite_result.original_question,
                    "standalone": rewrite_result.standalone_question,
                    "was_rewritten": rewrite_result.was_rewritten,
                    "question_type": question_type
                })
                
                if rewrite_result.was_rewritten:
                    logging.info(f"Rewrite: '{question}' -> '{standalone_question}' (type={question_type})")
                else:
                    logging.info(f"Question classified as: {question_type}")
                    
            except Exception as e:
                logging.warning(f"Rewrite failed, using original question with RAG: {e}")
                # Continue with original question and RAG type
        
        # ================================================================
        # CONVERSATION-ONLY PATH: Skip vector search, answer from chat history
        # ================================================================
        if question_type == "conversation":
            yield format_sse("thinking", {"status": "📝 Analyzing conversation..."})
            
            # No sources for conversation questions
            yield format_sse("sources", {
                "sources": [],
                "chunks_retrieved": 0,
                "search_time_ms": 0
            })
            
            llm_start = time.time()
            
            # Build conversation context for the LLM
            conversation_context = ""
            if summary:
                conversation_context += f"CONVERSATION SUMMARY:\n{summary}\n\n"
            if recent_messages:
                conversation_context += "RECENT MESSAGES:\n"
                for msg in (recent_messages or [])[-10:]:  # Last 10 messages
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    conversation_context += f"{role}: {msg.get('content', '')}\n\n"
            
            system_prompt = f"""You are an assistant helping users recall and summarize their conversation.

PROJECT: {project_name or project_id.replace('_', ' ').title()}

Your task is to answer questions about the conversation itself - summarizing what was discussed,
recalling specific points, or clarifying previous answers.

Be specific - include exact details, names, numbers, and findings that were mentioned."""

            user_prompt = f"""{conversation_context}

USER'S QUESTION: {standalone_question}

Provide a helpful response based on the conversation above."""

            # Stream response using Gemini (ProjectSearch and LLM_PROVIDER imported at top)
            search = ProjectSearch()
            
            if LLM_PROVIDER == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = search.gemini_model.models.generate_content_stream(
                    model="gemini-3-flash-preview",
                    contents=full_prompt,
                    config={"temperature": 0.3}
                )
                for chunk in response:
                    if chunk.text:
                        yield format_sse("chunk", {"text": chunk.text})
            else:
                response = search.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield format_sse("chunk", {"text": chunk.choices[0].delta.content})
            
            llm_time_ms = int((time.time() - llm_start) * 1000)
            total_time_ms = int((time.time() - total_start) * 1000)
            
            done_data = {
                "search_time_ms": 0,
                "llm_time_ms": llm_time_ms,
                "total_time_ms": total_time_ms,
                "chunks_retrieved": 0,
                "question_type": "conversation"
            }
            
            if rewrite_result:
                done_data["rewrite"] = {
                    "original": rewrite_result.original_question,
                    "standalone": rewrite_result.standalone_question,
                    "was_rewritten": rewrite_result.was_rewritten,
                    "question_type": question_type
                }
            
            yield format_sse("done", done_data)
            return  # Exit early - no RAG needed
        
        # ================================================================
        # RAG PATH: Full vector search pipeline
        # ================================================================
        yield format_sse("thinking", {"status": "🔍 Searching project data..."})
        
        # Create search instance
        search = ProjectSearch()
        
        # Check if namespace exists
        available_namespaces = search.list_projects()
        logging.info(f'stream_search: project_id="{project_id}", available={available_namespaces}')
        
        if project_id not in available_namespaces:
            # Try to find a matching namespace (in case of format mismatch)
            matching = [ns for ns in available_namespaces if ns.startswith(project_id.split('_')[0])]
            if matching:
                yield format_sse("error", {
                    "message": f"Project '{project_id}' not found. Did you mean: {matching}? The project may have been indexed with a different ID format."
                })
            else:
                yield format_sse("error", {
                    "message": f"Project '{project_id}' not found in database. Available projects: {available_namespaces}"
                })
            return
        
        # Determine top_k based on question type (use standalone question for better detection)
        if top_k is None:
            top_k = search._determine_top_k(standalone_question)
        
        # ================================================================
        # STEP 1: Embed the question (use STANDALONE question for retrieval)
        # ================================================================
        search_start = time.time()
        yield format_sse("thinking", {"status": "🧠 Embedding query..."})
        question_embedding = search._embed_question(standalone_question)
        
        # ================================================================
        # STEP 2: Search Pinecone
        # ================================================================
        yield format_sse("thinking", {"status": "📚 Retrieving relevant documents..."})
        raw_results = search._search_pinecone(
            embedding=question_embedding,
            namespace=project_id,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        search_time_ms = int((time.time() - search_start) * 1000)
        
        if not raw_results:
            yield format_sse("thinking", {"status": "No relevant documents found"})
            yield format_sse("chunk", {"text": "I couldn't find any relevant information for this question in the project data."})
            yield format_sse("done", {
                "search_time_ms": search_time_ms,
                "llm_time_ms": 0,
                "total_time_ms": int((time.time() - total_start) * 1000),
                "chunks_retrieved": 0
            })
            return
        
        # Sort by timestamp if "latest" query
        if search._is_latest_query(standalone_question):
            raw_results = search._sort_by_timestamp(raw_results)
        
        # ================================================================
        # STEP 3: Build sources for early return
        # ================================================================
        sources = []
        for src in raw_results[:5]:  # Top 5 sources
            sources.append({
                "chunk_id": src["id"],
                "chunk_type": src["metadata"].get("chunk_type", "unknown"),
                "text": src["metadata"].get("text", "")[:300] + "..." if len(src["metadata"].get("text", "")) > 300 else src["metadata"].get("text", ""),
                "score": src["score"],
                "sender": src["metadata"].get("sender_name") or src["metadata"].get("sender_email", "Unknown"),
                "timestamp": src["metadata"].get("timestamp", "")[:10],
                "subject": src["metadata"].get("thread_subject") or src["metadata"].get("email_subject", ""),
                "file_id": src["metadata"].get("file_id", ""),
                "page": src["metadata"].get("page", "")
            })
        
        # Yield sources early (so frontend can show them while answer generates)
        yield format_sse("sources", {
            "sources": sources,
            "chunks_retrieved": len(raw_results),
            "search_time_ms": search_time_ms
        })
        
        # ================================================================
        # STEP 4: Build context for LLM
        # ================================================================
        context = search._build_context(raw_results)
        
        # ================================================================
        # STEP 5: Generate answer with LLM (with conversation context)
        # ================================================================
        yield format_sse("thinking", {"status": f"✨ Generating answer from {len(raw_results)} sources..."})
        
        llm_start = time.time()
        
        # Build system prompt
        system_prompt = f"""You are an expert assistant helping users find information in their project emails and documents.

PROJECT: {project_name or project_id.replace('_', ' ').title()}

Your job is to answer questions based ONLY on the provided context. The context contains:
- EMAIL messages (with sender, date, subject, body)
- DOCUMENT content (AI-extracted content from PDFs like invoices, drawings, reports)

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

If asked about "latest" or "most recent", look at the timestamps in the sources to identify the newest."""

        # Build user prompt with conversation context for Claude-like accuracy
        conversation_context = ""
        if summary or recent_messages:
            conversation_context = "\n--- CONVERSATION CONTEXT ---\n"
            
            if summary:
                conversation_context += f"Summary of prior conversation:\n{summary}\n\n"
            
            if recent_messages:
                conversation_context += "Recent exchanges:\n"
                for msg in (recent_messages or [])[-4:]:  # Last 2 Q&A pairs
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")[:600]
                    if len(msg.get("content", "")) > 600:
                        content += "..."
                    conversation_context += f"{role}: {content}\n\n"
            
            conversation_context += "(The user's current question may reference this conversation.)\n--- END CONVERSATION CONTEXT ---\n\n"

        user_prompt = f"""{conversation_context}CONTEXT FROM PROJECT EMAILS AND DOCUMENTS:

{context}

---

QUESTION: {standalone_question}
{f"(Original phrasing: {question})" if rewrite_result and rewrite_result.was_rewritten else ""}

Please provide a clear, specific answer based on the context above. Cite relevant sources."""

        # Stream from LLM
        if LLM_PROVIDER == "gemini":
            # Gemini streaming
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = search.gemini_model.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=full_prompt,
                config={"temperature": 0.2}
            )
            for chunk in response:
                if chunk.text:
                    yield format_sse("chunk", {"text": chunk.text})
        else:
            # OpenAI streaming
            response = search.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield format_sse("chunk", {"text": chunk.choices[0].delta.content})
        
        llm_time_ms = int((time.time() - llm_start) * 1000)
        total_time_ms = int((time.time() - total_start) * 1000)
        
        # ================================================================
        # STEP 6: Yield final done event with full debug info
        # ================================================================
        done_data = {
            "search_time_ms": search_time_ms,
            "llm_time_ms": llm_time_ms,
            "total_time_ms": total_time_ms,
            "chunks_retrieved": len(raw_results),
            "question_type": "rag"
        }
        
        # Include rewrite info in done event for persistence
        if rewrite_result:
            done_data["rewrite"] = {
                "original": rewrite_result.original_question,
                "standalone": rewrite_result.standalone_question,
                "was_rewritten": rewrite_result.was_rewritten,
                "question_type": question_type
            }
        
        yield format_sse("done", done_data)
        
    except Exception as e:
        logging.error(f"search_project_stream: Error - {str(e)}")
        yield format_sse("error", {"message": str(e)})


# CORS headers for frontend direct access
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",  # Allow any origin in dev; restrict in production
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Stream search results as Server-Sent Events.
    
    POST Body:
    {
        "project_id": "88_supermarket",
        "question": "What is the scope of work?",
        "top_k": 50,  // Optional
        // NEW: Conversation context for follow-ups
        "summary": "## Current Focus\\n...",  // Optional
        "recent_messages": [  // Optional
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "project_name": "88 Supermarket"  // Optional
    }
    
    Returns: SSE stream with events: thinking, rewrite, sources, chunk, done, error
    
    CORS enabled so frontend can call this DIRECTLY (bypassing Database Backend).
    This removes one buffering layer for faster response delivery.
    """
    logging.info('search_project_stream: Processing streaming request')
    
    # Handle CORS preflight
    if req.method == "OPTIONS":
        return func.HttpResponse(
            body="",
            status_code=204,
            headers=CORS_HEADERS
        )
    
    # Parse request
    try:
        body = req.get_json()
    except ValueError:
        error_response = format_sse("error", {"message": "Request body must be valid JSON"})
        return func.HttpResponse(
            body=error_response,
            status_code=400,
            mimetype="text/event-stream",
            headers={
                **CORS_HEADERS,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    project_id = body.get("project_id")
    question = body.get("question")
    
    if not project_id or not question:
        error_response = format_sse("error", {"message": "Missing required parameters: project_id and question"})
        return func.HttpResponse(
            body=error_response,
            status_code=400,
            mimetype="text/event-stream",
            headers={
                **CORS_HEADERS,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    # Optional parameters
    top_k = body.get("top_k")
    filter_metadata = body.get("filter_metadata")
    
    # NEW: Conversation context parameters
    summary = body.get("summary")
    recent_messages = body.get("recent_messages")
    project_name = body.get("project_name")
    
    logging.info(f'search_project_stream: project={project_id}, question="{question[:50]}...", has_summary={bool(summary)}, messages={len(recent_messages) if recent_messages else 0}')
    
    # Collect all SSE events from generator
    # NOTE: v1 model buffers, but frontend calls directly = 1 less hop
    sse_events = ""
    for event in stream_search_generator(
        project_id=project_id,
        question=question,
        top_k=top_k,
        filter_metadata=filter_metadata,
        summary=summary,
        recent_messages=recent_messages,
        project_name=project_name
    ):
        sse_events += event
    
    return func.HttpResponse(
        body=sse_events,
        status_code=200,
        mimetype="text/event-stream",
        headers={
            **CORS_HEADERS,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

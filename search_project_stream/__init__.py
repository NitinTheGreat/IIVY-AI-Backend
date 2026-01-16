"""
Azure Function: search_project_stream

HTTP Trigger with SSE streaming for real-time search responses.
Like ChatGPT/Gemini - sends "thinking" status, then streams answer tokens.

NOTE: Uses manual SSE streaming compatible with Azure Functions v1 model.
"""
import logging
import sys
import os
import json
import time
from typing import Generator

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from project_search import ProjectSearch, LLM_PROVIDER


def format_sse(event: str, data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_search_generator(project_id: str, question: str, top_k: int = None, filter_metadata: dict = None) -> Generator[str, None, None]:
    """
    Generator that yields SSE events for streaming search.
    
    Events:
    - thinking: Status updates during processing
    - sources: Retrieved sources (sent before answer)
    - chunk: Answer text chunks
    - done: Final timing stats
    - error: Error information
    """
    total_start = time.time()
    
    try:
        # Yield initial thinking status
        yield format_sse("thinking", {"status": "🔍 Searching project data..."})
        
        # Create search instance
        search = ProjectSearch()
        
        # Determine top_k based on question type
        if top_k is None:
            top_k = search._determine_top_k(question)
        
        # Step 1: Embed the question
        search_start = time.time()
        yield format_sse("thinking", {"status": "🧠 Understanding your question..."})
        question_embedding = search._embed_question(question)
        
        # Step 2: Search Pinecone
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
        if search._is_latest_query(question):
            raw_results = search._sort_by_timestamp(raw_results)
        
        # Build sources for early return
        sources = []
        for src in raw_results[:5]:  # Top 5 sources
            sources.append({
                "chunk_id": src["id"],
                "chunk_type": src["metadata"].get("chunk_type", "unknown"),
                "text": src["metadata"].get("text", "")[:300] + "..." if len(src["metadata"].get("text", "")) > 300 else src["metadata"].get("text", ""),
                "score": src["score"],
                "sender": src["metadata"].get("sender_name") or src["metadata"].get("sender_email", "Unknown"),
                "timestamp": src["metadata"].get("timestamp", "")[:10],
                "subject": src["metadata"].get("thread_subject") or src["metadata"].get("email_subject", "")
            })
        
        # Yield sources early (so frontend can show them while answer generates)
        yield format_sse("sources", {
            "sources": sources,
            "chunks_retrieved": len(raw_results),
            "search_time_ms": search_time_ms
        })
        
        # Step 3: Build context for LLM
        context = search._build_context(raw_results)
        
        # Step 4: Generate answer with LLM (streaming)
        yield format_sse("thinking", {"status": f"✨ Generating answer from {len(raw_results)} sources..."})
        
        llm_start = time.time()
        
        # Build prompts
        system_prompt = f"""You are an expert assistant helping users find information in their project emails and documents.

PROJECT: {project_id.replace('_', ' ').title()}

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

        user_prompt = f"""CONTEXT FROM PROJECT EMAILS AND DOCUMENTS:

{context}

---

QUESTION: {question}

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
        
        # Yield final done event
        yield format_sse("done", {
            "search_time_ms": search_time_ms,
            "llm_time_ms": llm_time_ms,
            "total_time_ms": total_time_ms,
            "chunks_retrieved": len(raw_results)
        })
        
    except Exception as e:
        logging.error(f"search_project_stream: Error - {str(e)}")
        yield format_sse("error", {"message": str(e)})


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Stream search results as Server-Sent Events.
    
    POST Body:
    {
        "project_id": "88_supermarket",
        "question": "What is the scope of work?",
        "top_k": 50  // Optional
    }
    
    Returns: SSE stream with events: thinking, sources, chunk, done, error
    
    NOTE: Azure Functions v1 model doesn't support true streaming, so this
    collects all SSE events and returns them as a complete response.
    The frontend should still parse them as SSE events line-by-line.
    For true real-time streaming, migrate to v2 model with FastAPI extension.
    """
    logging.info('search_project_stream: Processing streaming request')
    
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
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    top_k = body.get("top_k")
    filter_metadata = body.get("filter_metadata")
    
    # Collect all SSE events from generator
    # NOTE: In v1 model, we can't do true streaming, but we return SSE format
    # so the frontend can consume it the same way as true streaming
    sse_events = ""
    for event in stream_search_generator(project_id, question, top_k, filter_metadata):
        sse_events += event
    
    return func.HttpResponse(
        body=sse_events,
        status_code=200,
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

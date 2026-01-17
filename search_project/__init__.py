"""
Azure Function: search_project

HTTP Trigger that wraps ProjectSearch to perform RAG search and generate answers.
"""
import logging
import sys
import os
from dataclasses import asdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from shared import create_response, create_error_response, parse_request_body, get_required_param
from project_search import ProjectSearch


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search a project and generate an answer.
    
    POST Body:
    {
        "project_id": "88_supermarket",
        "question": "What is the scope of work?",
        "top_k": 50  // Optional: number of chunks to retrieve
    }
    
    Returns:
    {
        "project_id": "88_supermarket",
        "question": "What is the scope of work?",
        "answer": "The scope of work includes...",
        "sources": [ { "chunk_id": "...", "text": "...", "score": 0.89, ... } ],
        "chunks_retrieved": 50,
        "search_time_ms": 120,
        "llm_time_ms": 2500,
        "total_time_ms": 2620
    }
    """
    logging.info('search_project: Processing request')
    
    # Parse request body
    body = parse_request_body(req)
    if not body:
        return create_error_response("Request body must be valid JSON")
    
    # Get required parameters
    project_id, err = get_required_param(body, "project_id")
    if err:
        return err
    
    question, err = get_required_param(body, "question")
    if err:
        return err
    
    # Optional parameters
    top_k = body.get("top_k")
    filter_metadata = body.get("filter_metadata")
    
    try:
        # Create search instance
        search = ProjectSearch()
        
        # Debug: Log what project_id is received and what namespaces exist
        logging.info(f'search_project: Received project_id="{project_id}"')
        available_namespaces = search.list_projects()
        logging.info(f'search_project: Available namespaces in Pinecone: {available_namespaces}')
        
        if project_id not in available_namespaces:
            logging.warning(f'search_project: project_id "{project_id}" NOT FOUND in Pinecone namespaces!')
        
        # Run search
        logging.info(f'search_project: Searching "{project_id}" for: {question[:50]}...')
        result = search.ask(
            project_id=project_id,
            question=question,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # The answer may be a generator (streaming), collect it into a string
        answer = result.answer
        if hasattr(answer, '__iter__') and not isinstance(answer, str):
            answer = ''.join(answer)
        
        # Convert sources to serializable format
        sources = []
        for src in result.sources:
            sources.append({
                "chunk_id": src.chunk_id,
                "chunk_type": src.chunk_type,
                "text": src.text[:500] + "..." if len(src.text) > 500 else src.text,
                "score": src.score,
                "metadata": src.metadata
            })
        
        return create_response({
            "project_id": result.project_id,
            "question": result.question,
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": result.chunks_retrieved,
            "search_time_ms": result.search_time_ms,
            "llm_time_ms": result.llm_time_ms,
            "total_time_ms": result.total_time_ms
        })
        
    except Exception as e:
        logging.error(f'search_project: Error - {str(e)}')
        return create_error_response(
            f"Search failed: {str(e)}",
            status_code=500
        )

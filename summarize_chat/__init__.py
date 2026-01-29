"""
Azure Function: summarize_chat

HTTP Trigger to generate or update a conversation summary.
Called by the Database Backend when conversation needs compression.

Triggers:
1. Boot summary at message 4
2. Regular update every 8 messages
3. Token threshold exceeded
"""
import logging
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import azure.functions as func

from conversation import ConversationManager, SummaryResult


# CORS headers for backend access
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate or update a conversation summary.
    
    POST Body:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "existing_summary": "## Current Focus\\n...",  // Optional
        "project_name": "88 Supermarket"  // Optional
    }
    
    Returns:
    {
        "summary": "## Current Focus\\n...",
        "word_count": 180,
        "entities_preserved": 12
    }
    """
    logging.info('summarize_chat: Processing request')
    
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
        return func.HttpResponse(
            body=json.dumps({"error": "Request body must be valid JSON"}),
            status_code=400,
            mimetype="application/json",
            headers=CORS_HEADERS
        )
    
    messages = body.get("messages")
    if not messages:
        return func.HttpResponse(
            body=json.dumps({"error": "Missing required parameter: messages"}),
            status_code=400,
            mimetype="application/json",
            headers=CORS_HEADERS
        )
    
    existing_summary = body.get("existing_summary")
    project_name = body.get("project_name", "")
    
    logging.info(f'summarize_chat: {len(messages)} messages, has_existing={bool(existing_summary)}')
    
    try:
        manager = ConversationManager()
        result: SummaryResult = manager.generate_summary(
            messages=messages,
            existing_summary=existing_summary,
            project_name=project_name
        )
        
        return func.HttpResponse(
            body=json.dumps({
                "summary": result.summary,
                "word_count": result.word_count,
                "entities_preserved": result.entities_preserved
            }),
            status_code=200,
            mimetype="application/json",
            headers=CORS_HEADERS
        )
        
    except Exception as e:
        logging.error(f'summarize_chat: Error - {str(e)}')
        return func.HttpResponse(
            body=json.dumps({"error": f"Summary generation failed: {str(e)}"}),
            status_code=500,
            mimetype="application/json",
            headers=CORS_HEADERS
        )

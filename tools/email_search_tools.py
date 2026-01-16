"""
Smart Email Search Tool for Donna SUPERHUMAN.

This tool provides a Perplexity-style search over Outlook inbox + attachments.
It internally runs a LangGraph subgraph (email_search_graph) that:
1. Generates multiple search queries
2. Searches emails in parallel
3. Groups by threads
4. Skims PDF attachments
5. Deep-reads most relevant attachments
6. Synthesizes a final answer

The tool returns structured results that Donna can use for follow-up questions.
"""
from typing import TypedDict, List, Dict, Any, Optional


# ============================================================
# TYPE DEFINITIONS (external API)
# ============================================================

class AttachmentSummary(TypedDict):
    """Summary of an attachment in search results."""
    message_id: str
    attachment_id: str
    file_name: str
    size_bytes: int
    content_type: str
    skim_summary: Optional[str]
    deep_used: bool


class ThreadSummary(TypedDict):
    """Summary of an email thread in search results."""
    thread_id: str
    subject: str
    participants: List[str]
    latest_message_preview: str
    received_at: str
    score: float
    message_ids: List[str]
    attachments: List[AttachmentSummary]


class SmartEmailSearchResult(TypedDict):
    """Result from smart email search."""
    final_answer: str
    threads_used: List[ThreadSummary]
    all_threads_considered: List[ThreadSummary]
    debug_info: Dict[str, Any]


# ============================================================
# MAIN TOOL FUNCTION
# ============================================================

def smart_email_search(query: str) -> SmartEmailSearchResult:
    """
    Smart email search - Perplexity for your Outlook inbox.
    
    This function runs a LangGraph subgraph that intelligently searches
    your emails and attachments to answer natural language questions.
    
    Examples:
    - "What did Harv say about the Richmond project schedule last week?"
    - "Did we ever confirm the rate with BuildSmartr for detailing work?"
    - "Summarize all emails from Sarah about change orders for Surrey job."
    
    Args:
        query: Natural language question about emails/attachments
    
    Returns:
        SmartEmailSearchResult with:
        - final_answer: Direct answer to the question
        - threads_used: Top 3-5 threads that contributed to the answer
        - all_threads_considered: All 15-20 candidate threads
        - debug_info: Search metadata (queries used, counts, etc.)
    """
    from email_search_graph import get_email_search_graph, SearchState
    
    print("\n" + "="*70)
    print("🔍 SMART EMAIL SEARCH")
    print("="*70)
    print(f"Query: {query}\n")
    
    # Get the email search subgraph
    search_graph = get_email_search_graph()
    
    # Initialize state
    initial_state: SearchState = {
        "user_query": query,
        "final_answer": None,
        "threads_used": [],
        "all_threads_considered": [],
        "debug_info": {}
    }
    
    # Run the subgraph
    try:
        final_state = search_graph.invoke(initial_state)
        
        # Extract results
        result: SmartEmailSearchResult = {
            "final_answer": final_state.get("final_answer") or "No answer generated.",
            "threads_used": final_state.get("threads_used", []),
            "all_threads_considered": final_state.get("all_threads_considered", []),
            "debug_info": final_state.get("debug_info", {})
        }
        
        # Print detailed summary
        print(format_search_result(result))
        
        return result
    
    except Exception as e:
        import traceback
        error_result: SmartEmailSearchResult = {
            "final_answer": f"Search failed: {str(e)}",
            "threads_used": [],
            "all_threads_considered": [],
            "debug_info": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }
        return error_result


# ============================================================
# HELPER FOR PRETTY-PRINTING RESULTS
# ============================================================

def format_search_result(result: SmartEmailSearchResult) -> str:
    """
    Format a SmartEmailSearchResult as a human-readable string.
    
    This is useful for displaying results to the user.
    """
    output = []
    
    output.append("\n" + "="*70)
    output.append("📧 EMAIL SEARCH RESULTS")
    output.append("="*70 + "\n")
    
    # Show debug info first (what was searched)
    if result['debug_info']:
        output.append("🔍 **Search Summary:**")
        
        queries = result['debug_info'].get('queries_used', [])
        if queries:
            queries_str = ', '.join([f'"{q}"' for q in queries])
            output.append(f"   Queries: {queries_str}")
        
        total_msgs = result['debug_info'].get('total_messages_found', 0)
        output.append(f"   Messages found: {total_msgs}")
        
        pdfs_skimmed = result['debug_info'].get('pdfs_skimmed', 0)
        pdfs_deep = result['debug_info'].get('pdfs_deep_read', 0)
        if pdfs_skimmed > 0 or pdfs_deep > 0:
            output.append(f"   PDFs analyzed: {pdfs_skimmed} skimmed, {pdfs_deep} deep-read")
        
        iterations = result['debug_info'].get('iterations', 0)
        output.append(f"   Agent iterations: {iterations}")
        
        # Show sample of emails found
        emails_sample = result['debug_info'].get('emails_sample', [])
        if emails_sample:
            output.append(f"\n📨 **Emails Analyzed (sample):**")
            for i, email in enumerate(emails_sample[:5], 1):
                output.append(f"   {i}. {email['subject'][:50]}")
                output.append(f"      From: {email['from']}, {email['received']}")
                if email.get('has_attachments'):
                    output.append(f"      📎 Has attachments")
        
        output.append("")
    
    # Show the answer
    output.append("💬 **Answer:**")
    output.append(result['final_answer'])
    output.append("")
    
    # Show threads used (if populated)
    if result['threads_used']:
        output.append(f"🧵 **Threads Used ({len(result['threads_used'])}):**")
        for i, thread in enumerate(result['threads_used'], 1):
            output.append(f"\n{i}. **{thread['subject']}**")
            output.append(f"   - Date: {thread['received_at']}")
            output.append(f"   - Participants: {', '.join(thread['participants'][:3])}")
            output.append(f"   - Score: {thread['score']:.2f}")
            if thread['attachments']:
                output.append(f"   - Attachments ({len(thread['attachments'])}):")
                for att in thread['attachments'][:3]:
                    output.append(f"     • {att['file_name']}")
                    if att.get('skim_summary'):
                        output.append(f"       Summary: {att['skim_summary'][:80]}...")
    
    output.append("\n" + "="*70)
    
    return "\n".join(output)


"""
Conversation Memory Module for Follow-up Questions.

This module handles:
1. Query rewriting - Converting follow-up questions to standalone queries
2. Conversation summarization - Compressing conversation history
3. Entity preservation - Ensuring key facts survive compression

Architecture:
- Summary + recent messages = what the user means (conversation context)
- Vectors = what's true (project facts)
- Rewrite step = connects meaning → retrieval

Usage:
    from conversation import ConversationManager
    
    manager = ConversationManager()
    
    # Rewrite a follow-up question
    standalone = manager.rewrite_question(
        question="what about the second one?",
        summary="...",
        recent_messages=[...]
    )
    
    # Generate/update summary
    new_summary = manager.generate_summary(messages=[...], existing_summary=None)
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Summary constraints
SUMMARY_MAX_WORDS = 400
SUMMARY_TARGET_WORDS = 250

# Message window for rewriter (last N messages)
REWRITER_MESSAGE_WINDOW = 10

# When to trigger re-summarization
SUMMARY_MESSAGE_INTERVAL = 8  # Every N messages
SUMMARY_TOKEN_THRESHOLD = 2000  # Or when tokens exceed this

# LLM settings
REWRITER_MODEL = "gemini-2.0-flash"  # Fast model for rewriting
SUMMARIZER_MODEL = "gemini-2.0-flash"  # Fast model for summarization
REWRITER_TEMPERATURE = 0.1  # Low temp for deterministic rewrites
SUMMARIZER_TEMPERATURE = 0.2


# ============================================================
# SUMMARY SCHEMA
# ============================================================

SUMMARY_SCHEMA = """## Current Focus
[One sentence: what the user is currently investigating]

## Entities (preserve verbatim - NEVER paraphrase these)
- People: [names with affiliations, e.g., "John from Acme", "Sarah (PM at BuildCo)"]
- Vendors: [company names exactly as mentioned]
- Invoices: [invoice numbers with amounts, e.g., "INV-1043 ($47,500)"]
- POs: [PO numbers]
- Drawings: [drawing IDs with revisions, e.g., "A-101 Rev 3"]
- Change Orders: [CO numbers with amounts and status]
- Amounts: [dollar amounts with context]
- Dates: [specific dates mentioned]

## Findings (bullet points with specific numbers)
- [Finding 1 with exact figures]
- [Finding 2]

## Open Questions
- [Unanswered question or pending item]"""


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RewriteResult:
    """Result of query rewriting and classification."""
    original_question: str
    standalone_question: str
    was_rewritten: bool  # False if question was already standalone
    question_type: str  # "rag" (needs vector search) or "conversation" (answer from chat history)


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    entities_preserved: int  # Count of entities extracted
    word_count: int


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:
    """
    Manages conversation context for follow-up questions.
    
    Handles:
    - Rewriting follow-up questions to standalone queries
    - Generating and updating conversation summaries
    - Preserving key entities across summary updates
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        self._gemini_client = None
    
    @property
    def gemini_client(self):
        """Lazy-load Gemini client."""
        if self._gemini_client is None:
            from google import genai
            
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client
    
    def rewrite_question(
        self,
        question: str,
        summary: Optional[str],
        recent_messages: List[Dict],
        project_name: str = ""
    ) -> RewriteResult:
        """
        Rewrite a follow-up question into a standalone question.
        
        This is the critical step that connects conversation meaning to retrieval.
        Without it, "what about the second one?" would retrieve random results.
        
        Args:
            question: The user's question (may be a follow-up)
            summary: Existing conversation summary (may be None for new chats)
            recent_messages: Last N messages [{"role": "user/assistant", "content": "..."}]
            project_name: Name of the project for context
        
        Returns:
            RewriteResult with original and standalone questions
        
        Example:
            Input:
                question: "what about the cost?"
                summary: "## Entities\\n- Invoices: INV-1043 ($47,500)\\n..."
                recent_messages: [{"role": "user", "content": "Tell me about INV-1043"}, ...]
            
            Output:
                standalone_question: "What is the cost breakdown for Invoice INV-1043 ($47,500)?"
        """
        # If no conversation context, return as-is (always RAG for first question)
        if not summary and not recent_messages:
            return RewriteResult(
                original_question=question,
                standalone_question=question,
                was_rewritten=False,
                question_type="rag"  # First question always needs document search
            )
        
        # Build context for the rewriter
        context_parts = []
        
        if summary:
            context_parts.append(f"CONVERSATION SUMMARY:\n{summary}")
        
        if recent_messages:
            messages_text = self._format_messages_for_prompt(recent_messages[-REWRITER_MESSAGE_WINDOW:])
            context_parts.append(f"RECENT CONVERSATION:\n{messages_text}")
        
        context = "\n\n".join(context_parts)
        
        rewrite_prompt = f"""You are a query rewriter and classifier for a project document search system.

PROJECT: {project_name or "Construction Project"}

{context}

---

USER'S NEW QUESTION: "{question}"

---

TASK 1 - REWRITE: If the question references the conversation, rewrite it as standalone.

Reference indicators:
- Pronouns: "it", "they", "that", "this", "those", "the same"
- Partial references: "the second one", "the first", "that invoice"
- Implicit context: "what about...", "and the...", "how much for..."

If standalone already, keep it unchanged.

TASK 2 - CLASSIFY: Determine if this needs document search or just conversation memory.

"rag" - Question asks about PROJECT DOCUMENTS, EMAILS, or specific ENTITIES (companies, people, invoices, amounts, dates)
  Examples: "What did MG Transport quote?" → rag (needs document search)
            "Summarize all invoices from ABC Corp" → rag (needs to find ABC Corp docs)
            "What's the pile length?" → rag (project-specific fact)

"conversation" - Question asks about OUR CHAT ITSELF, not project documents
  Examples: "Summarize what we've discussed" → conversation
            "What were the key points?" → conversation  
            "Repeat your last answer" → conversation

KEY: If it mentions a specific entity (company, person, invoice number) that needs LOOKUP, it's "rag".
     If it's asking about the chat dialogue itself, it's "conversation".

RESPOND IN THIS EXACT FORMAT (two lines):
QUESTION: <the standalone question>
TYPE: <rag or conversation>"""

        try:
            response = self.gemini_client.models.generate_content(
                model=REWRITER_MODEL,
                contents=rewrite_prompt,
                config={
                    "temperature": REWRITER_TEMPERATURE,
                    "max_output_tokens": 250
                }
            )
            
            response_text = response.text.strip()
            
            # Parse the two-line response format
            standalone = question  # Default to original
            question_type = "rag"  # Default to RAG (safer)
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.upper().startswith('QUESTION:'):
                    standalone = line[9:].strip().strip('"').strip("'")
                elif line.upper().startswith('TYPE:'):
                    type_value = line[5:].strip().lower()
                    if type_value in ("rag", "conversation"):
                        question_type = type_value
            
            # Check if it was actually rewritten
            was_rewritten = standalone.lower() != question.lower()
            
            logger.info(f"Rewrite: '{question}' -> '{standalone}' (rewritten={was_rewritten}, type={question_type})")
            
            return RewriteResult(
                original_question=question,
                standalone_question=standalone,
                was_rewritten=was_rewritten,
                question_type=question_type
            )
            
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            # Fallback: return original question with RAG type (safer default)
            return RewriteResult(
                original_question=question,
                standalone_question=question,
                was_rewritten=False,
                question_type="rag"
            )
    
    def generate_summary(
        self,
        messages: List[Dict],
        existing_summary: Optional[str] = None,
        project_name: str = ""
    ) -> SummaryResult:
        """
        Generate or update a conversation summary.
        
        This compresses the conversation while preserving key entities.
        The summary should capture:
        - What the user is trying to accomplish
        - All specific entities mentioned (names, IDs, amounts)
        - Key findings so far
        - Open questions
        
        Args:
            messages: All messages in the conversation
            existing_summary: Previous summary to update (if any)
            project_name: Name of the project for context
        
        Returns:
            SummaryResult with the new summary
        """
        messages_text = self._format_messages_for_prompt(messages)
        
        if existing_summary:
            # Update existing summary
            prompt = f"""You are updating a conversation summary for a project research assistant.

PROJECT: {project_name or "Construction Project"}

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES TO INCORPORATE:
{messages_text}

---

Update the summary to include information from the new messages.

CRITICAL RULES:
1. PRESERVE ALL ENTITIES VERBATIM - Never paraphrase names, IDs, invoice numbers, PO numbers, amounts, or dates
2. Keep entities from the existing summary even if not mentioned in new messages (they may be referenced later)
3. Update "Current Focus" to reflect what user is NOW investigating
4. Add new findings to "Findings" section
5. Update "Open Questions" - remove answered ones, add new ones
6. Target {SUMMARY_TARGET_WORDS} words, maximum {SUMMARY_MAX_WORDS} words

OUTPUT FORMAT (use exactly this structure):
{SUMMARY_SCHEMA}

Generate the updated summary:"""
        else:
            # Create new summary
            prompt = f"""You are creating a conversation summary for a project research assistant.

PROJECT: {project_name or "Construction Project"}

CONVERSATION:
{messages_text}

---

Create a summary that captures the conversation context.

CRITICAL RULES:
1. PRESERVE ALL ENTITIES VERBATIM - Extract and list every:
   - Person name with affiliation (e.g., "John from Acme", not just "John")
   - Company/vendor name exactly as written
   - Invoice/PO numbers with amounts
   - Drawing IDs with revisions
   - Change order numbers
   - Specific dollar amounts
   - Specific dates
2. Be specific in Findings - include exact numbers and quotes
3. Target {SUMMARY_TARGET_WORDS} words, maximum {SUMMARY_MAX_WORDS} words

OUTPUT FORMAT (use exactly this structure):
{SUMMARY_SCHEMA}

Generate the summary:"""

        try:
            response = self.gemini_client.models.generate_content(
                model=SUMMARIZER_MODEL,
                contents=prompt,
                config={
                    "temperature": SUMMARIZER_TEMPERATURE,
                    "max_output_tokens": 800
                }
            )
            
            summary = response.text.strip()
            word_count = len(summary.split())
            
            # Count entities (rough estimate based on bullet points in Entities section)
            entities_count = summary.count("- ") if "## Entities" in summary else 0
            
            logger.info(f"Generated summary: {word_count} words, ~{entities_count} entity lines")
            
            return SummaryResult(
                summary=summary,
                entities_preserved=entities_count,
                word_count=word_count
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: return a minimal summary
            return SummaryResult(
                summary=f"## Current Focus\nConversation about {project_name or 'project'}\n\n## Entities\n(Summary generation failed)\n\n## Findings\n- Conversation in progress\n\n## Open Questions\n- To be determined",
                entities_preserved=0,
                word_count=20
            )
    
    def should_resummarize(
        self,
        current_message_count: int,
        message_count_at_last_summary: int,
        summary: Optional[str],
        recent_messages: List[Dict]
    ) -> bool:
        """
        Determine if we should regenerate the summary.
        
        Triggers:
        1. No summary exists and we have 4+ messages (boot summary)
        2. 8+ messages since last summary
        3. Token count of summary + recent messages exceeds threshold
        
        Args:
            current_message_count: Total messages in conversation
            message_count_at_last_summary: Messages when summary was last generated
            summary: Current summary (may be None)
            recent_messages: Recent messages to check token count
        
        Returns:
            True if we should regenerate the summary
        """
        # Boot summary at message 4
        if not summary and current_message_count >= 4:
            logger.info("Triggering boot summary (first summary at message 4)")
            return True
        
        # Regular interval check (every 8 messages)
        messages_since_summary = current_message_count - message_count_at_last_summary
        if messages_since_summary >= SUMMARY_MESSAGE_INTERVAL:
            logger.info(f"Triggering summary update ({messages_since_summary} messages since last)")
            return True
        
        # Token threshold check (rough estimate: 4 chars per token)
        if summary and recent_messages:
            summary_tokens = len(summary) // 4
            messages_tokens = sum(len(m.get("content", "")) // 4 for m in recent_messages)
            total_tokens = summary_tokens + messages_tokens
            
            if total_tokens > SUMMARY_TOKEN_THRESHOLD:
                logger.info(f"Triggering summary update (token count ~{total_tokens} exceeds threshold)")
                return True
        
        return False
    
    def _format_messages_for_prompt(self, messages: List[Dict]) -> str:
        """Format messages for inclusion in a prompt."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            
            prefix = "User" if role == "user" else "Assistant"
            formatted.append(f"{prefix}: {content}")
        
        return "\n\n".join(formatted)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def rewrite_followup(
    question: str,
    summary: Optional[str],
    recent_messages: List[Dict],
    project_name: str = ""
) -> str:
    """
    Convenience function to rewrite a follow-up question.
    
    Returns just the standalone question string.
    """
    manager = ConversationManager()
    result = manager.rewrite_question(question, summary, recent_messages, project_name)
    return result.standalone_question


def generate_chat_summary(
    messages: List[Dict],
    existing_summary: Optional[str] = None,
    project_name: str = ""
) -> str:
    """
    Convenience function to generate a summary.
    
    Returns just the summary string.
    """
    manager = ConversationManager()
    result = manager.generate_summary(messages, existing_summary, project_name)
    return result.summary


# ============================================================
# CLI TESTING
# ============================================================

if __name__ == "__main__":
    # Test the conversation manager
    manager = ConversationManager()
    
    # Test rewriting
    print("\n" + "="*60)
    print("TESTING QUERY REWRITER")
    print("="*60)
    
    test_summary = """## Current Focus
Investigating foundation costs for the supermarket project

## Entities
- Vendors: ABC Contractors, XYZ Foundation Services
- Invoices: INV-1043 ($47,500), INV-1089 ($12,300)
- People: John from Acme (sales rep)
- Drawings: S-201 (structural), A-101 Rev 3

## Findings
- ABC Contractors quoted $47,500 for pile work
- Pile length is 45 feet per Drawing S-201

## Open Questions
- Waiting on counter-quote from XYZ"""

    test_messages = [
        {"role": "user", "content": "What did ABC Contractors quote for the foundation?"},
        {"role": "assistant", "content": "ABC Contractors quoted $47,500 for the pile work according to Invoice INV-1043."},
        {"role": "user", "content": "And what about the second invoice?"},
    ]
    
    # Test with follow-up question
    result = manager.rewrite_question(
        question="what about the cost breakdown?",
        summary=test_summary,
        recent_messages=test_messages,
        project_name="88 Supermarket"
    )
    
    print(f"\nOriginal: {result.original_question}")
    print(f"Standalone: {result.standalone_question}")
    print(f"Was rewritten: {result.was_rewritten}")
    
    # Test with standalone question
    result2 = manager.rewrite_question(
        question="What is the total project budget?",
        summary=test_summary,
        recent_messages=test_messages,
        project_name="88 Supermarket"
    )
    
    print(f"\nOriginal: {result2.original_question}")
    print(f"Standalone: {result2.standalone_question}")
    print(f"Was rewritten: {result2.was_rewritten}")
    
    # Test summarization
    print("\n" + "="*60)
    print("TESTING SUMMARIZER")
    print("="*60)
    
    test_conversation = [
        {"role": "user", "content": "What did ABC Contractors quote for the foundation work?"},
        {"role": "assistant", "content": "According to Invoice INV-1043, ABC Contractors quoted $47,500 for the pile foundation work. The quote was sent by John from Acme on January 15th."},
        {"role": "user", "content": "What's the pile length they're proposing?"},
        {"role": "assistant", "content": "Based on Drawing S-201 (structural drawings), the pile length is specified as 45 feet. This matches the geotechnical report requirements."},
        {"role": "user", "content": "Do we have any other quotes to compare?"},
        {"role": "assistant", "content": "Yes, there's also Invoice INV-1089 from XYZ Foundation Services for $12,300, but this appears to be for a different scope - it covers soil testing rather than pile work."},
    ]
    
    summary_result = manager.generate_summary(
        messages=test_conversation,
        project_name="88 Supermarket"
    )
    
    print(f"\nGenerated Summary ({summary_result.word_count} words):")
    print("-" * 40)
    print(summary_result.summary)
    print("-" * 40)
    print(f"Entities preserved: ~{summary_result.entities_preserved} lines")

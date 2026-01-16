"""
Tool executor for Donna SUPERHUMAN.

Single dispatcher that:
1. Executes all tools by name
2. Updates shared GraphState
3. Calls original implementations (outlook_tools, web_tools, memory_tools)

This module can be used by any LangGraph node.
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from graph_state import GraphState
from tools.outlook_tools import get_outlook_tools
from tools.gmail_tools import get_gmail_tools
from tools.web_tools import get_web_search_tools
from tools.memory_tools import get_memory_tools
from tools.pdf_tools import get_pdf_skimmer
from state import (
    create_email_metadata_from_outlook,
    create_draft_metadata_from_outlook
)
from config import SHOW_STATE_CHANGES, SHOW_CACHE_OPERATIONS, SHOW_ID_RESOLUTION


class ToolExecutor:
    """
    Executes tools and updates GraphState.
    
    Each tool method:
    1. Parses arguments
    2. Calls original implementation (outlook_tools, web_tools, memory_tools)
    3. Updates state (cache, email editor, prefs, last_fact)
    4. Returns result
    
    The state reference is shared, so all updates are immediately visible.
    """
    
    def __init__(self, state: GraphState):
        """
        Initialize tool executor with state reference.
        Tools are lazy-loaded to avoid unnecessary auth flows.
        """
        self.state = state
        
        # Internal holders for lazy loading
        self._outlook = None
        self._gmail = None
        
        self.web = get_web_search_tools()
        self.memory = get_memory_tools()
        self.pdf_skimmer = get_pdf_skimmer()

    @property
    def outlook(self):
        """Lazy load Outlook tools."""
        if self._outlook is None:
            # Only init if we really need it
            self._outlook = get_outlook_tools()
        return self._outlook

    @property
    def gmail(self):
        """Lazy load Gmail tools."""
        if self._gmail is None:
            # Only init if we really need it
            self._gmail = get_gmail_tools()
        return self._gmail
    
    def execute(self, tool_name: str, arguments: str) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: JSON string of tool arguments
        
        Returns:
            Tool result (dict with success/error)
        """
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return {"error": "Invalid tool arguments JSON"}
        
        # Dispatch to appropriate tool method
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return method(args)
        except Exception as e:
            import traceback
            return {
                "error": f"Tool execution failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    # ============================================================
    # EMAIL OPERATIONS
    # ============================================================
    
    def _tool_list_emails(self, args: Dict) -> Dict:
        """List recent emails and update cache."""
        limit = args.get("limit", 10)
        days = args.get("days", 7)
        unread_only = args.get("unread_only", False)
        
        provider = self.state.get("provider", "outlook")
        if provider == "gmail":
            # Gmail doesn't support unread_only/days in simple list easily without query
            # We'll construct a query if needed, or just list
            query = ""
            if unread_only: query += "is:unread "
            # Date filtering in Gmail query is complex, skipping for MVP list
            
            # Use search if filters needed, else list
            if query:
                result = self.gmail.search_messages(query.strip(), limit)
            else:
                result = self.gmail.search_messages("category:primary", limit) # Default to primary inbox
        else:
            result = self.outlook.list_emails(limit, days=days, unread_only=unread_only)
        
        if "error" not in result:
            # Update cache with email metadata
            emails = result.get("emails", [])
            self.state["cache"].email_search_results = [
                create_email_metadata_from_outlook(email, idx + 1)
                for idx, email in enumerate(emails)
            ]
            
            if SHOW_CACHE_OPERATIONS:
                print(f"\n📋 [CACHE] Updated email_search_results: {len(emails)} emails")
        
        return result
    
    def _tool_get_email(self, args: Dict) -> Dict:
        """Load an email into EmailEditor."""
        email_id = args["email_id"]
        
        # Resolve position/name to actual ID
        actual_id = self._resolve_email_id(email_id)
        if isinstance(actual_id, dict) and "error" in actual_id:
            return actual_id
        
        # Fetch from Provider
        provider = self.state.get("provider", "outlook")
        if provider == "gmail":
            result = self.gmail.get_email(actual_id)
        else:
            result = self.outlook.get_email(actual_id)
        
        if "error" not in result:
            # Load into EmailEditor
            if SHOW_STATE_CHANGES:
                if not self.state["email_editor"].is_empty():
                    print(f"\n🔄 [STATE] Replacing EmailEditor content")
                print(f"\n📧 [STATE] Loading email into EmailEditor:")
                print(f"   From: {result.get('from_name', 'Unknown')}")
                print(f"   Subject: {result.get('subject', '(No subject)')[:60]}")
            
            self.state["email_editor"].type = "received_email"
            self.state["email_editor"].id = actual_id
            self.state["email_editor"].from_name = result.get("from_name", "")
            self.state["email_editor"].from_email = result.get("from_email", "") # Gmail uses from_email
            self.state["email_editor"].subject = result.get("subject", "")
            self.state["email_editor"].body = result.get("body", "") # Gmail returns body, Outlook body_text
            self.state["email_editor"].body_html = result.get("body_html", "")
            self.state["email_editor"].thread_id = result.get("conversationId") or result.get("thread_id")
            self.state["email_editor"].timestamp = result.get("receivedDateTime") or result.get("timestamp")
            self.state["email_editor"].has_attachments = result.get("has_attachments", False)
            
            # Auto-fetch attachments if present
            if result.get("has_attachments", False):
                if provider == "gmail":
                    attachments_result = self.gmail.list_attachments(actual_id)
                else:
                    attachments_result = self.outlook.list_attachments(actual_id)
                    
                if "error" not in attachments_result:
                    self.state["email_editor"].attachments = attachments_result.get("attachments", [])
                    if SHOW_STATE_CHANGES:
                        print(f"   📎 Attachments: {len(self.state['email_editor'].attachments)}")
                else:
                    self.state["email_editor"].attachments = []
            else:
                self.state["email_editor"].attachments = []
        
        return result
    
    def _tool_search_emails(self, args: Dict) -> Dict:
        """Search emails and update cache."""
        from_person = args.get("from_person", "")
        keywords = args.get("keywords", "")
        
        if from_person:
            if "@" not in from_person:
                return {
                    "error": f"Invalid sender reference '{from_person}'. Please resolve their email address before searching."
                }
            
            provider = self.state.get("provider", "outlook")
            if provider == "gmail":
                q = f"from:{from_person}"
                if keywords: q += f" {keywords}"
                result = self.gmail.search_messages(q, 20)
            else:
                result = self.outlook.search_latest_from(from_person, keywords)
        else:
            return {"error": "Must provide from_person for search"}
        
        if "error" not in result:
            # Update cache
            emails = result.get("emails", [])
            self.state["cache"].email_search_results = [
                create_email_metadata_from_outlook(email, idx + 1)
                for idx, email in enumerate(emails)
            ]
        
        return result
    
    def _tool_search_messages(self, args: Dict) -> Dict:
        """
        Search messages using full-text search.
        Updates cache with results.
        """
        query = args.get("query", "")
        top = args.get("top", 20)
        page_cursor = args.get("page_cursor")
        lightweight = bool(args.get("lightweight", False))
        
        if not query:
            return {"error": "Must provide query for search"}
        
        provider = self.state.get("provider", "outlook")
        if provider == "gmail":
            result = self.gmail.search_messages(query, top, page_cursor=page_cursor, lightweight=lightweight)
        else:
            result = self.outlook.search_messages(query, top, page_cursor=page_cursor)
        
        if result.get("success") and "error" not in result:
            # Update cache with search results
            emails = result.get("emails", [])
            self.state["cache"].email_search_results = [
                create_email_metadata_from_outlook(email, idx + 1)
                for idx, email in enumerate(emails)
            ]
            
            if SHOW_CACHE_OPERATIONS:
                print(f"\n🔍 [CACHE] Updated email_search_results: {len(emails)} emails from query '{query}'")
        
        return result

    def _tool_get_thread_context(self, args: Dict) -> Dict:
        """
        Fetch ALL messages in a conversation thread (Total Recall).
        """
        thread_id = args.get("thread_id") or args.get("conversation_id")
        if not thread_id:
            return {"error": "Must provide thread_id"}

        provider = self.state.get("provider", "outlook")
        # print(f"\n🔍 [EXECUTOR] Fetching thread {thread_id[:10]}... from {provider.upper()}")
        
        if provider == "gmail":
            return self.gmail.get_conversation_threads(thread_id)
        else:
            return self.outlook.get_conversation_threads(thread_id)
    
    def _tool_list_attachments(self, args: Dict) -> Dict:
        """
        List attachments for a specific message.
        Note: get_email already auto-fetches attachments, but this tool
        can be used if you only want attachment metadata without loading the full email.
        """
        message_id = args.get("message_id", "")
        
        if not message_id:
            return {"error": "Must provide message_id"}
        
        # Check if this is the currently loaded email (already has attachments)
        email = self.state["email_editor"]
        if not email.is_empty() and email.id == message_id and email.attachments:
            if SHOW_CACHE_OPERATIONS:
                print(f"\n📎 [CACHE] Attachments already loaded for current email")
            return {
                "attachments": email.attachments,
                "count": len(email.attachments),
                "cached": True
            }
        
        # Fetch from Provider
        provider = self.state.get("provider", "outlook")
        if provider == "gmail":
            result = self.gmail.list_attachments(message_id)
        else:
            result = self.outlook.list_attachments(message_id)
        
        return result
    
    def _tool_download_attachment(self, args: Dict) -> Dict:
        """
        Download attachment bytes.
        Returns content_bytes, name, size, content_type.
        """
        message_id = args.get("message_id", "")
        attachment_id = args.get("attachment_id", "")
        
        if not message_id or not attachment_id:
            return {"error": "Must provide message_id and attachment_id"}
        
        # Check if attachment_id is already an actual ID (not a position number)
        # If it starts with AAMk or AQMk, it's already the actual ID - use directly
        if attachment_id.startswith("AAMk") or attachment_id.startswith("AQMk"):
            # Already an actual ID - use directly
            actual_attachment_id = attachment_id
        elif attachment_id.isdigit():
            # It's a position number - need to resolve via EmailEditor
            actual_attachment_id = self._resolve_attachment_id(message_id, attachment_id)
            if isinstance(actual_attachment_id, dict) and "error" in actual_attachment_id:
                return actual_attachment_id
        else:
            # Try to resolve (could be a name or other reference)
            actual_attachment_id = self._resolve_attachment_id(message_id, attachment_id)
            if isinstance(actual_attachment_id, dict) and "error" in actual_attachment_id:
                # If resolution fails and email not loaded, just try using it as-is
                # (might be a valid ID that doesn't need resolution)
                actual_attachment_id = attachment_id
        
        # Download from Provider
        provider = self.state.get("provider", "outlook")
        if provider == "gmail":
            result = self.gmail.download_attachment(message_id, actual_attachment_id)
        else:
            result = self.outlook.download_attachment(message_id, actual_attachment_id)
        
        return result
    
    # ============================================================
    # DRAFT OPERATIONS
    # ============================================================
    
    def _tool_create_draft(self, args: Dict) -> Dict:
        """Create a new draft and load into EmailEditor."""
        to = args["to"]
        subject = args["subject"]
        body = args["body"]
        cc = args.get("cc", "")
        
        def _first_invalid_address(raw_addresses: str) -> Optional[str]:
            if not raw_addresses:
                return None
            addresses = [addr.strip() for addr in raw_addresses.split(",") if addr.strip()]
            for addr in addresses:
                if "@" not in addr:
                    return addr
            return None
        
        invalid_to = _first_invalid_address(to)
        if invalid_to:
            return {
                "error": f"Invalid recipient '{invalid_to}'. Call resolve_person or provide a full email address."
            }
        
        invalid_cc = _first_invalid_address(cc)
        if invalid_cc:
            return {
                "error": f"Invalid CC recipient '{invalid_cc}'. Call resolve_person or provide a full email address."
            }
        
        # Create in Outlook
        prefs = self.state["prefs"]
        result = self.outlook.create_draft(
            to, subject, prefs.tone, prefs.signature, body, cc
        )
        
        if "error" not in result:
            # Load into EmailEditor
            if SHOW_STATE_CHANGES:
                if not self.state["email_editor"].is_empty():
                    print(f"\n🔄 [STATE] Replacing EmailEditor content")
                print(f"\n✉️  [STATE] Loading new draft into EmailEditor:")
                print(f"   To: {to}")
                print(f"   Subject: {subject[:60]}")
            
            self.state["email_editor"].type = "draft"
            self.state["email_editor"].draft_id = result["draft_id"]
            self.state["email_editor"].to = to
            self.state["email_editor"].subject = subject
            self.state["email_editor"].body = body
            self.state["email_editor"].cc = cc if cc else None
            self.state["email_editor"].created_at = datetime.now()
        
        return result
    
    def _tool_create_reply(self, args: Dict) -> Dict:
        """Create reply to current email in EmailEditor."""
        email = self.state["email_editor"]
        
        if email.is_empty() or not email.is_received_email():
            return {"error": "No email in EmailEditor to reply to. Read an email first."}
        
        body = args["body"]
        
        # Create reply in Outlook
        prefs = self.state["prefs"]
        result = self.outlook.create_reply_draft(
            email.id,
            body,
            prefs.signature,
            reply_all=False
        )
        
        if "error" not in result:
            # Load reply into EmailEditor
            if SHOW_STATE_CHANGES:
                print(f"\n💬 [STATE] Converting EmailEditor: Email → Reply Draft")
                print(f"   Replying to: {email.from_name}")
            
            original_from = email.from_email
            self.state["email_editor"].type = "draft"
            self.state["email_editor"].draft_id = result["draft_id"]
            self.state["email_editor"].to = original_from
            self.state["email_editor"].subject = "RE: " + (email.subject or "")
            self.state["email_editor"].body = body
            self.state["email_editor"].in_reply_to = email.id
            self.state["email_editor"].created_at = datetime.now()
        
        return result
    
    def _tool_forward_email(self, args: Dict) -> Dict:
        """Forward the current email in EmailEditor."""
        email = self.state["email_editor"]
        
        if email.is_empty() or not email.is_received_email():
            return {"error": "No email in EmailEditor to forward. Read an email first."}
        
        to = args["to"]
        body = args.get("body", "")
        cc = args.get("cc", "")
        bcc = args.get("bcc", "")
        
        # Safeguard: If body is too long, it likely contains copied original content
        # Truncate to prevent duplication (Outlook includes the original automatically)
        if body and len(body) > 500:
            print(f"\n⚠️  [FORWARD] Body too long ({len(body)} chars), likely contains original email. Truncating to brief intro only.")
            # Try to keep just the first paragraph/sentence
            if "<p>" in body:
                first_p_end = body.find("</p>") + 4
                if first_p_end > 4:
                    body = body[:first_p_end]
            else:
                body = body[:200] + "..."
        
        prefs = self.state["prefs"]
        result = self.outlook.create_forward_draft(
            email.id,
            to,
            body_text=body,
            signature=prefs.signature,
            cc=cc,
            bcc=bcc
        )
        
        if "error" not in result:
            if SHOW_STATE_CHANGES:
                print(f"\n📨 [STATE] Converting EmailEditor: Email → Forward Draft")
                print(f"   Forwarding to: {to}")
            
            self.state["email_editor"].type = "draft"
            self.state["email_editor"].draft_id = result.get("draft_id", "")
            self.state["email_editor"].to = result.get("to") or to
            self.state["email_editor"].subject = result.get("subject", "")
            # IMPORTANT: For a forward, only show the author's brief intro in the draft body.
            # The original message is preserved by Outlook in the forwarded content automatically.
            # Never paste or reconstruct the original body here to avoid duplication.
            self.state["email_editor"].body = body or ""
            self.state["email_editor"].cc = cc if cc else None
            self.state["email_editor"].created_at = datetime.now()
        
        return result
    
    def _tool_edit_draft(self, args: Dict) -> Dict:
        """Edit current draft in EmailEditor."""
        email = self.state["email_editor"]
        
        if not email.is_draft():
            return {"error": "No draft in EmailEditor to edit"}
        
        body = args.get("body")
        subject = args.get("subject")
        cc = args.get("cc")
        
        # Update in Outlook
        prefs = self.state["prefs"]
        result = self.outlook.edit_draft(
            email.draft_id,
            body_text=body,
            subject=subject,
            cc=cc,
            signature=prefs.signature
        )
        
        if "error" not in result:
            # Update EmailEditor
            if body:
                self.state["email_editor"].body = body
            if subject:
                self.state["email_editor"].subject = subject
            if cc:
                self.state["email_editor"].cc = cc
        
        return result
    
    def _tool_send_draft(self, args: Dict) -> Dict:
        """Send current draft in EmailEditor."""
        email = self.state["email_editor"]
        
        if not email.is_draft():
            return {"error": "No draft in EmailEditor to send"}
        
        if SHOW_STATE_CHANGES:
            print(f"\n📤 [STATE] Sending draft from EmailEditor:")
            print(f"   To: {email.to}")
            print(f"   Subject: {email.subject[:60]}")
        
        draft_id_to_send = email.draft_id
        
        # Send via Outlook
        result = self.outlook.send_draft(draft_id_to_send)
        
        if "error" not in result:
            if SHOW_STATE_CHANGES:
                print(f"\n✅ [STATE] Email sent! Clearing EmailEditor")
            
            # Clear EmailEditor after sending
            self.state["email_editor"].clear()
            
            # Remove from draft cache if present
            self.state["cache"].remove_draft(draft_id_to_send)
        
        return result
    
    def _tool_list_drafts(self, args: Dict) -> Dict:
        """List drafts and update cache."""
        limit = args.get("limit", 20)
        
        result = self.outlook.list_drafts(limit)
        
        if "error" not in result:
            # Update cache
            drafts = result.get("drafts", [])
            self.state["cache"].recent_drafts = [
                create_draft_metadata_from_outlook(draft, idx + 1)
                for idx, draft in enumerate(drafts)
            ]
            
            if SHOW_CACHE_OPERATIONS:
                print(f"\n📝 [CACHE] Updated recent_drafts: {len(drafts)} drafts")
        
        return result
    
    def _tool_get_draft(self, args: Dict) -> Dict:
        """Load a draft into EmailEditor."""
        draft_id = args["draft_id"]
        
        # Resolve position/name to actual ID
        actual_id = self._resolve_draft_id(draft_id)
        if isinstance(actual_id, dict) and "error" in actual_id:
            return actual_id
        
        # Fetch from Outlook
        result = self.outlook.get_draft(actual_id)
        
        if "error" not in result:
            draft_data = result.get("draft", {})
            
            # Load into EmailEditor
            if SHOW_STATE_CHANGES:
                if not self.state["email_editor"].is_empty():
                    print(f"\n🔄 [STATE] Replacing EmailEditor content")
                print(f"\n📝 [STATE] Loading draft into EmailEditor:")
                print(f"   To: {draft_data.get('to', 'Unknown')}")
                print(f"   Subject: {draft_data.get('subject', '(No subject)')[:60]}")
            
            self.state["email_editor"].type = "draft"
            self.state["email_editor"].draft_id = actual_id
            self.state["email_editor"].to = draft_data.get("to", "")
            self.state["email_editor"].subject = draft_data.get("subject", "")
            self.state["email_editor"].body = draft_data.get("body_text", "")
            self.state["email_editor"].body_html = draft_data.get("body_html", "")
            self.state["email_editor"].cc = draft_data.get("cc")
            self.state["email_editor"].created_at = draft_data.get("createdDateTime")
        
        return result
    
    # ============================================================
    # UTILITY TOOLS
    # ============================================================
    
    def _tool_resolve_person(self, args: Dict) -> Dict:
        """Resolve a person's name to an email address."""
        name = args["name"]
        result = self.outlook.resolve_person(name)
        return result
    
    def _tool_smart_email_search(self, args: Dict) -> Dict:
        """
        Smart email search - Perplexity for Outlook inbox + attachments.
        
        Runs a LangGraph subgraph that:
        1. Generates multiple search queries
        2. Searches emails in parallel
        3. Groups by threads
        4. Skims PDF attachments
        5. Deep-reads most relevant attachments
        6. Synthesizes final answer
        
        Returns structured results with threads_used, all_threads_considered, etc.
        """
        from tools.email_search_tools import smart_email_search
        
        query = args.get("query", "")
        
        if not query:
            return {"error": "Must provide query for smart email search"}
        
        # Run the search
        result = smart_email_search(query)
        
        # Convert to dict (already is, but ensure serialization)
        return {
            "final_answer": result["final_answer"],
            "threads_used": result["threads_used"],
            "all_threads_considered": result["all_threads_considered"],
            "debug_info": result["debug_info"],
            "success": True
        }
    
    def _tool_web_search(self, args: Dict) -> Dict:
        """Search the web."""
        query = args["query"]
        
        result = self.web.search(query)
        
        if result.get("success"):
            # Update last_fact in state
            self.state["last_fact"].value = result.get("answer", "")
            self.state["last_fact"].source = "web_search"
            self.state["last_fact"].fetched_at = datetime.now()
        
        return result
    
    def _tool_save_preference(self, args: Dict) -> Dict:
        """Save user preference."""
        key = args["key"]
        value = args["value"]
        
        # Save to memory
        result = self.memory.write("preferences", key, value)
        
        # Update state
        prefs = self.state["prefs"]
        if hasattr(prefs, key):
            setattr(prefs, key, value)
        
        return result

    def _tool_switch_provider(self, args: Dict) -> Dict:
        """Switch the active email provider (outlook/gmail)."""
        provider = args.get("provider", "").lower()
        
        if provider not in ["outlook", "gmail"]:
            return {"error": f"Invalid provider '{provider}'. Use 'outlook' or 'gmail'."}
            
        self.state["provider"] = provider
        
        # Save to memory so it persists next time
        self.memory.write("system", "last_provider", provider)
        
        if SHOW_STATE_CHANGES:
            print(f"\n🔄 [STATE] Switched Email Provider to: {provider.upper()}")
        
        return {"success": True, "provider": provider, "message": f"Switched to {provider.title()}"}
    
    def _tool_skim_pdf(self, args: Dict) -> Dict:
        """Skim a PDF attachment and return summary."""
        message_id = args["message_id"]
        attachment_id = args["attachment_id"]
        
        # Resolve attachment ID (handle position numbers like "1", "2", etc.)
        actual_attachment_id = self._resolve_attachment_id(message_id, attachment_id)
        if isinstance(actual_attachment_id, dict) and "error" in actual_attachment_id:
            return actual_attachment_id
        
        # Check cache first
        cached_summary = self.state["cache"].get_pdf_summary(message_id, actual_attachment_id)
        if cached_summary:
            if SHOW_CACHE_OPERATIONS:
                print(f"\n📄 [CACHE] PDF summary found in cache (skipping re-skim)")
            return {
                "summary": cached_summary,
                "cached": True,
                "message": "Summary retrieved from cache"
            }
        
        # Download attachment
        print(f"\n📥 Downloading PDF attachment...")
        download_result = self.outlook.download_attachment(message_id, actual_attachment_id)
        
        if "error" in download_result:
            return download_result
        
        pdf_bytes = download_result.get("content_bytes")
        filename = download_result.get("name", "document.pdf")
        
        # Only skim PDFs
        content_type = download_result.get("content_type", "")
        if "pdf" not in content_type.lower():
            return {
                "error": f"File '{filename}' is not a PDF (content type: {content_type})",
                "summary": f"Non-PDF attachment: {filename}"
            }
        
        print(f"📄 Skimming PDF: {filename} ({download_result.get('size', 0) / 1024:.1f} KB)")
        
        # Skim with Gemini Flash
        skim_result = self.pdf_skimmer.skim_pdf(pdf_bytes, filename)
        
        if skim_result.get("success"):
            summary = skim_result.get("summary", "")
            
            # Cache the summary
            self.state["cache"].cache_pdf_summary(
                message_id=message_id,
                attachment_id=actual_attachment_id,
                attachment_name=filename,
                summary=summary
            )
            
            return {
                "summary": summary,
                "cached": False,
                "message": f"Skimmed {filename} successfully"
            }
        else:
            return {
                "error": skim_result.get("error", "Unknown error"),
                "message": f"Failed to skim {filename}"
            }
    
    def _tool_read_pdf(self, args: Dict) -> Dict:
        """Deep read a PDF attachment and return comprehensive analysis."""
        message_id = args["message_id"]
        attachment_id = args["attachment_id"]
        question = args.get("question")  # Optional specific question
        
        # Resolve attachment ID (handle position numbers like "1", "2", etc.)
        actual_attachment_id = self._resolve_attachment_id(message_id, attachment_id)
        if isinstance(actual_attachment_id, dict) and "error" in actual_attachment_id:
            return actual_attachment_id
        
        # Download attachment
        print(f"\n📥 Downloading PDF attachment for deep analysis...")
        download_result = self.outlook.download_attachment(message_id, actual_attachment_id)
        
        if "error" in download_result:
            return download_result
        
        pdf_bytes = download_result.get("content_bytes")
        filename = download_result.get("name", "document.pdf")
        
        # Only read PDFs
        content_type = download_result.get("content_type", "")
        if "pdf" not in content_type.lower():
            return {
                "error": f"File '{filename}' is not a PDF (content type: {content_type})",
                "analysis": f"Non-PDF attachment: {filename}"
            }
        
        print(f"📖 Deep reading PDF: {filename} ({download_result.get('size', 0) / 1024:.1f} KB)")
        if question:
            print(f"   Question: {question}")
        print(f"   Using Gemini Pro for comprehensive analysis...")
        
        # Deep read with Gemini Pro
        read_result = self.pdf_skimmer.read_pdf(pdf_bytes, filename, question)
        
        if read_result.get("success"):
            analysis = read_result.get("analysis", "")
            
            return {
                "analysis": analysis,
                "key_details": read_result.get("key_details", {}),
                "model_used": read_result.get("model_used", ""),
                "message": f"Deep read {filename} successfully"
            }
        else:
            return {
                "error": read_result.get("error", "Unknown error"),
                "message": f"Failed to read {filename}"
            }
    
    def _tool_skim_pdf_bytes(self, args: Dict) -> Dict:
        """
        Skim a PDF from raw bytes (not Outlook attachment).
        For use in subgraphs where PDF bytes are already downloaded.
        """
        import base64
        
        pdf_bytes_b64 = args.get("pdf_bytes_base64", "")
        filename = args.get("filename", "document.pdf")
        
        if not pdf_bytes_b64:
            return {"error": "Must provide pdf_bytes_base64"}
        
        try:
            # Decode base64 to bytes
            pdf_bytes = base64.b64decode(pdf_bytes_b64)
        except Exception as e:
            return {"error": f"Failed to decode base64 PDF bytes: {str(e)}"}
        
        print(f"📄 Skimming PDF: {filename} ({len(pdf_bytes) / 1024:.1f} KB)")
        
        # Skim with Gemini Flash
        skim_result = self.pdf_skimmer.skim_pdf(pdf_bytes, filename)
        
        if skim_result.get("success"):
            return {
                "summary": skim_result.get("summary", ""),
                "message": f"Skimmed {filename} successfully"
            }
        else:
            return {
                "error": skim_result.get("error", "Unknown error"),
                "message": f"Failed to skim {filename}"
            }
    
    def _tool_read_pdf_bytes(self, args: Dict) -> Dict:
        """
        Deep read a PDF from raw bytes (not Outlook attachment).
        For use in subgraphs where PDF bytes are already downloaded.
        """
        import base64
        
        pdf_bytes_b64 = args.get("pdf_bytes_base64", "")
        filename = args.get("filename", "document.pdf")
        question = args.get("question")
        
        if not pdf_bytes_b64:
            return {"error": "Must provide pdf_bytes_base64"}
        
        try:
            # Decode base64 to bytes
            pdf_bytes = base64.b64decode(pdf_bytes_b64)
        except Exception as e:
            return {"error": f"Failed to decode base64 PDF bytes: {str(e)}"}
        
        print(f"📖 Deep reading PDF: {filename} ({len(pdf_bytes) / 1024:.1f} KB)")
        if question:
            print(f"   Question: {question}")
        
        # Deep read with Gemini Pro
        read_result = self.pdf_skimmer.read_pdf(pdf_bytes, filename, question)
        
        if read_result.get("success"):
            return {
                "analysis": read_result.get("analysis", ""),
                "key_details": read_result.get("key_details", {}),
                "model_used": read_result.get("model_used", ""),
                "message": f"Deep read {filename} successfully"
            }
        else:
            return {
                "error": read_result.get("error", "Unknown error"),
                "message": f"Failed to read {filename}"
            }
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _extract_recent_email_references(self) -> Dict[str, Any]:
        """
        Extract recently mentioned emails from conversation context.
        
        Looks for patterns like:
        - "the other email" → Email #2 (if Email #1 was just discussed)
        - "that email" → Last email mentioned
        - "the second one" → Email #2
        
        Returns:
            Dict with context info (recently_shown_emails, last_mentioned_position, etc.)
        """
        context = {
            "recently_shown_emails": [],
            "last_mentioned_position": None,
            "last_mentioned_id": None
        }
        
        # Look at last few messages for context
        messages = self.state.get("messages", [])
        
        # Check if list_emails was just called (emails were recently listed)
        # If cache has emails and we just listed them, assume all cached emails were shown
        cache = self.state["cache"]
        if cache.email_search_results:
            # Check last few tool messages to see if list_emails was called
            for msg in messages[-5:]:
                from langchain_core.messages import ToolMessage
                if isinstance(msg, ToolMessage):
                    try:
                        import json
                        result = json.loads(msg.content)
                        if isinstance(result, dict) and "emails" in result:
                            # list_emails was called - all cached emails were shown
                            context["recently_shown_emails"] = list(range(1, len(cache.email_search_results) + 1))
                            break
                    except:
                        pass
        
        # Check last 10 messages for email references
        for msg in messages[-10:]:
            from langchain_core.messages import AIMessage, HumanMessage
            
            if isinstance(msg, AIMessage):
                # Agent just showed emails - try to extract positions
                content = msg.content or ""
                # Look for patterns like:
                # "1. **Subject:**", "2. **Subject:**"
                # "1. Subject:", "2. Subject:"
                # "Email 1:", "Email 2:"
                import re
                # Try multiple patterns to detect listed emails
                patterns = [
                    r'\b(\d+)\.\s+\*\*Subject:\*\*',  # "1. **Subject:**"
                    r'\b(\d+)\.\s+Subject:',          # "1. Subject:"
                    r'Email\s+(\d+):',                # "Email 1:"
                    r'^\s*(\d+)\.\s+\*\*',            # "1. **" at start of line
                    r'\n\s*(\d+)\.\s+\*\*',           # "1. **" after newline
                ]
                
                all_positions = []
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    all_positions.extend([int(p) for p in matches])
                
                if all_positions:
                    # Remove duplicates, keep order, sort
                    unique_positions = list(dict.fromkeys(all_positions))
                    unique_positions.sort()
                    context["recently_shown_emails"] = unique_positions
            
            elif isinstance(msg, HumanMessage):
                content = msg.content.lower()
                # Check for position hints
                if "email 1" in content or "first email" in content:
                    context["last_mentioned_position"] = 1
                elif "email 2" in content or "second email" in content:
                    context["last_mentioned_position"] = 2
                elif "email 3" in content or "third email" in content:
                    context["last_mentioned_position"] = 3
        
        return context
    
    def _resolve_email_id(self, email_id: str) -> Any:
        """
        Resolve position number or name to actual email ID.
        
        Now with smart context-aware resolution:
        - Normalizes keywords (handles spaces)
        - Checks all matches (not just first)
        - Uses conversation context
        - Handles ambiguous cases
        - Accepts direct Outlook message IDs without cache lookup
        """
        # 1. FIRST: Check if it's already a valid Outlook message ID (skip cache)
        # Outlook IDs typically start with AAMk, AQMk, or are long base64-like strings
        if email_id.startswith("AAMk") or email_id.startswith("AQMk") or len(email_id) > 50:
            if SHOW_ID_RESOLUTION:
                print(f"\n🔍 [ID_RESOLVE] Direct Outlook ID (no cache needed): {email_id[:20]}...")
            return email_id
        
        # 2. For position numbers or keyword searches, we need cache
        cache = self.state["cache"]
        
        if not cache.email_search_results:
            return {"error": "No emails cached. Use list_emails first."}
        
        # 3. Check if it's a position number
        if email_id.isdigit():
            metadata = cache.get_email_by_position(int(email_id))
            if metadata:
                if SHOW_ID_RESOLUTION:
                    print(f"\n🔍 [ID_RESOLVE] Position {email_id} → {metadata.from_name} ({metadata.id[:20]}...)")
                return metadata.id
            else:
                return {"error": f"Email position {email_id} not found"}
        
        # 4. Check for position hints in keyword
        email_id_lower = email_id.lower()
        position_hints = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5
        }
        
        for hint, position in position_hints.items():
            if hint in email_id_lower:
                metadata = cache.get_email_by_position(position)
                if metadata:
                    if SHOW_ID_RESOLUTION:
                        print(f"\n🔍 [ID_RESOLVE] Position hint '{hint}' → {metadata.from_name} ({metadata.id[:20]}...)")
                    return metadata.id
        
        # 5. Get conversation context
        context = self._extract_recent_email_references()
        
        # 6. Check for "other" or "last" email (needs context)
        if "other" in email_id_lower or "last" in email_id_lower:
            if context["recently_shown_emails"]:
                if "other" in email_id_lower:
                    # "the other email" after showing [1, 2, 3] → #2
                    if len(context["recently_shown_emails"]) >= 2:
                        position = context["recently_shown_emails"][1]  # Second email shown
                        metadata = cache.get_email_by_position(position)
                        if metadata:
                            if SHOW_ID_RESOLUTION:
                                print(f"\n🔍 [ID_RESOLVE] Context 'other email' → Position {position} ({metadata.id[:20]}...)")
                            return metadata.id
                elif "last" in email_id_lower:
                    # "the last email" → last in recently shown
                    position = context["recently_shown_emails"][-1]
                    metadata = cache.get_email_by_position(position)
                    if metadata:
                        if SHOW_ID_RESOLUTION:
                            print(f"\n🔍 [ID_RESOLVE] Context 'last email' → Position {position} ({metadata.id[:20]}...)")
                        return metadata.id
        
        # 7. Try to find by keyword (with normalization)
        matches = cache.find_all_email_matches(email_id)
        
        if len(matches) == 0:
            return {"error": f"Could not find email matching '{email_id}'"}
        
        elif len(matches) == 1:
            # Single match - perfect!
            metadata = matches[0]
            if SHOW_ID_RESOLUTION:
                print(f"\n🔍 [ID_RESOLVE] Keyword '{email_id}' → {metadata.from_name} ({metadata.id[:20]}...)")
            return metadata.id
        
        else:
            # Multiple matches - use context to disambiguate
            if context["recently_shown_emails"]:
                # Prefer emails from recently shown list
                for match in matches:
                    if match.position in context["recently_shown_emails"]:
                        if SHOW_ID_RESOLUTION:
                            print(f"\n🔍 [ID_RESOLVE] Keyword '{email_id}' (ambiguous, using context) → Position {match.position} ({match.id[:20]}...)")
                        return match.id
            
            # Still ambiguous - return helpful error
            options = [f"{m.position}. {m.subject} from {m.from_name}" for m in matches]
            return {
                "error": f"Found {len(matches)} emails matching '{email_id}'. Please specify:",
                "options": options,
                "suggestion": f"Use position number ({', '.join([str(m.position) for m in matches])}) or be more specific"
            }
    
    def _resolve_draft_id(self, draft_id: str) -> Any:
        """Resolve position number or name to actual draft ID."""
        cache = self.state["cache"]
        
        if not cache.recent_drafts:
            return {"error": "No drafts cached. Use list_drafts first."}
        
        # Check if it's a position number
        if draft_id.isdigit():
            metadata = cache.get_draft_by_position(int(draft_id))
            if metadata:
                if SHOW_ID_RESOLUTION:
                    print(f"\n🔍 [ID_RESOLVE] Draft position {draft_id} → To {metadata.to_name} ({metadata.id[:20]}...)")
                return metadata.id
            else:
                return {"error": f"Draft position {draft_id} not found"}
        
        # Check if it's already an ID
        if draft_id.startswith("AAMk") or draft_id.startswith("AQMk"):
            if SHOW_ID_RESOLUTION:
                print(f"\n🔍 [ID_RESOLVE] Already a draft ID: {draft_id[:20]}...")
            return draft_id
        
        # Try to find by keyword
        metadata = cache.find_draft_by_keyword(draft_id)
        if metadata:
            if SHOW_ID_RESOLUTION:
                print(f"\n🔍 [ID_RESOLVE] Keyword '{draft_id}' → To {metadata.to_name} ({metadata.id[:20]}...)")
            return metadata.id
        
        return {"error": f"Could not find draft matching '{draft_id}'"}
    
    def _resolve_attachment_id(self, message_id: str, attachment_id: str) -> Any:
        """
        Resolve attachment position number or name to actual attachment ID.
        
        Args:
            message_id: Outlook message ID
            attachment_id: Position number (1, 2, 3...) or actual attachment ID
        
        Returns:
            Actual attachment ID or error dict
        """
        email = self.state["email_editor"]
        
        # Check if email is loaded
        if email.is_empty() or not email.is_received_email():
            return {"error": "No email loaded in EmailEditor. Read an email first."}
        
        # Check if message_id matches current email
        if email.id != message_id:
            return {"error": f"Message ID mismatch. Current email ID: {email.id[:20]}..."}
        
        # Check if email has attachments
        if not email.attachments:
            return {"error": "No attachments found for this email. Use get_email first to load attachments."}
        
        # Check if it's a position number
        if attachment_id.isdigit():
            position = int(attachment_id)
            if 1 <= position <= len(email.attachments):
                actual_id = email.attachments[position - 1]["id"]
                if SHOW_ID_RESOLUTION:
                    att_name = email.attachments[position - 1].get("name", "unknown")
                    print(f"\n🔍 [ID_RESOLVE] Attachment position {position} → {att_name} ({actual_id[:20]}...)")
                return actual_id
            else:
                return {"error": f"Attachment position {position} not found. Email has {len(email.attachments)} attachment(s)."}
        
        # Check if it's already an attachment ID (starts with AAMk or similar)
        if attachment_id.startswith("AAMk") or attachment_id.startswith("AQMk"):
            # Verify it exists in the attachments list
            for att in email.attachments:
                if att["id"] == attachment_id:
                    if SHOW_ID_RESOLUTION:
                        print(f"\n🔍 [ID_RESOLVE] Already an attachment ID: {attachment_id[:20]}...")
                    return attachment_id
            return {"error": f"Attachment ID '{attachment_id[:20]}...' not found in email attachments."}
        
        # Try to find by name (partial match)
        attachment_id_lower = attachment_id.lower()
        for att in email.attachments:
            att_name = att.get("name", "").lower()
            if attachment_id_lower in att_name or att_name in attachment_id_lower:
                if SHOW_ID_RESOLUTION:
                    print(f"\n🔍 [ID_RESOLVE] Attachment name '{attachment_id}' → {att['name']} ({att['id'][:20]}...)")
                return att["id"]
        
        return {"error": f"Could not find attachment matching '{attachment_id}'. Available: {[att.get('name', 'unknown') for att in email.attachments]}"}


def get_tool_executor(state: GraphState) -> ToolExecutor:
    """
    Get tool executor with state bound.
    
    Args:
        state: GraphState to be modified by tools
    
    Returns:
        ToolExecutor instance
    """
    return ToolExecutor(state)


def get_tool_schemas() -> List[Dict]:
    """
    Get tool schemas in OpenAI function calling format.
    
    These schemas are used by the LLM to understand what tools are available.
    
    Returns:
        List of tool definitions compatible with OpenAI API
    """
    return [
        # Email operations
        {
            "type": "function",
            "function": {
                "name": "list_emails",
                "description": "List recent emails from inbox. Updates cache with email metadata for quick lookups.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of emails to retrieve (default 10, max 50)"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to look back (default 7, e.g., 30 for last month, 180 for 6 months, 365 for a year)"
                        },
                        "unread_only": {
                            "type": "boolean",
                            "description": "Only show unread emails (default false)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_email",
                "description": "Load an email into EmailEditor by position number (1, 2, 3...) or sender name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "Position number (e.g., '3') or sender name (e.g., 'Sarah')"
                        }
                    },
                    "required": ["email_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_emails",
                "description": "Search emails by sender or keywords. Updates cache with results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_person": {
                            "type": "string",
                            "description": "Person's name or email to search from"
                        },
                        "keywords": {
                            "type": "string",
                            "description": "Keywords to search for in subject/body"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_messages",
                "description": "Search messages using full-text search across subject, body, sender, recipients. Does NOT require a specific sender - searches everything. Perfect for finding emails about specific topics, projects, or keywords. Uses Microsoft Graph API $search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Richmond project', 'BuildSmartr detailing', 'change orders Surrey')"
                        },
                        "top": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 20, max 50)"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_thread_context",
                "description": "Fetch the FULL history of a conversation thread. Returns every message in chronological order (replies, forwards) with full body content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thread_id": {
                            "type": "string",
                            "description": "The unique conversationId (Outlook) or threadId (Gmail)"
                        }
                    },
                    "required": ["thread_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_attachments",
                "description": "List all attachments for a specific email message. Returns attachment metadata (id, name, size, content_type). Note: get_email already auto-fetches attachments, so you usually don't need this unless you want attachments without loading the full email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Outlook message ID"
                        }
                    },
                    "required": ["message_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "download_attachment",
                "description": "Download attachment bytes for processing. Returns content_bytes, name, size, content_type. Use this to download files before passing to PDF tools or other processors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Outlook message ID"
                        },
                        "attachment_id": {
                            "type": "string",
                            "description": "Attachment ID or position number (1, 2, 3...)"
                        }
                    },
                    "required": ["message_id", "attachment_id"]
                }
            }
        },
        # Draft operations
        {
            "type": "function",
            "function": {
                "name": "create_draft",
                "description": "Create a new email draft. Loads into EmailEditor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject"
                        },
                        "body": {
                            "type": "string",
                            "description": "Complete HTML email body (must be valid HTML using <p>, <br>, <strong>, etc. - never markdown or plain text). Must include signature from user preferences at the end."
                        },
                        "cc": {
                            "type": "string",
                            "description": "CC recipients (comma-separated)"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_reply",
                "description": "Create a reply to current email in EmailEditor. Uses threading.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "body": {
                            "type": "string",
                            "description": "Complete HTML reply body (must be valid HTML using <p>, <br>, <strong>, etc. - never markdown or plain text). Must include signature from user preferences at the end."
                        }
                    },
                    "required": ["body"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "forward_email",
                "description": "Forward the email currently loaded in EmailEditor to another recipient. Outlook will automatically include the original email below your note.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "Recipient email address to forward to"
                        },
                        "body": {
                            "type": "string",
                            "description": "BRIEF intro note in HTML format (1-2 sentences max, e.g., '<p>Sharing this update with you.</p>'). DO NOT copy or quote the original email - Outlook includes it automatically. Must be valid HTML using <p>, <br> tags. Include signature from preferences at the end if appropriate."
                        },
                        "cc": {
                            "type": "string",
                            "description": "CC recipients (comma-separated)"
                        },
                        "bcc": {
                            "type": "string",
                            "description": "BCC recipients (comma-separated)"
                        }
                    },
                    "required": ["to"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_draft",
                "description": "Edit the current draft in EmailEditor. Can update subject, body, recipients.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "body": {
                            "type": "string",
                            "description": "Complete HTML email body (must be valid HTML using <p>, <br>, <strong>, etc. - never markdown or plain text). Must include signature from user preferences at the end."
                        },
                        "subject": {
                            "type": "string",
                            "description": "New subject"
                        },
                        "cc": {
                            "type": "string",
                            "description": "CC recipients"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_draft",
                "description": "Send the current draft in EmailEditor. Always ask for confirmation first.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_drafts",
                "description": "List all saved drafts. Updates cache with draft metadata.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of drafts to retrieve (default 20)"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_draft",
                "description": "Load a specific draft into EmailEditor. Use position number or recipient name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "draft_id": {
                            "type": "string",
                            "description": "Position number (e.g., '2') or recipient name (e.g., 'John')"
                        }
                    },
                    "required": ["draft_id"]
                }
            }
        },
        # Utility tools
        {
            "type": "function",
            "function": {
                "name": "resolve_person",
                "description": "Look up a person's email address by name using contacts and directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Person's name to resolve (e.g., 'Sarah Johnson')"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for factual information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "smart_email_search",
                "description": "Perplexity-style search over Outlook inbox + attachments. Answers natural language questions by: (1) generating multiple search queries, (2) searching emails in parallel, (3) grouping by threads, (4) skimming PDF attachments, (5) deep-reading most relevant attachments, (6) synthesizing final answer. Returns structured results with threads_used and all_threads_considered for follow-up questions. Use this for complex queries like 'What did X say about Y project last week?' or 'Did we confirm the rate with Z company?'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language question about emails/attachments (e.g., 'What did Harv say about Richmond project schedule?', 'Summarize emails from Sarah about Surrey change orders')"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_preference",
                "description": "Save user preference (tone, signature, user_name).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Preference key",
                            "enum": ["tone", "signature", "user_name"]
                        },
                        "value": {
                            "type": "string",
                            "description": "Preference value"
                        }
                    },
                    "required": ["key", "value"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "switch_provider",
                "description": "Switch the active email provider between 'outlook' and 'gmail'. Use this when the user explicitly asks to change providers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "description": "Provider to switch to",
                            "enum": ["outlook", "gmail"]
                        }
                    },
                    "required": ["provider"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "skim_pdf",
                "description": "Quickly skim a PDF attachment and return a 2-3 sentence summary for triage. Perfect for inbox 'catch me up' mode. Uses Gemini Flash - fast and cheap. Checks cache first to avoid re-skimming the same PDF. IMPORTANT: Use position number (1, 2, 3...) or attachment name - the tool will resolve it to the actual ID automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Outlook message ID (use email.id from EmailEditor - the currently loaded email)"
                        },
                        "attachment_id": {
                            "type": "string",
                            "description": "Attachment position number (1, 2, 3...) or attachment name. The tool will automatically resolve this to the actual attachment ID from email.attachments list."
                        }
                    },
                    "required": ["message_id", "attachment_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_pdf",
                "description": "Deep read a PDF attachment and return comprehensive analysis. Uses Gemini Pro for full document analysis (all pages). Perfect for answering specific questions or extracting detailed information. This is more thorough than skim_pdf but slower and more expensive - use when user needs details, not just a quick summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Outlook message ID (use email.id from EmailEditor - the currently loaded email)"
                        },
                        "attachment_id": {
                            "type": "string",
                            "description": "Attachment position number (1, 2, 3...) or attachment name. The tool will automatically resolve this to the actual attachment ID from email.attachments list."
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional specific question to answer about the PDF (e.g., 'What are the payment terms?', 'When is the delivery deadline?', 'What are the key obligations?'). If provided, the analysis will focus on answering this question in addition to comprehensive extraction."
                        }
                    },
                    "required": ["message_id", "attachment_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "skim_pdf_bytes",
                "description": "Skim a PDF from raw bytes (for use when PDF is already downloaded). Returns 2-3 sentence summary. Uses Gemini Flash - fast and cheap. Perfect for email search workflows where attachments are downloaded separately.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_bytes_base64": {
                            "type": "string",
                            "description": "Base64-encoded PDF bytes (use base64.b64encode(pdf_bytes).decode('utf-8'))"
                        },
                        "filename": {
                            "type": "string",
                            "description": "PDF filename for context (e.g., 'invoice.pdf')"
                        }
                    },
                    "required": ["pdf_bytes_base64", "filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_pdf_bytes",
                "description": "Deep read a PDF from raw bytes (for use when PDF is already downloaded). Returns comprehensive analysis. Uses Gemini Pro for full document analysis. Perfect for email search workflows where attachments are downloaded separately and need detailed extraction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_bytes_base64": {
                            "type": "string",
                            "description": "Base64-encoded PDF bytes (use base64.b64encode(pdf_bytes).decode('utf-8'))"
                        },
                        "filename": {
                            "type": "string",
                            "description": "PDF filename for context (e.g., 'contract.pdf')"
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional specific question to answer about the PDF"
                        }
                    },
                    "required": ["pdf_bytes_base64", "filename"]
                }
            }
        }
    ]


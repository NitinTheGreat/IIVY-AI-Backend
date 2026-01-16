"""
Gmail tools using Google API.
Standardized to match OutlookTools output format for compatibility.
"""
import os.path
import base64
import email
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup

# Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']
TOKEN_PATH = 'token_gmail.json'
CREDENTIALS_PATH = 'credentials_gmail.json'

class GmailTools:
    """Google Gmail API tools."""

    def __init__(self, credentials=None):
        """
        Initialize Gmail API service.
        
        Args:
            credentials: Optional credentials object (for Session/Web mode).
                         If None, falls back to local file/env (Terminal mode).
        """
        self.creds = credentials
        self._thread_local = threading.local()
        self._authenticate()

    @property
    def service(self):
        """Get thread-local Gmail service instance."""
        if not hasattr(self._thread_local, 'service'):
            if self.creds:
                try:
                    self._thread_local.service = build('gmail', 'v1', credentials=self.creds)
                    # print(f"✅ Gmail Service initialized for thread {threading.get_ident()}")
                except Exception as e:
                    print(f"❌ Failed to build Gmail service for thread: {e}")
                    return None
            else:
                return None
        return self._thread_local.service

    @service.setter
    def service(self, value):
        """Allow manual setting (used by _authenticate initially)."""
        self._thread_local.service = value

    def _authenticate(self):
        """Authenticate using provided creds, local file, or Environment Variable."""
        # 0. If credentials were passed in directly (Web Mode), use them.
        if self.creds:
            try:
                # Convert Dictionary to Credentials Object if needed
                if isinstance(self.creds, dict):
                    # We need to reconstruct the Credentials object from the dict
                    # The dict from auth_utils.py (exchange_code) usually has: access_token, refresh_token, etc.
                    
                    # Map standard OAuth2 response to Google Credentials format
                    # PRIORITY: Use client_id/secret from request body if provided, else env vars
                    client_id = self.creds.get("client_id") or os.environ.get("GOOGLE_WEB_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID")
                    client_secret = self.creds.get("client_secret") or os.environ.get("GOOGLE_WEB_CLIENT_SECRET") or os.environ.get("GOOGLE_CLIENT_SECRET")
                    
                    # Debug: Log which client_id is being used
                    print(f"🔑 Using client_id: {client_id[:20] if client_id else 'NONE'}...")
                    print(f"🔑 Client secret provided: {'YES' if client_secret else 'NO'}")
                    
                    creds_data = {
                        "token": self.creds.get("access_token"),
                        "refresh_token": self.creds.get("refresh_token"),
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scopes": SCOPES
                    }
                    self.creds = Credentials.from_authorized_user_info(creds_data, SCOPES)
                    
                    # Check if token needs refresh
                    if not self.creds.valid:
                        if self.creds.expired and self.creds.refresh_token:
                            print("🔄 Access token expired, refreshing...")
                            self.creds.refresh(Request())
                            print("✅ Token refreshed successfully")

                self.service = build('gmail', 'v1', credentials=self.creds)
                print(f"✅ Gmail Service initialized for user: {self.get_user_profile().get('email')}")
                return
            except Exception as e:
                print(f"⚠️ Provided credentials failed: {e}")
                import traceback
                traceback.print_exc()
                self.creds = None # Fallback to other methods

        import json
        
        # 1. Try to load from Environment Variable (Best for Streamlit Cloud Global Mode - Deprecated for Multi-user but kept for fallback)
        token_json_str = os.environ.get("GMAIL_TOKEN_JSON")
        if token_json_str:
            try:
                # Convert string to dict
                token_info = json.loads(token_json_str)
                self.creds = Credentials.from_authorized_user_info(token_info, SCOPES)
                print("✅ Authenticated via GMAIL_TOKEN_JSON environment variable")
            except Exception as e:
                print(f"⚠️ Failed to load token from environment: {e}")

        # 2. If no env var, try local file
        if not self.creds and os.path.exists(TOKEN_PATH):
            self.creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        
        # 3. Validation and Refresh
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Token refresh failed: {e}")
                    self.creds = None # Force re-auth

            # 4. Interactive Login (Local only)
            if not self.creds:
                if not os.path.exists(CREDENTIALS_PATH):
                    # Check if credentials are in env (JSON string)
                    creds_json_str = os.environ.get("GMAIL_CREDENTIALS_JSON")
                    
                    # Check if credentials are in env (Individual keys - DESKTOP)
                    desktop_id = os.environ.get("GOOGLE_DESKTOP_CLIENT_ID")
                    desktop_secret = os.environ.get("GOOGLE_DESKTOP_CLIENT_SECRET")

                    if creds_json_str:
                        # Write to temp file because InstalledAppFlow expects a file
                        with open(CREDENTIALS_PATH, 'w') as f:
                            f.write(creds_json_str)
                    elif desktop_id and desktop_secret:
                        # Construct the JSON structure dynamically
                        import json
                        creds_dict = {
                            "installed": {
                                "client_id": desktop_id,
                                "client_secret": desktop_secret,
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                                "redirect_uris": ["http://localhost"]
                            }
                        }
                        with open(CREDENTIALS_PATH, 'w') as f:
                            json.dump(creds_dict, f)
                    else:
                        print(f"⚠️  Gmail credentials not found at {CREDENTIALS_PATH} or env. Skipping Gmail init.")
                        return
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
                    # Use a fixed port to avoid random port issues if possible, but 0 is fine for local
                    self.creds = flow.run_local_server(port=0)
                    
                    # Save the new token locally
                    with open(TOKEN_PATH, 'w') as token:
                        token.write(self.creds.to_json())
                except OSError:
                     print("❌ Cannot run local server (likely running in cloud). Set 'GMAIL_TOKEN_JSON' in secrets.")
                     return

        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
            # print("✅ Gmail service initialized successfully")
        except HttpError as error:
            print(f"❌ An error occurred initializing Gmail: {error}")

    def get_user_profile(self) -> Dict[str, str]:
        """Get the authenticated user's profile information."""
        if not self.service: return {"name": "Unknown", "email": "Unknown"}
        try:
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                "name": "Gmail User", # Gmail API profile doesn't always give name easily without People API
                "email": profile.get("emailAddress", ""),
                "messages_total": profile.get("messagesTotal")
            }
        except Exception:
            return {"name": "Unknown", "email": "Unknown"}

    def _get_header(self, headers: List[Dict], name: str) -> str:
        """Helper to extract header value."""
        for h in headers:
            if h['name'].lower() == name.lower():
                return h['value']
        return ""
    
    def _parse_body(self, payload: Dict) -> Dict[str, str]:
        """Recursively extract body text and HTML."""
        body_text = ""
        body_html = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType')
                data = part.get('body', {}).get('data', '')
                
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode('utf-8')
                    if mime_type == 'text/plain':
                        body_text += decoded
                    elif mime_type == 'text/html':
                        body_html += decoded
                
                # Recursive for nested parts
                if 'parts' in part:
                    nested = self._parse_body(part)
                    body_text += nested['text']
                    body_html += nested['html']
        else:
            # Single part message
            data = payload.get('body', {}).get('data', '')
            mime_type = payload.get('mimeType')
            if data:
                decoded = base64.urlsafe_b64decode(data).decode('utf-8')
                if mime_type == 'text/plain':
                    body_text = decoded
                elif mime_type == 'text/html':
                    body_html = decoded

        return {"text": body_text, "html": body_html}

    def _standardize_message(self, msg_detail: Dict) -> Dict[str, Any]:
        """Convert Gmail format to our standard format (Outlook-like)."""
        headers = msg_detail.get('payload', {}).get('headers', [])
        
        subject = self._get_header(headers, 'Subject') or "(No Subject)"
        from_header = self._get_header(headers, 'From')
        
        # Parse From: "Name <email>"
        if '<' in from_header:
            from_name = from_header.split('<')[0].strip().replace('"', '')
            from_email = from_header.split('<')[1].strip('> ')
        else:
            from_name = from_header
            from_email = from_header
            
        # Date conversion
        internal_date = int(msg_detail.get('internalDate', 0)) / 1000.0
        dt = datetime.fromtimestamp(internal_date)
        received_iso = dt.isoformat()
        received_str = dt.strftime("%b %d, %I:%M %p")
        
        snippet = msg_detail.get('snippet', '')
        
        return {
            "id": msg_detail['id'],
            "subject": subject,
            "from_name": from_name,
            "from_email": from_email,
            "received": received_str,
            "receivedDateTime": received_iso, # Standardized field
            "preview": snippet, # Standardized field
            "bodyPreview": snippet, # For compatibility
            "is_read": 'UNREAD' not in msg_detail.get('labelIds', []),
            "has_attachments": 'hasAttachment' in self._get_header(headers, 'X-Gmail-Labels') or any(p.get('filename') for p in msg_detail.get('payload', {}).get('parts', [])), # Rough check
            "conversationId": msg_detail.get('threadId'), # Map threadId to conversationId
            "threadId": msg_detail.get('threadId')
        }

    def search_messages(
        self,
        query: str,
        top: int = 20,
        page_cursor: Optional[str] = None,
        lightweight: bool = False,
    ) -> Dict[str, Any]:
        """
        Search Gmail messages with optional pagination.

        Returns unified pagination fields:
        - page_cursor: Gmail nextPageToken (or None)
        - has_more: bool
        """
        if not self.service: return {"error": "Gmail service not initialized"}
        
        try:
            list_kwargs: Dict[str, Any] = {"userId": "me", "q": query, "maxResults": top}
            if page_cursor:
                list_kwargs["pageToken"] = page_cursor

            results = self.service.users().messages().list(**list_kwargs).execute()
            messages = results.get('messages', [])
            next_token = results.get("nextPageToken")

            # IMPORTANT:
            # When doing paginated "dragnet" retrieval, do NOT fetch full message bodies per hit.
            # That can explode into thousands of API calls and can hard-crash the process (native segfault).
            # Lightweight mode returns minimal fields needed to union thread IDs safely.
            email_list = []
            if lightweight:
                for m in messages:
                    email_list.append(
                        {
                            "id": m.get("id", ""),
                            "threadId": m.get("threadId", ""),
                            "conversationId": m.get("threadId", ""),
                            "subject": "",
                            "from_name": "",
                            "from_email": "",
                            "received": "",
                            "receivedDateTime": "",
                            "preview": "",
                            "bodyPreview": "",
                            "has_attachments": False,
                        }
                    )
            else:
                for msg in messages:
                    # Fetch details for each message to match Outlook's rich return
                    detail = self.service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                    email_list.append(self._standardize_message(detail))
            
            return {
                "emails": email_list,
                "count": len(email_list),
                "query": query,
                "success": True,
                "page_cursor": next_token,
                "has_more": bool(next_token)
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def get_email(self, email_id: str) -> Dict[str, Any]:
        """Get full email details."""
        if not self.service: return {"error": "Gmail service not initialized"}

        try:
            detail = self.service.users().messages().get(userId='me', id=email_id, format='full').execute()
            standardized = self._standardize_message(detail)
            
            # Parse body
            body_content = self._parse_body(detail.get('payload', {}))
            
            # Use BeautifulSoup to strip HTML if plain text is empty
            body_text = body_content['text']
            if not body_text and body_content['html']:
                soup = BeautifulSoup(body_content['html'], 'html.parser')
                body_text = soup.get_text('\n')
                
            standardized["body"] = body_text
            standardized["body_html"] = body_content['html']
            
            # Attachments check (more robust)
            parts = detail.get('payload', {}).get('parts', [])
            has_atts = False
            
            def check_atts(parts_list):
                nonlocal has_atts
                for p in parts_list:
                    if p.get('filename'): has_atts = True
                    if 'parts' in p: check_atts(p['parts'])
            
            check_atts(parts)
            standardized["has_attachments"] = has_atts
            
            return standardized
        except Exception as e:
            return {"error": str(e)}

    def get_conversation_threads(self, thread_id: str) -> Dict[str, Any]:
        """
        Fetch ALL messages in a specific Gmail thread.
        """
        if not self.service: return {"error": "Gmail service not initialized"}
        
        try:
            # Gmail has a native 'threads.get' method
            # format='full' ensures we get payloads (bodies)
            thread_data = self.service.users().threads().get(userId='me', id=thread_id, format='full').execute()
            messages_raw = thread_data.get('messages', [])
            
            thread_messages = []
            for msg_detail in messages_raw:
                # We reuse the standardization logic, but we need to ensure we parse the body
                standardized = self._standardize_message(msg_detail)
                
                # Parse body
                body_content = self._parse_body(msg_detail.get('payload', {}))
                
                # Use BeautifulSoup to strip HTML if plain text is empty (Copy of get_email logic)
                body_text = body_content['text']
                if not body_text and body_content['html']:
                    try:
                        soup = BeautifulSoup(body_content['html'], 'html.parser')
                        body_text = soup.get_text('\n')
                    except:
                        pass
                
                standardized["body"] = body_text
                standardized["body_html"] = body_content['html']
                
                # Attachments check (reuse logic from get_email or simple check)
                # standardized["has_attachments"] is already set by _standardize_message mostly
                
                thread_messages.append(standardized)
                
            return {
                "conversation_id": thread_id, # Gmail calls it threadId
                "message_count": len(thread_messages),
                "messages": thread_messages
            }
            
        except Exception as e:
             return {"error": f"Failed to fetch Gmail thread: {str(e)}", "conversation_id": thread_id, "messages": []}

    def list_attachments(self, message_id: str) -> Dict[str, Any]:
        """List attachments."""
        if not self.service: return {"error": "Gmail service not initialized"}
        
        try:
            detail = self.service.users().messages().get(userId='me', id=message_id, format='full').execute()
            parts = detail.get('payload', {}).get('parts', [])
            
            atts = []
            def extract_atts(parts_list):
                for p in parts_list:
                    if p.get('filename'):
                        atts.append({
                            "id": p['body'].get('attachmentId'),
                            "name": p['filename'],
                            "size": p['body'].get('size'),
                            "contentType": p['mimeType']
                        })
                    if 'parts' in p:
                        extract_atts(p['parts'])
            
            extract_atts(parts)
            return {"attachments": atts}
        except Exception as e:
            return {"error": str(e)}

    def download_attachment(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download attachment bytes."""
        if not self.service: return {"error": "Gmail service not initialized"}
        
        try:
            att = self.service.users().messages().attachments().get(userId='me', messageId=message_id, id=attachment_id).execute()
            data = att.get('data', '')
            file_data = base64.urlsafe_b64decode(data)
            
            # We need to find the name/type again since get_attachment doesn't return it
            # Fetch message to get metadata
            # Efficiency improvement: Pass name/type in arguments if possible, but for compatibility we fetch.
            
            # For now, just return bytes. The caller usually knows the name from list_attachments.
            # But standardized format expects name.
            
            # Quick lookup in message
            msg = self.service.users().messages().get(userId='me', id=message_id).execute()
            parts = msg.get('payload', {}).get('parts', [])
            
            name = "unknown"
            content_type = "application/octet-stream"
            
            def find_att(parts_list):
                nonlocal name, content_type
                for p in parts_list:
                    if p.get('body', {}).get('attachmentId') == attachment_id:
                        name = p.get('filename')
                        content_type = p.get('mimeType')
                    if 'parts' in p: find_att(p['parts'])
            
            find_att(parts)

            return {
                "content_bytes": file_data,
                "name": name,
                "content_type": content_type,
                "size": len(file_data)
            }
        except Exception as e:
            return {"error": str(e)}

# Singleton
_gmail_tools_instance = None
def get_gmail_tools(credentials=None):
    """
    Get or create the Gmail tools singleton.
    If credentials are provided, it forces a new instance AND updates singleton (Web Mode).
    """
    global _gmail_tools_instance
    
    # If credentials are provided, we are in Web Mode -> Always return a NEW instance for this user
    if credentials:
        print("🔄 Updating Gmail Singleton with WEB TOKEN instance")
        _gmail_tools_instance = GmailTools(credentials=credentials)
        return _gmail_tools_instance
        
    # Otherwise (Terminal Mode), use the singleton
    if _gmail_tools_instance is None:
        _gmail_tools_instance = GmailTools()
    return _gmail_tools_instance


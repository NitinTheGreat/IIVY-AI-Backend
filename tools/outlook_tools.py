"""
Outlook tools using Microsoft Graph API.
"""
import requests
import os
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timedelta
import json
import re
from urllib.parse import quote
from html.parser import HTMLParser
from msal import ConfidentialClientApplication, SerializableTokenCache
from config import (
    CLIENT_ID, CLIENT_SECRET, AUTHORITY, SCOPES, GRAPH_API_ENDPOINT, TOKEN_CACHE_PATH, DEBUG_MODE
)


class OutlookTools:
    """Microsoft Graph API tools for Outlook integration."""

    def __init__(self, token_dict=None):
        """
        Initialize MSAL client.
        
        Args:
            token_dict: Optional dictionary containing access_token (for Session/Web mode).
        """
        self.token_cache = SerializableTokenCache()
        self.access_token = None
        self.app = None # Will be init below

        # 0. Web Mode: Token passed directly
        if token_dict and "access_token" in token_dict:
            self.access_token = token_dict["access_token"]
            print(f"🔍 DEBUG: OutlookTools initialized with WEB TOKEN (Hash: {hash(self.access_token)})")
            # We don't strictly need self.app for simple requests if we have the token
            # But we init it anyway for consistency if needed
            pass

        # 1. Try loading from Environment Variable (Cloud Global)
        elif os.environ.get("MSAL_TOKEN_CACHE"):
             try:
                 self.token_cache.deserialize(os.environ.get("MSAL_TOKEN_CACHE"))
                 print("✅ Loaded Outlook token from environment variable")
             except Exception as e:
                 print(f"⚠️ Failed to load Outlook token from env: {e}")

        # 2. Try loading from File (Local)
        elif TOKEN_CACHE_PATH.exists():
            print(f"📂 DEBUG: Found local token file at {TOKEN_CACHE_PATH}")
            with open(TOKEN_CACHE_PATH, 'r') as f:
                self.token_cache.deserialize(f.read())

        self.app = ConfidentialClientApplication(
            CLIENT_ID,
            client_credential=CLIENT_SECRET,
            authority=AUTHORITY,
            token_cache=self.token_cache
        )
        
        if not self.access_token:
            self._authenticate()

    def _save_cache(self):
        """Save token cache to disk."""
        if self.token_cache.has_state_changed:
            with open(TOKEN_CACHE_PATH, 'w') as f:
                f.write(self.token_cache.serialize())

    def _authenticate(self):
        """Authenticate using authorization code flow with manual redirect."""
        accounts = self.app.get_accounts()

        if accounts:
            # Try silent authentication
            result = self.app.acquire_token_silent(SCOPES, account=accounts[0])
            if result and "access_token" in result:
                self.access_token = result["access_token"]
                self._save_cache()
                return

        # Need interactive authentication - use auth code flow
        redirect_uri = "http://localhost:7071/api/OutlookCallback"

        auth_url = self.app.get_authorization_request_url(
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            prompt="consent"  # Force consent screen every time
        )

        print("\n" + "="*70)
        print("🔐 MICROSOFT OUTLOOK AUTHENTICATION")
        print("="*70)
        print("\n📋 STEP 1: Open this URL in your browser:\n")
        print(f"   {auth_url}\n")
        print("📋 STEP 2: Sign in and grant permissions\n")
        print("📋 STEP 3: After redirect, copy the FULL URL from browser address bar\n")
        print("           (It will look like: http://localhost:7071/api/OutlookCallback?code=...)\n")
        print("="*70 + "\n")

        redirect_response = input("📥 Paste the full redirect URL here: ").strip()
        if not redirect_response:
            raise Exception("No redirect URL provided")

        if "code=" not in redirect_response:
            raise Exception("No authorization code found in URL. Make sure you copied the full URL after signing in.")

        import urllib.parse
        parsed = urllib.parse.urlparse(redirect_response)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" not in params:
            raise Exception("No authorization code found in URL")

        auth_code = params["code"][0]

        # Exchange code for token
        result = self.app.acquire_token_by_authorization_code(
            code=auth_code,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )

        if "access_token" in result:
            self.access_token = result["access_token"]
            self._save_cache()

            granted_scopes = result.get('scope', 'Unknown')
            print("\n✅ Authentication successful!")
            print(f"📋 Granted scopes: {granted_scopes}\n")

            required = ['Mail.Read', 'Mail.ReadWrite', 'Mail.Send', 'Contacts.Read', 'People.Read']
            missing = [s for s in required if s not in granted_scopes]
            if missing:
                print(f"⚠️  WARNING: Missing scopes: {', '.join(missing)}")
                print(f"   This may cause permission errors.\n")
        else:
            error_msg = result.get('error_description', result.get('error', 'Unknown error'))
            raise Exception(f"Authentication failed: {error_msg}")

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for Graph API requests."""
        if not self.access_token:
            self._authenticate()
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make a Graph API request with error handling."""
        url = f"{GRAPH_API_ENDPOINT}{endpoint}"
        headers = self._get_headers()

        # Add ConsistencyLevel for $search
        if params and "$search" in params:
            headers = dict(headers)
            headers["ConsistencyLevel"] = "eventual"

        if extra_headers:
            headers.update(extra_headers)

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=json_data, params=params)
            elif method.upper() == "PATCH":
                response = requests.patch(url, headers=headers, json=json_data, params=params)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            from config import DEBUG_MODE
            if DEBUG_MODE:
                print(f"DEBUG: {method} {url}")
                if params:
                    print(f"Params: {params}")
                print(f"-> Status {response.status_code}")

            response.raise_for_status()

            if response.status_code in [202, 204] or not response.content:
                return {"success": True, "status_code": response.status_code}

            try:
                return response.json()
            except ValueError:
                return {"success": True, "status_code": response.status_code, "content": response.text}

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                self._authenticate()
                return self._make_request(method, endpoint, json_data, params, extra_headers)
            error_text = e.response.text
            print(f"\n⚠️  Graph API Error Details:")
            print(f"   Status Code: {e.response.status_code}")
            print(f"   URL: {url}")
            if params:
                print(f"   Params: {params}")
            print(f"   Response: {error_text}\n")
            raise Exception(f"Graph API error: {error_text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")


    def resolve_person(self, name: str) -> Dict[str, str]:
        """
        Find a person's email address by name.
        Searches contacts and directory.
        Requires Contacts.Read and People.Read permissions.
        
        SECURITY NOTE: Uses /me/ endpoints which access the SIGNED-IN USER's data,
        NOT the developer's data. Each user authenticates separately and only sees
        their own contacts and organization directory.
        """
        # Try searching in the SIGNED-IN USER's personal contacts
        # /me/contacts = the authenticated user's Outlook contacts (not developer's)
        try:
            endpoint = f"/me/contacts"
            params = {
                "$filter": f"startswith(displayName,'{name}') or startswith(givenName,'{name}') or startswith(surname,'{name}')",
                "$top": 5
            }
            result = self._make_request("GET", endpoint, params=params)
            if result.get("value"):
                contact = result["value"][0]
                return {
                    "name": contact.get("displayName", name),
                    "email": contact.get("emailAddresses", [{}])[0].get("address", ""),
                    "source": "contacts"
                }
        except Exception as e:
            error_msg = str(e)
            if "Access is denied" in error_msg or "ErrorAccessDenied" in error_msg:
                raise Exception(
                    "❌ Missing required permissions!\n"
                    "   Please add these permissions in Azure:\n"
                    "   - Contacts.Read (Delegated)\n"
                    "   - People.Read (Delegated)\n"
                    "   Then grant admin consent and re-authenticate."
                )
            print(f"⚠️  Contacts search error: {error_msg}")

        # Try searching in the SIGNED-IN USER's organization directory
        # /me/people = the authenticated user's Azure AD directory (their org, not developer's)
        try:
            endpoint = f"/me/people"
            # Use $search with quotes escaped
            search_value = name.replace('"', r'\"')
            params = {"$search": search_value, "$top": 5}
            result = self._make_request("GET", endpoint, params=params)
            if result.get("value"):
                person = result["value"][0]
                return {
                    "name": person.get("displayName", name),
                    "email": person.get("scoredEmailAddresses", [{}])[0].get("address", ""),
                    "source": "directory"
                }
        except Exception as e:
            error_msg = str(e)
            if "Access is denied" in error_msg or "ErrorAccessDenied" in error_msg:
                raise Exception(
                    "❌ Missing required permissions!\n"
                    "   Please add these permissions in Azure:\n"
                    "   - Contacts.Read (Delegated)\n"
                    "   - People.Read (Delegated)\n"
                    "   Then grant admin consent and re-authenticate."
                )
            print(f"⚠️  People search error: {error_msg}")

        return {
            "name": name,
            "email": "",
            "source": "not_found",
            "message": f"Could not find email address for '{name}'. Please provide the full email address."
        }

    def create_draft(
        self,
        recipient: str,
        subject: str,
        body_pref: str = "friendly_short",
        signature: str = "",
        body_text: str = "",
        cc: str = "",
        bcc: str = ""
    ) -> Dict[str, Any]:
        """Create a new email draft."""
        # Pass through HTML exactly as LLM generated it (complete HTML fragment including signature)
        body_html = body_text if body_text else ""

        draft_data = {
            "subject": subject,
            "toRecipients": [{"emailAddress": {"address": recipient}}],
            "body": {"contentType": "HTML", "content": body_html}
        }

        if cc:
            cc_list = [email.strip() for email in cc.split(",") if email.strip()]
            draft_data["ccRecipients"] = [{"emailAddress": {"address": email}} for email in cc_list]

        if bcc:
            bcc_list = [email.strip() for email in bcc.split(",") if email.strip()]
            draft_data["bccRecipients"] = [{"emailAddress": {"address": email}} for email in bcc_list]

        result = self._make_request("POST", "/me/messages", json_data=draft_data)

        return {
            "draft_id": result.get("id", ""),
            "to": recipient,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "body_html": body_html,
            "created_at": result.get("createdDateTime", "")
        }

    def create_reply_draft(
        self,
        email_id: str,
        body_text: str,
        signature: str = "",
        reply_all: bool = False,
        cc: str = "",
        bcc: str = ""
    ) -> Dict[str, Any]:
        """Create a reply draft to an existing email."""
        try:
            endpoint = f"/me/messages/{email_id}/createReplyAll" if reply_all else f"/me/messages/{email_id}/createReply"
            result = self._make_request("POST", endpoint)
            draft_id = result.get("id", "")

            # Pass through HTML exactly as LLM generated it (complete HTML fragment including signature)
            body_html = body_text

            update_data = {"body": {"contentType": "HTML", "content": body_html}}

            if cc:
                cc_list = [email.strip() for email in cc.split(",") if email.strip()]
                existing_draft = self._make_request("GET", f"/me/messages/{draft_id}")
                existing_cc = existing_draft.get("ccRecipients", [])
                new_cc = existing_cc + [{"emailAddress": {"address": email}} for email in cc_list]
                update_data["ccRecipients"] = new_cc

            if bcc:
                bcc_list = [email.strip() for email in bcc.split(",") if email.strip()]
                update_data["bccRecipients"] = [{"emailAddress": {"address": email}} for email in bcc_list]

            updated = self._make_request("PATCH", f"/me/messages/{draft_id}", json_data=update_data)

            to_recipients = updated.get("toRecipients", [])
            to_emails = [r.get("emailAddress", {}).get("address", "") for r in to_recipients]

            return {
                "draft_id": draft_id,
                "to": ", ".join(to_emails),
                "cc": cc,
                "bcc": bcc,
                "subject": updated.get("subject", ""),
                "body_html": body_html,
                "is_reply": True,
                "reply_all": reply_all,
                "created_at": updated.get("createdDateTime", ""),
                "message": f"Reply draft created (threaded in conversation)"
            }

        except Exception as e:
            return {"error": f"Failed to create reply draft: {str(e)}", "draft_id": ""}

    def create_forward_draft(
        self,
        email_id: str,
        to: str,
        body_text: str = "",
        signature: str = "",
        cc: str = "",
        bcc: str = ""
    ) -> Dict[str, Any]:
        """Create a forward draft from an existing email."""
        try:
            result = self._make_request("POST", f"/me/messages/{email_id}/createForward")
            draft_id = result.get("id", "")

            if not draft_id:
                return {"error": "Failed to create forward draft", "details": result}

            update_data: Dict[str, Any] = {
                "toRecipients": [
                    {
                        "emailAddress": {
                            "address": to
                        }
                    }
                ]
            }

            if cc:
                cc_list = [email.strip() for email in cc.split(",") if email.strip()]
                update_data["ccRecipients"] = [
                    {"emailAddress": {"address": email}} for email in cc_list
                ]

            if bcc:
                bcc_list = [email.strip() for email in bcc.split(",") if email.strip()]
                update_data["bccRecipients"] = [
                    {"emailAddress": {"address": email}} for email in bcc_list
                ]

            if body_text or signature:
                # IMPORTANT: For a forward, Outlook will include the original message automatically.
                # We only set the intro/note (and optional signature) as the body content.
                intro_html = body_text if body_text else ""
                if signature:
                    intro_html = (intro_html + "<br>" if intro_html else "") + signature
                if intro_html:
                    update_data["body"] = {
                        "contentType": "HTML",
                        "content": intro_html
                    }

            updated = self._make_request(
                "PATCH",
                f"/me/messages/{draft_id}",
                json_data=update_data
            )

            to_recipients = updated.get("toRecipients", [])
            to_addresses = [
                recipient.get("emailAddress", {}).get("address", "")
                for recipient in to_recipients
            ]

            return {
                "draft_id": draft_id,
                "to": ", ".join([addr for addr in to_addresses if addr]),
                "cc": cc,
                "bcc": bcc,
                "subject": updated.get("subject", ""),
                "is_forward": True,
                "body_html": update_data.get("body", {}).get("content") if update_data.get("body") else None,
                "created_at": updated.get("createdDateTime", ""),
                "message": "Forward draft created"
            }

        except Exception as e:
            return {"error": f"Failed to create forward draft: {str(e)}", "draft_id": ""}


    def edit_draft(
        self,
        draft_id: str,
        recipient: str = None,
        cc: List[str] = None,
        bcc: List[str] = None,
        subject: str = None,
        body_text: str = None,
        signature: str = ""
    ) -> Dict[str, Any]:
        """Unified method to update any combination of draft fields in a single API call."""
        update_data: Dict[str, Any] = {}
        changes = []

        if recipient:
            update_data["toRecipients"] = [{"emailAddress": {"address": recipient}}]
            changes.append(f"recipient to {recipient}")

        if cc:
            update_data["ccRecipients"] = [{"emailAddress": {"address": email}} for email in cc]
            changes.append(f"CC to {', '.join(cc)}")

        if bcc:
            update_data["bccRecipients"] = [{"emailAddress": {"address": email}} for email in bcc]
            changes.append(f"BCC to {', '.join(bcc)}")

        if subject:
            update_data["subject"] = subject
            changes.append(f"subject to '{subject}'")

        body_html = None
        if body_text is not None:
            # Pass through HTML exactly as LLM generated it (complete HTML fragment including signature)
            body_html = body_text if body_text else ""

            update_data["body"] = {"contentType": "HTML", "content": body_html}
            changes.append("body")

        result = self._make_request("PATCH", f"/me/messages/{draft_id}", json_data=update_data)

        response = {
            "draft_id": draft_id,
            "updated_at": result.get("lastModifiedDateTime", ""),
            "message": f"Updated {', '.join(changes)}" if changes else "No changes made"
        }
        if body_html:
            response["body_html"] = body_html
        return response

    def send_draft(self, draft_id: str) -> Dict[str, Any]:
        """Send an email draft."""
        result = self._make_request("POST", f"/me/messages/{draft_id}/send")
        if result.get("success") or result.get("status_code") in [200, 202, 204]:
            return {
                "success": True,
                "draft_id": draft_id,
                "sent_at": datetime.now().isoformat(),
                "message": "Email sent successfully"
            }
        return {"success": False, "error": "Failed to send email", "details": result}

    def search_latest_from(self, sender_email: str, search_terms: Optional[Union[List[str], str]] = None) -> Dict[str, Any]:
        """
        Search for the latest email from a specific sender using a simple Graph filter.
        Requires an exact email address; optional keywords are filtered client-side.
        """

        def _normalize_terms(raw_terms: Optional[Union[List[str], str]]) -> List[str]:
            if not raw_terms:
                return []
            if isinstance(raw_terms, str):
                normalized = [raw_terms]
            else:
                normalized = [term for term in raw_terms if term]
            return [term.strip().lower() for term in normalized if term and term.strip()]

        if not sender_email or "@" not in sender_email:
            return {"error": "Sender must be a full email address (e.g., user@example.com)."}

        params = {
            "$filter": f"from/emailAddress/address eq '{sender_email}'",
            "$top": 10
        }

        result = self._make_request("GET", "/me/messages", params=params)
        messages = result.get("value", [])

        if not messages:
            return {
                "found": False,
                "message": "No emails found from this sender",
                "emails": [],
                "source": "basic_filter"
            }

        terms = _normalize_terms(search_terms)
        if terms:
            filtered = []
            for msg in messages:
                subject = (msg.get("subject") or "").lower()
                preview = ""
                body_preview = msg.get("bodyPreview")
                if isinstance(body_preview, str):
                    preview = body_preview.lower()
                if any(term in subject or term in preview for term in terms):
                    filtered.append(msg)
            if filtered:
                messages = filtered

        normalized_messages: List[Dict[str, Any]] = []
        for msg in sorted(messages, key=lambda m: m.get("receivedDateTime", ""), reverse=True):
            msg_copy = dict(msg)
            preview = msg_copy.get("bodyPreview")
            if isinstance(preview, dict):
                msg_copy["bodyPreview"] = preview.get("content", "")
            elif not isinstance(preview, str):
                msg_copy["bodyPreview"] = ""

            from_field = msg_copy.get("from")
            if isinstance(from_field, dict):
                email_address = from_field.get("emailAddress", {}) if from_field else {}
                if isinstance(email_address, dict):
                    msg_copy["from"] = email_address.get("address", "")
                elif isinstance(email_address, str):
                    msg_copy["from"] = email_address
            else:
                msg_copy["from"] = from_field if isinstance(from_field, str) else ""

            normalized_messages.append(msg_copy)

        if not normalized_messages:
            return {
                "found": False,
                "message": "No emails found from this sender",
                "emails": [],
                "source": "basic_filter"
            }

        latest = normalized_messages[0]

        return {
            "found": True,
            "subject": latest.get("subject", ""),
            "from": latest.get("from", ""),
            "received_at": latest.get("receivedDateTime", ""),
            "preview": latest.get("bodyPreview", ""),
            "message_id": latest.get("id", ""),
            "conversation_id": latest.get("conversationId", ""),
            "emails": normalized_messages,
            "source": "basic_filter"
        }

    def get_user_profile(self) -> Dict[str, str]:
        """Get the authenticated user's profile information."""
        result = self._make_request("GET", "/me")
        return {
            "name": result.get("displayName", ""),
            "email": result.get("mail", result.get("userPrincipalName", "")),
            "first_name": result.get("givenName", ""),
            "last_name": result.get("surname", "")
        }

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags and return plain text."""
        if not html_content:
            return ""
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<[^>]+>', '', html_content)
        html_content = html_content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<') \
                                   .replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
        html_content = re.sub(r'\n\s*\n', '\n\n', html_content)
        html_content = re.sub(r' +', ' ', html_content)
        return html_content.strip()

    def get_email(self, email_id: str, include_body: bool = True) -> Dict[str, Any]:
        """Read a specific email by ID."""
        try:
            endpoint = f"/me/messages/{email_id}"
            params = None
            if include_body:
                params = {"$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,conversationId,hasAttachments"}
            email = self._make_request("GET", endpoint, params=params)

            sender = email.get("from", {}).get("emailAddress", {})
            sender_name = sender.get("name", "Unknown")
            sender_email = sender.get("address", "")

            to_recipients = [r.get("emailAddress", {}).get("address", "") for r in email.get("toRecipients", [])]
            cc_recipients = [r.get("emailAddress", {}).get("address", "") for r in email.get("ccRecipients", [])]

            body_data = email.get("body", {})
            body_html = body_data.get("content", "")
            body_text = self._strip_html(body_html)

            max_body_length = 3000
            if len(body_text) > max_body_length:
                body_text = body_text[:max_body_length] + "\n\n[Email truncated for length...]"

            received_dt = email.get("receivedDateTime", "")
            try:
                dt = datetime.fromisoformat(received_dt.replace('Z', '+00:00'))
                received_str = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                received_str = received_dt

            return {
                "id": email.get("id", ""),
                "subject": email.get("subject", "(No Subject)"),
                "from_name": sender_name,
                "from_email": sender_email,
                "to": to_recipients,
                "cc": cc_recipients,
                "received": received_str,
                "body": body_text,
                "body_html": body_html,
                "conversation_id": email.get("conversationId", ""),
                "has_attachments": email.get("hasAttachments", False),
                "message": f"Email from {sender_name} about '{email.get('subject', '(No Subject)')}'"
            }

        except Exception as e:
            return {"error": f"Failed to read email: {str(e)}", "id": email_id}

    def get_conversation_threads(self, conversation_id: str) -> Dict[str, Any]:
        """
        Fetch ALL messages in a specific conversation thread.
        This ensures 'Total Recall' - getting every reply, forward, and update.
        """
        try:
            # OPTIMIZATION: Microsoft Graph API often errors with "InefficientFilter" if we combine
            # filtering by conversationId AND sorting by receivedDateTime.
            # Fix: Remove sorting from the API query. We will sort manually in Python.
            
            params = {
                "$filter": f"conversationId eq '{conversation_id}'",
                # "$orderby": "receivedDateTime asc",  <-- REMOVED to fix InefficientFilter error
                "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,conversationId,hasAttachments,bodyPreview"
            }
            
            # Note: We might need to paginate if the thread is HUGE (e.g. > 50 messages)
            # For now, we grab the top 100 which covers 99.9% of threads.
            params["$top"] = 100
            
            response = self._make_request("GET", "/me/messages", params=params)
            messages = response.get("value", [])
            
            # Manual Client-Side Sorting (Oldest -> Newest)
            messages.sort(key=lambda x: x.get("receivedDateTime", ""), reverse=False)
            
            # Normalize them immediately
            thread_messages = []
            for msg in messages:
                sender = msg.get("from", {}).get("emailAddress", {})
                sender_name = sender.get("name", "Unknown")
                sender_email = sender.get("address", "")
                
                body_data = msg.get("body", {})
                body_html = body_data.get("content", "")
                body_text = self._strip_html(body_html)
                
                received_dt = msg.get("receivedDateTime", "")
                try:
                    dt = datetime.fromisoformat(received_dt.replace('Z', '+00:00'))
                    received_str = dt.strftime("%b %d, %I:%M %p")
                except:
                    received_str = received_dt
                
                thread_messages.append({
                    "id": msg.get("id"),
                    "subject": msg.get("subject", "(No Subject)"),
                    "from_name": sender_name,
                    "from_email": sender_email,
                    "received": received_str,
                    "receivedDateTime": received_dt,
                    "body": body_text, # Full body
                    "body_html": body_html,
                    "has_attachments": msg.get("hasAttachments", False),
                    "preview": (msg.get("bodyPreview") or "")[:200]
                })
                
            return {
                "conversation_id": conversation_id,
                "message_count": len(thread_messages),
                "messages": thread_messages
            }
            
        except Exception as e:
            return {"error": f"Failed to fetch thread: {str(e)}", "conversation_id": conversation_id, "messages": []}

    def search_messages(self, query: str, top: int = 20, page_cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        Search messages using Microsoft Graph API $search query.
        
        This uses the full-text search capability to find messages matching
        keywords across subject, body, sender, recipients, etc.
        
        Args:
            query: Search query string (e.g., "Richmond project", "BuildSmartr detailing")
            top: Maximum number of results to return (default 20)
        
        Returns:
            Dict with:
                "emails": List of matching email objects
                "count": Number of results
                "query": Original query
                "success": True/False
                "page_cursor": unified cursor (Graph @odata.nextLink) or None
                "has_more": bool
        """
        try:
            if not query or not query.strip():
                return {
                    "error": "Search query cannot be empty",
                    "success": False
                }

            def _sanitize_graph_search(qs: str) -> str:
                """
                Graph /me/messages?$search uses KQL-like full-text search, NOT Gmail boolean syntax.
                Strip boolean operators, quotes, and punctuation that trigger 400s.
                """
                s = (qs or "").strip()
                # Remove our app-level pseudo operators (kept for Node A portability)
                s = re.sub(r"\bhas:attachments?\b", " ", s, flags=re.IGNORECASE)
                # Hyphens often trigger syntax issues in Graph search
                s = s.replace("-", " ")
                # Drop boolean tokens and parentheses/quotes
                s = re.sub(r"\bOR\b|\bAND\b|\bNOT\b", " ", s, flags=re.IGNORECASE)
                s = s.replace("(", " ").replace(")", " ").replace("\"", " ")
                # Collapse whitespace
                s = re.sub(r"\s+", " ", s).strip()
                return s

            # Support lightweight "attachment bias" in the query string by translating
            # common tokens into a Graph $filter when possible.
            q = query.strip()
            wants_attachments = False
            for tok in ["has:attachment", "has:attachments", "hasAttachments:true", "has_attachments:true"]:
                if tok in q:
                    wants_attachments = True
                    q = q.replace(tok, "").strip()

            # Paging:
            # - First page uses /me/messages with $search
            # - Next pages use the provided @odata.nextLink (full URL) as page_cursor
            if page_cursor:
                # Graph nextLink is a full URL. Call it directly.
                url = str(page_cursor)
                headers = self._get_headers()
                headers = dict(headers)
                headers["ConsistencyLevel"] = "eventual"
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                response = resp.json()
            else:
                # Use Graph API $search (full-text).
                # NOTE: Graph $search does not implement boolean operators like Gmail.
                # Keep the query as plain text; Graph handles relevance matching.
                q_sanitized = _sanitize_graph_search(q or query.strip())
                params: Dict[str, Any] = {
                    "$search": f"\"{q_sanitized}\"",
                    "$top": top,
                    "$select": "id,subject,from,receivedDateTime,isRead,hasAttachments,bodyPreview,conversationId",
                }
                response = self._make_request("GET", "/me/messages", params=params)

            messages = response.get("value", [])
            next_link = response.get("@odata.nextLink")
            
            # IMPORTANT: Graph does NOT support $filter with $search in many tenants.
            # If the caller wanted attachments only, enforce it client-side.
            if wants_attachments:
                messages = [m for m in messages if m.get("hasAttachments") is True]
            
            # Sort by date manually (since $orderby not supported with $search)
            messages.sort(key=lambda m: m.get("receivedDateTime", ""), reverse=True)
            
            # Normalize message format to match list_emails format
            email_list = []
            for msg in messages:
                sender = msg.get("from", {}).get("emailAddress", {})
                sender_name = sender.get("name", "Unknown")
                
                received_dt = msg.get("receivedDateTime", "")
                try:
                    dt = datetime.fromisoformat(received_dt.replace('Z', '+00:00'))
                    received_str = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    received_str = received_dt
                
                email_list.append({
                    "id": msg.get("id", ""),
                    "subject": msg.get("subject", "(No Subject)"),
                    "from_name": sender_name,
                    "from_email": sender.get("address", ""),
                    "received": received_str,
                    "receivedDateTime": received_dt,  # Keep ISO format for sorting
                    "preview": (msg.get("bodyPreview", "") or "")[:150],
                    "is_read": msg.get("isRead", False),
                    "has_attachments": msg.get("hasAttachments", False),
                    "conversation_id": msg.get("conversationId", "")
                })
            
            return {
                "emails": email_list,
                "count": len(email_list),
                "query": query,
                "success": True,
                "page_cursor": next_link,
                "has_more": bool(next_link)
            }
        
        except Exception as e:
            import traceback
            return {
                "error": f"Failed to search messages: {str(e)}",
                "query": query,
                "success": False,
                "traceback": traceback.format_exc()
            }

    def list_emails(self, limit: int = 10, days: int = 7, unread_only: bool = False) -> Dict[str, Any]:
        """List recent emails from the inbox."""
        try:
            filters = []
            date_from = datetime.utcnow() - timedelta(days=days)
            date_str = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
            filters.append(f"receivedDateTime ge {date_str}")
            if unread_only:
                filters.append("isRead eq false")

            params = {
                "$top": limit,
                "$orderby": "receivedDateTime desc",
                "$select": "id,subject,from,receivedDateTime,isRead,hasAttachments,bodyPreview,conversationId"
            }
            if filters:
                params["$filter"] = " and ".join(filters)

            response = self._make_request("GET", "/me/mailFolders/inbox/messages", params=params)
            emails = response.get("value", [])

            email_list = []
            for email in emails:
                sender = email.get("from", {}).get("emailAddress", {})
                sender_name = sender.get("name", "Unknown")

                received_dt = email.get("receivedDateTime", "")
                try:
                    dt = datetime.fromisoformat(received_dt.replace('Z', '+00:00'))
                    received_str = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    received_str = received_dt

                email_list.append({
                    "id": email.get("id", ""),
                    "subject": email.get("subject", "(No Subject)"),
                    "from_name": sender_name,
                    "from_email": sender.get("address", ""),
                    "received": received_str,
                    "preview": (email.get("bodyPreview", "") or "")[:150],
                    "is_read": email.get("isRead", False),
                    "has_attachments": email.get("hasAttachments", False),
                    "conversation_id": email.get("conversationId", "")
                })

            return {"count": len(email_list), "emails": email_list, "message": f"Found {len(email_list)} email(s) from the last {days} day(s)"}

        except Exception as e:
            return {"error": f"Failed to list emails: {str(e)}", "count": 0, "emails": []}

    def list_drafts(self, limit: int = 20) -> Dict[str, Any]:
        """List all email drafts."""
        try:
            params = {
                "$top": limit,
                "$orderby": "lastModifiedDateTime desc",
                "$select": "id,subject,toRecipients,ccRecipients,lastModifiedDateTime,bodyPreview,hasAttachments"
            }
            response = self._make_request("GET", "/me/mailFolders/drafts/messages", params=params)
            drafts = response.get("value", [])

            draft_list = []
            for draft in drafts:
                to_recipients = draft.get("toRecipients", [])
                to_names = [r.get("emailAddress", {}).get("name", r.get("emailAddress", {}).get("address", "Unknown")) for r in to_recipients]
                to_str = ", ".join(to_names) if to_names else "(No recipient)"

                modified_dt = draft.get("lastModifiedDateTime", "")
                try:
                    dt = datetime.fromisoformat(modified_dt.replace('Z', '+00:00'))
                    modified_str = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    modified_str = modified_dt

                draft_list.append({
                    "id": draft.get("id", ""),
                    "subject": draft.get("subject", "(No Subject)"),
                    "to": to_str,
                    "to_list": to_names,
                    "modified": modified_str,
                    "preview": (draft.get("bodyPreview", "") or "")[:150],
                    "has_attachments": draft.get("hasAttachments", False)
                })

            return {"count": len(draft_list), "drafts": draft_list, "message": f"Found {len(draft_list)} draft(s)"}

        except Exception as e:
            return {"error": f"Failed to list drafts: {str(e)}", "count": 0, "drafts": []}

    def get_draft(self, draft_id: str) -> Dict[str, Any]:
        """Get a specific draft by ID."""
        try:
            params = {"$select": "id,subject,from,toRecipients,ccRecipients,bccRecipients,body,lastModifiedDateTime,hasAttachments"}
            draft = self._make_request("GET", f"/me/messages/{draft_id}", params=params)

            to_recipients = [r.get("emailAddress", {}).get("address", "") for r in draft.get("toRecipients", [])]
            cc_recipients = [r.get("emailAddress", {}).get("address", "") for r in draft.get("ccRecipients", [])]
            bcc_recipients = [r.get("emailAddress", {}).get("address", "") for r in draft.get("bccRecipients", [])]

            body_data = draft.get("body", {})
            body_html = body_data.get("content", "")
            body_text = self._strip_html(body_html)

            modified_dt = draft.get("lastModifiedDateTime", "")
            try:
                dt = datetime.fromisoformat(modified_dt.replace('Z', '+00:00'))
                modified_str = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                modified_str = modified_dt

            return {
                "id": draft.get("id", ""),
                "subject": draft.get("subject", "(No Subject)"),
                "to": to_recipients,
                "cc": cc_recipients,
                "bcc": bcc_recipients,
                "body": body_text,
                "body_html": body_html,
                "modified": modified_str,
                "has_attachments": draft.get("hasAttachments", False),
                "message": f"Draft to {', '.join(to_recipients) if to_recipients else '(no recipient)'}"
            }

        except Exception as e:
            return {"error": f"Failed to get draft: {str(e)}", "id": draft_id}
    
    def list_attachments(self, message_id: str) -> Dict[str, Any]:
        """
        List all attachments for a message.
        
        Args:
            message_id: Outlook message ID
        
        Returns:
            {"attachments": [{"id": "...", "name": "...", "size": ..., "contentType": "..."}]}
        """
        try:
            endpoint = f"/me/messages/{message_id}/attachments"
            params = {"$select": "id,name,size,contentType"}
            
            response = self._make_request("GET", endpoint, params=params)
            attachments = response.get("value", [])
            
            # Filter and format attachment info
            attachment_list = []
            for att in attachments:
                attachment_list.append({
                    "id": att.get("id", ""),
                    "name": att.get("name", "unknown"),
                    "size": att.get("size", 0),
                    "contentType": att.get("contentType", "application/octet-stream")
                })
            
            return {
                "attachments": attachment_list,
                "count": len(attachment_list),
                "message": f"Found {len(attachment_list)} attachment(s)"
            }
        
        except Exception as e:
            return {"error": f"Failed to list attachments: {str(e)}", "attachments": []}
    
    def download_attachment(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """
        Download attachment bytes from a message.
        
        Args:
            message_id: Outlook message ID
            attachment_id: Outlook attachment ID
        
        Returns:
            {"name": "...", "content_bytes": b"...", "content_type": "..."}
            or {"error": "..."}
        """
        try:
            endpoint = f"/me/messages/{message_id}/attachments/{attachment_id}"
            
            attachment = self._make_request("GET", endpoint)
            
            # Extract attachment data
            name = attachment.get("name", "unknown")
            content_type = attachment.get("contentType", "application/octet-stream")
            
            # Get content bytes
            # Attachments come back as base64 encoded in the "contentBytes" field
            import base64
            content_bytes_b64 = attachment.get("contentBytes", "")
            
            if not content_bytes_b64:
                return {"error": f"No content bytes found for attachment {name}"}
            
            # Decode base64 to bytes
            content_bytes = base64.b64decode(content_bytes_b64)
            
            return {
                "name": name,
                "content_bytes": content_bytes,
                "content_type": content_type,
                "size": len(content_bytes),
                "message": f"Downloaded {name} ({len(content_bytes)} bytes)"
            }
        
        except Exception as e:
            return {"error": f"Failed to download attachment: {str(e)}"}


# Singleton instance
_outlook_tools_instance = None

def get_outlook_tools(token_dict=None) -> OutlookTools:
    """
    Get or create the Outlook tools singleton.
    If token_dict is provided, forces a new instance AND updates the singleton (Web Mode).
    """
    global _outlook_tools_instance
    
    if token_dict:
        # WEB MODE: Create new instance and OVERWRITE the singleton
        # This ensures subsequent calls without args (like in nodes) use this authenticated instance
        print("🔄 Updating Outlook Singleton with WEB TOKEN instance")
        _outlook_tools_instance = OutlookTools(token_dict=token_dict)
        return _outlook_tools_instance
        
    if _outlook_tools_instance is None:
        _outlook_tools_instance = OutlookTools()
    return _outlook_tools_instance

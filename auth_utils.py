"""
OAuth2 helper functions for Streamlit Authentication.
Handles Google and Microsoft OAuth flows.
"""
import os
import requests
import urllib.parse
import streamlit as st
from typing import Dict, Any, Optional

# ============================================================
# CONFIGURATION
# ============================================================

# AUTO-DETECT ENVIRONMENT
# Robust check: If we are on Streamlit Cloud, we usually don't have the manual 'IS_LOCAL_DEV' flag.
# We will default to CLOUD unless the user explicitly sets 'IS_LOCAL_DEV=true' in their .env file.

is_local_flag = os.environ.get("IS_LOCAL_DEV", "false").lower() == "true"

if is_local_flag:
    REDIRECT_URI = "http://localhost:8501"
    print("🔗 Environment: LOCAL (Redirect to localhost)")
else:
    REDIRECT_URI = "https://donna-email-assistant.streamlit.app"
    print("🔗 Environment: CLOUD (Redirect to streamlit.app)")

print(f"🔗 Auth Redirect URI set to: {REDIRECT_URI}")

# ============================================================
# GOOGLE OAUTH
# ============================================================
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]

def get_google_auth_url() -> str:
    """Generate Google Login URL."""
    # Use WEB client ID for Streamlit
    client_id = os.environ.get("GOOGLE_WEB_CLIENT_ID")
    if not client_id:
        # Fallback to old name if new one isn't set (migration safety)
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
    
    if not client_id:
        return "#error-missing-google-client-id"
        
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": "google"  # To identify provider on return
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"

def exchange_google_code(code: str) -> Optional[Dict[str, Any]]:
    """Exchange authorization code for tokens."""
    # Use WEB keys for Streamlit
    client_id = os.environ.get("GOOGLE_WEB_CLIENT_ID") or os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_WEB_CLIENT_SECRET") or os.environ.get("GOOGLE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("❌ Missing Google Client ID/Secret")
        return None
        
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI
    }
    
    try:
        response = requests.post(GOOGLE_TOKEN_URL, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Google Token Exchange Failed: {e}")
        if 'response' in locals():
             print(f"Response Body: {response.text}") # Print detailed error from Google
        return None

# ============================================================
# MICROSOFT OAUTH
# ============================================================
MICROSOFT_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MICROSOFT_SCOPES = [
    "User.Read",
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "Contacts.Read",
    "People.Read",
    "offline_access"
]

def get_microsoft_auth_url() -> str:
    """Generate Microsoft Login URL."""
    client_id = os.environ.get("MICROSOFT_CLIENT_ID")
    if not client_id:
        return "#error-missing-microsoft-client-id"
        
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "response_mode": "query",
        "scope": " ".join(MICROSOFT_SCOPES),
        "state": "microsoft"
    }
    return f"{MICROSOFT_AUTH_URL}?{urllib.parse.urlencode(params)}"

def exchange_microsoft_code(code: str) -> Optional[Dict[str, Any]]:
    """Exchange authorization code for tokens."""
    client_id = os.environ.get("MICROSOFT_CLIENT_ID")
    client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("❌ Missing Microsoft Client ID/Secret")
        return None
        
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": " ".join(MICROSOFT_SCOPES),
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    
    try:
        response = requests.post(MICROSOFT_TOKEN_URL, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Microsoft Token Exchange Failed: {e}")
        # print(response.text)
        return None


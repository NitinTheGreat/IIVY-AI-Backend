"""
Configuration and environment variables for Donna email assistant.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================
# MICROSOFT GRAPH API CONFIGURATION
# ============================================================
CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "common")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

SCOPES = [
    "User.Read", 
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Send",
    "Contacts.Read",
    "People.Read"
]

GRAPH_API_ENDPOINT = "https://graph.microsoft.com/v1.0"

# Store everything in the DonnaEmail project folder
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / ".donna_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_CACHE_PATH = CACHE_DIR / "msal_token.json"

# ============================================================
# LLM CONFIGURATION
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # For Gemini (PDF skimming)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Superhuman dispatcher - GPT-5 special access
LLM_TEMPERATURE = 0.2  # Zero = perfectly deterministic, no hallucinations

# PDF Skimmer Configuration (lightweight triage)
PDF_SKIMMER_MODEL = "gemini-2.5-flash"  # Latest Gemini Flash - fastest and most efficient
PDF_SKIMMER_TEMPERATURE = 0.1  # Very low for consistent summaries

# PDF Deep Reader Configuration (detailed analysis)
PDF_READER_MODEL = "gemini-2.5-flash"  # Latest Gemini Pro - most capable for deep reading
PDF_READER_TEMPERATURE = 0.2  # Slightly higher for nuanced analysis
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# ============================================================
# DEBUGGING
# ============================================================
VERBOSE_MODE = os.getenv("VERBOSE_MODE", "true").lower() == "true"  # Show thinking process
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"    # Extra debug info

# Detailed debugging flags
SHOW_STATE_CHANGES = VERBOSE_MODE  # Log EmailEditor state transitions
SHOW_CACHE_OPERATIONS = VERBOSE_MODE  # Log cache updates
SHOW_ID_RESOLUTION = VERBOSE_MODE  # Log position → ID resolution

# ============================================================
# WEB SEARCH
# ============================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")  # For web search via serper.dev
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")    # Alternative: Brave Search API
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Perplexity-style search (recommended)

# ============================================================
# PREFERENCES
# ============================================================
DEFAULT_TONE = "friendly_short"

# ============================================================
# GUARDRAILS
# ============================================================
REQUIRE_SEND_CONFIRMATION = True
QUIET_HOURS_START = 22  # 10 PM
QUIET_HOURS_END = 7     # 7 AM
ENABLE_QUIET_HOURS = False

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = CACHE_DIR / "donna.log"

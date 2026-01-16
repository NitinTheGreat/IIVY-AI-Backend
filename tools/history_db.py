import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel

# Configuration from Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

class HistoryDB:
    """
    Handles interactions with Supabase 'research_history' table.
    """
    
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            print("⚠️  Supabase URL/KEY not found. History will be in-memory only.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            except Exception as e:
                print(f"❌ Failed to connect to Supabase: {e}")
                self.client = None

    def save_card(self, user_email: str, card_data: Dict[str, Any]) -> bool:
        """
        Save a research card to Supabase.
        
        Args:
            user_email: The email address of the active provider (e.g., harv@buildsmartr.com)
            card_data: Dictionary containing topic, answer, entities, source, etc.
        """
        if not self.client:
            return False
            
        try:
            # Prepare payload matching table schema
            payload = {
                "provider_email": user_email,
                "topic": card_data.get("topic"),
                "answer": card_data.get("answer"),
                "entities": card_data.get("entities", []),
                "source": card_data.get("source", "email_research"),
                # created_at is handled by DB default, or we can pass it if we want to sync clocks
                # "created_at": datetime.utcnow().isoformat()
            }
            
            # Execute Insert
            self.client.table("research_history").insert(payload).execute()
            return True
            
        except Exception as e:
            print(f"❌ Failed to save history to Supabase: {e}")
            return False

    def get_recent_cards(self, user_email: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch recent research history for a specific email account.
        """
        if not self.client:
            return []
            
        try:
            response = self.client.table("research_history") \
                .select("*") \
                .eq("provider_email", user_email) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
                
            return response.data
            
        except Exception as e:
            print(f"❌ Failed to fetch history from Supabase: {e}")
            return []

# Singleton
_history_db_instance = None

def get_history_db():
    global _history_db_instance
    if _history_db_instance is None:
        _history_db_instance = HistoryDB()
    return _history_db_instance


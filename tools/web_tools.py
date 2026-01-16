"""
Web search tools for fetching external information.
"""
import requests
from typing import Dict, List, Any, Optional
from config import SERPER_API_KEY, BRAVE_API_KEY


class WebSearchTools:
    """Web search integration using Serper.dev or Brave Search API."""
    
    def __init__(self):
        """Initialize with available API key."""
        self.serper_key = SERPER_API_KEY
        self.brave_key = BRAVE_API_KEY
        
        if not self.serper_key and not self.brave_key:
            print("⚠️  No web search API key configured. Web search will be limited.")
    
    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        Search the web for information.
        Returns concise factual answers.
        """
        if self.serper_key:
            return self._search_serper(query, num_results)
        elif self.brave_key:
            return self._search_brave(query, num_results)
        else:
            return {
                "success": False,
                "query": query,
                "answer": "Web search not configured. Please set SERPER_API_KEY or BRAVE_API_KEY.",
                "sources": []
            }
    
    def _search_serper(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Serper.dev API."""
        url = "https://google.serper.dev/search"
        
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract answer box if available
            answer = data.get("answerBox", {}).get("answer", "")
            
            # If no answer box, try knowledge graph
            if not answer:
                answer = data.get("knowledgeGraph", {}).get("description", "")
            
            # Get organic results
            results = []
            for item in data.get("organic", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", "")
                })
            
            # If still no answer, use first snippet
            if not answer and results:
                answer = results[0]["snippet"]
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": results
            }
        
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "answer": f"Search failed: {str(e)}",
                "sources": []
            }
    
    def _search_brave(self, query: str, num_results: int) -> Dict[str, Any]:
        """Search using Brave Search API."""
        url = "https://api.search.brave.com/res/v1/web/search"
        
        headers = {
            "X-Subscription-Token": self.brave_key,
            "Accept": "application/json"
        }
        
        params = {
            "q": query,
            "count": num_results
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract results
            results = []
            for item in data.get("web", {}).get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "link": item.get("url", "")
                })
            
            # Use first snippet as answer
            answer = results[0]["snippet"] if results else "No results found"
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": results
            }
        
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "answer": f"Search failed: {str(e)}",
                "sources": []
            }


# Singleton instance
_web_search_instance = None

def get_web_search_tools() -> WebSearchTools:
    """Get or create the web search tools singleton."""
    global _web_search_instance
    if _web_search_instance is None:
        _web_search_instance = WebSearchTools()
    return _web_search_instance


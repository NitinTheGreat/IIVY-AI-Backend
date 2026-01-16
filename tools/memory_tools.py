"""
Memory tools for storing and retrieving user preferences and context.
"""
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from config import CACHE_DIR


class MemoryTools:
    """Simple file-based memory storage for preferences and context."""
    
    def __init__(self):
        """Initialize memory storage."""
        self.memory_file = CACHE_DIR / "memory.json"
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load memory: {e}")
                return {}
        return {}
    
    def _save_memory(self):
        """Save memory to disk."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save memory: {e}")
    
    def read(self, scope: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read from memory.
        
        Args:
            scope: Memory scope (e.g., 'preferences', 'contacts', 'context')
            keys: Optional list of specific keys to retrieve
        
        Returns:
            Dictionary with requested data
        """
        scope_data = self.memory.get(scope, {})
        
        if keys:
            return {key: scope_data.get(key) for key in keys if key in scope_data}
        
        return scope_data
    
    def write(self, scope: str, key: str, value: Any) -> Dict[str, Any]:
        """
        Write to memory.
        
        Args:
            scope: Memory scope (e.g., 'preferences', 'contacts', 'context')
            key: Key to store
            value: Value to store
        
        Returns:
            Success status
        """
        if scope not in self.memory:
            self.memory[scope] = {}
        
        self.memory[scope][key] = value
        self._save_memory()
        
        return {
            "success": True,
            "scope": scope,
            "key": key
        }
    
    def delete(self, scope: str, key: str) -> Dict[str, Any]:
        """Delete a key from memory."""
        if scope in self.memory and key in self.memory[scope]:
            del self.memory[scope][key]
            self._save_memory()
            return {"success": True, "deleted": True}
        
        return {"success": True, "deleted": False, "message": "Key not found"}
    
    def clear_scope(self, scope: str) -> Dict[str, Any]:
        """Clear an entire scope."""
        if scope in self.memory:
            del self.memory[scope]
            self._save_memory()
        
        return {"success": True, "scope": scope}
    
    def get_all_scopes(self) -> List[str]:
        """Get all memory scopes."""
        return list(self.memory.keys())


# Singleton instance
_memory_tools_instance = None

def get_memory_tools() -> MemoryTools:
    """Get or create the memory tools singleton."""
    global _memory_tools_instance
    if _memory_tools_instance is None:
        _memory_tools_instance = MemoryTools()
    return _memory_tools_instance


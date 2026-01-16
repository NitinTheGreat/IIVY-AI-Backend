"""
Provider-agnostic LLM interface for Donna SUPERHUMAN.

Supports multiple LLM providers: OpenAI, Gemini, Claude, etc.
The thinking node calls brain.step() without knowing which provider is active.

This abstraction allows easy switching between models via environment variables.
"""
# Suppress known deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE


@dataclass
class LLMResponse:
    """
    Standardized LLM response across all providers.
    
    Attributes:
        content: The text content of the response (or None if only tool calls)
        tool_calls: List of tool call objects (empty if no tools called)
    """
    content: Optional[str]
    tool_calls: List[Any]


class LLMBrain(ABC):
    """
    Abstract interface for LLM providers.
    
    Each provider implements the step() method which:
    1. Takes messages and tool schemas
    2. Calls the provider's API
    3. Returns standardized LLMResponse
    """
    
    @abstractmethod
    def step(self, messages: List[Dict], tool_schemas: List[Dict]) -> LLMResponse:
        """
        One inference step with tool calling support.
        
        Args:
            messages: Chat messages in standard format [{"role": "user", "content": "..."}]
            tool_schemas: Available tools in OpenAI function calling format
        
        Returns:
            LLMResponse with content and/or tool_calls
        """
        pass


class OpenAIBrain(LLMBrain):
    """
    OpenAI implementation (GPT-4, GPT-4o, GPT-4o-mini, etc.).
    
    Uses OpenAI's function calling API.
    """
    
    def __init__(self, model: str = None, temperature: float = None):
        """
        Initialize OpenAI brain.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini"). Defaults to config.
            temperature: Sampling temperature. Defaults to config.
        """
        from openai import OpenAI
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI brain")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
    
    def step(self, messages: List[Dict], tool_schemas: List[Dict]) -> LLMResponse:
        """
        Call OpenAI API with function calling.
        
        Args:
            messages: Chat messages
            tool_schemas: Tool definitions in OpenAI format
        
        Returns:
            LLMResponse with standardized format
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None,
            temperature=self.temperature
        )
        
        message = response.choices[0].message
        
        return LLMResponse(
            content=message.content,
            tool_calls=list(message.tool_calls) if message.tool_calls else []
        )


class GeminiBrain(LLMBrain):
    """
    Google Gemini implementation with function calling support.
    
    Supports:
    - Text generation with tool/function calling
    - File uploads (PDF, images, etc.)
    - Multi-modal content
    """
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.2):
        """
        Initialize Gemini brain.
        
        Args:
            model: Gemini model name (e.g., "gemini-2.0-flash-exp", "gemini-2.5-flash")
            temperature: Sampling temperature
        """
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.genai = genai
        
        # Initialize model with tools if needed
        self.client = genai.GenerativeModel(model)
    
    def step(self, messages: List[Dict], tool_schemas: List[Dict]) -> LLMResponse:
        """
        Call Gemini API with function calling.
        
        Args:
            messages: Chat messages in OpenAI format
            tool_schemas: Tool definitions in OpenAI format
        
        Returns:
            LLMResponse with standardized format
        """
        # Convert OpenAI tool schemas to Gemini function declarations
        tools = None
        if tool_schemas:
            tools = [self._convert_tool_schema(schema) for schema in tool_schemas]
        
        # Convert OpenAI messages to Gemini format
        gemini_messages = self._convert_messages(messages)
        
        # Create generation config
        generation_config = {
            "temperature": self.temperature,
        }
        
        try:
            # Initialize model
            if tools:
                model = self.genai.GenerativeModel(
                    self.model,
                    tools=tools,
                    generation_config=generation_config
                )
            else:
                model = self.genai.GenerativeModel(
                    self.model,
                    generation_config=generation_config
                )
            
            # Use chat session for multi-turn conversations
            # If only one user message, use direct generation
            if not gemini_messages:
                 # Should not happen if conversion is correct, but safe fallback
                 return LLMResponse(content="Error: No valid messages for Gemini.", tool_calls=[])

            if len(gemini_messages) == 1 and gemini_messages[0]["role"] == "user":
                # Single user message - direct generation
                content = gemini_messages[0]["parts"][0] if gemini_messages[0]["parts"] else ""
                response = model.generate_content(content)
            else:
                # Multi-turn conversation - use chat
                chat = model.start_chat(history=gemini_messages[:-1])
                last_message = gemini_messages[-1]
                content = last_message["parts"][0] if last_message["parts"] else ""
                response = chat.send_message(content)
            
            # Parse response
            return self._parse_response(response)
        
        except Exception as e:
            # Fallback: return error as text response
            import traceback
            return LLMResponse(
                content=f"Gemini API error: {str(e)}\n{traceback.format_exc()}",
                tool_calls=[]
            )
    
    def _convert_messages(self, messages: List[Dict]) -> List[Any]:
        """
        Convert OpenAI message format to Gemini format.
        
        OpenAI format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        Gemini format: [{"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}]
        """
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Skip system messages (Gemini doesn't have system role, merge into first user message)
            if role == "system":
                continue
            
            # Convert role
            gemini_role = "model" if role == "assistant" else "user"
            
            # Skip empty messages
            if not content and not msg.get("tool_calls"):
                continue
            
            # Handle tool calls
            if msg.get("tool_calls"):
                # Gemini represents function calls differently
                # For now, we'll include them as text in the message
                # Full implementation would convert to Gemini's function call format
                tool_text = f"\n[Called tools: {len(msg['tool_calls'])} function(s)]"
                content = (content or "") + tool_text
            
            gemini_messages.append({
                "role": gemini_role,
                "parts": [content] if content else []
            })
        
        return gemini_messages
    
    def _convert_tool_schema(self, openai_schema: Dict) -> Any:
        """
        Convert OpenAI tool schema to Gemini function declaration.
        
        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {...}}
            }
        }
        
        Gemini format:
        FunctionDeclaration(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {...}}
        )
        """
        from google.generativeai.types import FunctionDeclaration
        
        func = openai_schema.get("function", {})
        
        return FunctionDeclaration(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=func.get("parameters", {})
        )
    
    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Gemini response to standard LLMResponse format.
        
        Args:
            response: Gemini GenerateContentResponse
        
        Returns:
            LLMResponse with content and/or tool_calls
        """
        # Extract text content
        content = None
        if response.text:
            content = response.text
        
        # Extract function calls (if any)
        tool_calls = []
        
        # Check for function calls in response
        for part in response.parts:
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                
                # Convert to OpenAI-style tool call format
                import json
                import uuid
                
                tool_call = type('ToolCall', (), {
                    'id': f"call_{uuid.uuid4().hex[:24]}",
                    'function': type('Function', (), {
                        'name': fc.name,
                        'arguments': json.dumps(dict(fc.args))
                    })()
                })()
                
                tool_calls.append(tool_call)
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls
        )


class ClaudeBrain(LLMBrain):
    """
    Anthropic Claude implementation.
    
    TODO: Implement Claude API integration.
    Claude uses a different API format, so we'll need to:
    1. Convert messages to Claude format
    2. Convert tool schemas to Claude format
    3. Call Claude API
    4. Convert response back to standard format
    """
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", temperature: float = 0.2):
        """
        Initialize Claude brain.
        
        Args:
            model: Claude model name
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature
        
        # TODO: Initialize Claude client
        # from anthropic import Anthropic
        # self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def step(self, messages: List[Dict], tool_schemas: List[Dict]) -> LLMResponse:
        """
        Call Claude API with function calling.
        
        TODO: Implement Claude-specific logic:
        - Convert OpenAI message format to Claude format
        - Convert OpenAI tool schemas to Claude tools
        - Handle Claude's tool use response format
        """
        raise NotImplementedError(
            "Claude support coming soon. "
            "To use Claude, implement the conversion logic for:\n"
            "1. Messages format (OpenAI -> Claude)\n"
            "2. Tool schemas (OpenAI -> Claude tools)\n"
            "3. Response parsing (Claude -> LLMResponse)"
        )


def get_llm_brain(provider: str = None, model: str = None, temperature: float = None) -> LLMBrain:
    """
    Factory function to get the configured LLM brain.
    
    Provider selection priority:
    1. Explicit provider argument
    2. LLM_PROVIDER environment variable
    3. Default to "openai"
    
    Args:
        provider: LLM provider name ("openai", "gemini", "claude")
        model: Model name (provider-specific)
        temperature: Sampling temperature (0.0 to 1.0)
    
    Returns:
        LLMBrain instance for the specified provider
    
    Raises:
        ValueError: If provider is unknown or configuration is missing
    
    Examples:
        >>> brain = get_llm_brain()  # Uses defaults from config
        >>> brain = get_llm_brain(provider="openai", model="gpt-4o")
        >>> brain = get_llm_brain(provider="gemini", model="gemini-2.5-flash")
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    
    if provider == "openai":
        return OpenAIBrain(model=model, temperature=temperature)
    
    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        return GeminiBrain(model=model or "gemini-2.5-flash", temperature=temperature or 0.2)
    
    elif provider == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Claude. "
                "Get your API key from: https://console.anthropic.com/"
            )
        return ClaudeBrain(model=model or "claude-3-sonnet-20240229", temperature=temperature or 0.2)
    
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: openai, gemini, claude"
        )


# Convenience function for testing
def test_brain():
    """
    Test the LLM brain with a simple message.
    
    Usage:
        python -c "from llm_brain import test_brain; test_brain()"
    """
    brain = get_llm_brain()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
    
    response = brain.step(messages, tool_schemas=[])
    
    print(f"Provider: {brain.__class__.__name__}")
    print(f"Response: {response.content}")
    print(f"Tool calls: {len(response.tool_calls)}")


if __name__ == "__main__":
    test_brain()


"""
Base LLM Reasoner - Base class for LLM-based reasoning.

This module provides the base class that all LLM reasoners must extend.
"""

from typing import Any, Dict, List, Optional

from ..providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider
)

class BaseLLMReasoner:
    """Base class for LLM-based reasoning."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        vision: bool = False,
        base_url: Optional[str] = None
    ):
        """Initialize the base LLM reasoner.
        
        Args:
            llm_provider: LLM provider name
            model_name: Model name to use
            api_key: API key for the LLM provider
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            vision: Whether vision capabilities are enabled
            base_url: Optional base URL for the API
        """
        # Auto-detect Gemini models
        if model_name and model_name.startswith("gemini-"):
            llm_provider = "gemini"
            
        self.llm_provider = llm_provider.lower()
        
        # Initialize the appropriate provider
        provider_class = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "ollama": OllamaProvider
        }.get(self.llm_provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.provider = provider_class(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            vision=vision,
            base_url=base_url
        )

    def get_token_usage_stats(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return self.provider.get_token_usage_stats()

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_data: Optional[bytes] = None
    ) -> str:
        """Generate a response using the LLM.
        
        Args:
            system_prompt: System prompt string
            user_prompt: User prompt string
            screenshot_data: Optional screenshot data for vision tasks
            additional_context: Optional additional context dictionary
            
        Returns:
            Generated response string
        """
        return await self.provider.generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            screenshot_data=screenshot_data
        ) 
"""
Anthropic LLM provider for DroidRun.

This module provides the Anthropic (Claude) implementation for the Agent to use for reasoning.
"""

import os
from typing import Optional
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    """
    Anthropic-based LLM provider for Claude models.
    """
    
    def __init__(
        self, 
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Anthropic LLM provider.
        
        Args:
            model: Model name to use (e.g., 'claude-3-opus-20240229')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional parameters to pass to the base LLM class
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Validate required configuration
        if not api_key:
            raise ValueError("Anthropic API key not provided and not found in environment (ANTHROPIC_API_KEY)")
        
        # Initialize the base LLM with Anthropic configuration
        super().__init__(
            llm_provider="anthropic",
            model_name=model,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def from_env(cls, **kwargs) -> 'AnthropicLLM':
        """Create an instance using environment variables for configuration.
        
        Args:
            **kwargs: Override parameters from environment variables
            
        Returns:
            An AnthropicLLM instance configured from environment variables
        """
        return cls(**kwargs)
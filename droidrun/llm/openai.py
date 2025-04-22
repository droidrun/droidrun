"""
OpenAI LLM provider for DroidRun.

This module provides the OpenAI implementation for the Agent to use for reasoning.
"""

import os
from typing import Optional
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    """
    OpenAI-based LLM provider.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the OpenAI LLM provider.
        
        Args:
            model: Model name to use (e.g., 'gpt-4o', 'gpt-4o-mini')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional parameters to pass to the base LLM class
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Validate required configuration
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment (OPENAI_API_KEY)")
        
        # Initialize the base LLM with OpenAI configuration
        super().__init__(
            llm_provider="openai",
            model_name=model,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def from_env(cls, **kwargs) -> 'OpenAILLM':
        """Create an instance using environment variables for configuration.
        
        Args:
            **kwargs: Override parameters from environment variables
            
        Returns:
            An OpenAILLM instance configured from environment variables
        """
        return cls(**kwargs)
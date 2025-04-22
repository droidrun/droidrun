"""
Google Gemini LLM provider for DroidRun.

This module provides the Google Gemini implementation for the Agent to use for reasoning.
"""

import os
from typing import Optional
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    """
    Google Gemini-based LLM provider.
    """
    
    def __init__(
        self, 
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Gemini LLM provider.
        
        Args:
            model: Model name to use (e.g., 'gemini-2.0-flash')
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            **kwargs: Additional parameters to pass to the base LLM class
        """
        # Get API key from environment variable if not provided
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Validate required configuration
        if not api_key:
            raise ValueError("Gemini API key not provided and not found in environment (GEMINI_API_KEY)")
        
        # Initialize the base LLM with Gemini configuration
        super().__init__(
            llm_provider="gemini",
            model_name=model,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def from_env(cls, **kwargs) -> 'GeminiLLM':
        """Create an instance using environment variables for configuration.
        
        Args:
            **kwargs: Override parameters from environment variables
            
        Returns:
            A GeminiLLM instance configured from environment variables
        """
        return cls(**kwargs)
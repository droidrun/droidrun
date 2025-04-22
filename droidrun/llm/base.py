"""
Base LLM provider for DroidRun.

This module provides the base LLM class that all provider-specific implementations extend.
"""

from typing import Optional, Dict, Any, List
from droidrun.agent.llm_reasoning import LLMReasoner

class BaseLLM(LLMReasoner):
    """
    Base class for all LLM providers.
    
    This class extends the LLMReasoner and serves as a base for
    provider-specific implementations.
    """
    
    def __init__(
        self,
        llm_provider: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        vision: bool = False,
        **kwargs
    ):
        """Initialize the base LLM provider.
        
        Args:
            llm_provider: Provider name ('openai', 'azure', 'anthropic', or 'gemini')
            model_name: Model name to use
            api_key: API key for the LLM provider
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            vision: Whether vision capabilities are enabled
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            vision=vision,
            **kwargs
        )
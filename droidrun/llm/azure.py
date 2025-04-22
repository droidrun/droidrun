"""
Azure OpenAI LLM provider for DroidRun.

This module provides the Azure OpenAI implementation for the Agent to use for reasoning.
"""

import os
from typing import Optional, Dict, Any
from .base import BaseLLM

class AzureOpenAILLM(BaseLLM):
    """
    Azure OpenAI-based LLM provider.
    
    This class provides integration with Azure OpenAI service,
    allowing use of Azure-hosted models as an alternative to direct OpenAI API.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Azure OpenAI LLM provider.
        
        Args:
            model: Base model name (e.g., 'gpt-4')
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_KEY env var)
            azure_endpoint: Azure OpenAI endpoint URL (defaults to AZURE_OPENAI_ENDPOINT env var)
            azure_deployment: Azure OpenAI deployment name (defaults to AZURE_OPENAI_DEPLOYMENT env var)
            azure_api_version: Azure OpenAI API version (defaults to AZURE_OPENAI_API_VERSION env var)
            **kwargs: Additional parameters to pass to the base LLM class
        """
        # Get configuration from environment variables if not provided
        api_key = api_key or os.environ.get("AZURE_OPENAI_KEY")
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or "2023-05-15"
        
        # Validate required configuration
        if not api_key:
            raise ValueError("Azure OpenAI API key not provided and not found in environment (AZURE_OPENAI_KEY)")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint not provided and not found in environment (AZURE_OPENAI_ENDPOINT)")
        if not azure_deployment:
            raise ValueError("Azure OpenAI deployment name not provided and not found in environment (AZURE_OPENAI_DEPLOYMENT)")
        
        # Initialize the base LLM with Azure configuration
        super().__init__(
            llm_provider="azure",
            model_name=model,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version,
            **kwargs
        )
    
    @classmethod
    def from_env(cls, **kwargs) -> 'AzureOpenAILLM':
        """Create an instance using environment variables for configuration.
        
        Args:
            **kwargs: Override parameters from environment variables
            
        Returns:
            An AzureOpenAILLM instance configured from environment variables
        """
        return cls(**kwargs)
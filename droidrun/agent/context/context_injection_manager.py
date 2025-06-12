"""
Context Injection Manager - Manages specialized agent personas with dynamic tool and context injection.

This module provides the ContextInjectionManager class that manages different agent personas,
each with specific system prompts, contexts, and tool subsets tailored for specialized tasks.
"""

import logging
from typing import Optional, List
from droidrun.agent.context.agent_persona import AgentPersona
#import chromadb
import json
from pathlib import Path

logger = logging.getLogger("droidrun")

class ContextInjectionManager:
    """
    Manages different agent personas with specialized contexts and tool subsets.

    This class is responsible for:
    - Defining agent personas with specific capabilities
    - Injecting appropriate system prompts based on agent type
    - Filtering tool lists to match agent specialization
    - Providing context-aware configurations for CodeActAgent instances
    """

    def __init__(
            self,
            personas: List[AgentPersona]
        ):
        """Initialize the Context Injection Manager with predefined personas."""

        self.personas = {}
        for persona in personas:
            self.personas[persona.name] = persona


    def _load_persona(self, data: str) -> AgentPersona:
        persona = json.loads(data)
        logger.info(f"🎭 Loaded persona: {persona['name']}")
        return AgentPersona(
            name=persona['name'],
            system_prompt=persona['system_prompt'],
            allowed_tools=persona['allowed_tools'],
            description=persona['description'],
            expertise_areas=persona['expertise_areas'],
            user_prompt=persona['user_prompt'],
            required_context=persona['required_context'],
        )

    def get_persona(self, agent_type: str) -> Optional[AgentPersona]:
        """
        Get a specific agent persona by type.

        Args:
            agent_type: The type of agent ("UIExpert" or "AppStarterExpert")

        Returns:
            AgentPersona instance or None if not found
        """
        
        return self.personas.get(agent_type)
        
    def get_all_personas(self) -> List[str]:
        return self.personas

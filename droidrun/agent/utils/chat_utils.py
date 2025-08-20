import base64
import re
import inspect


import json
import logging
from typing import List, TYPE_CHECKING, Optional, Tuple
from droidrun.agent.context import Reflection
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

if TYPE_CHECKING:
    from droidrun.tools import Tools

logger = logging.getLogger("droidrun")

def message_copy(message: ChatMessage, deep = True) -> ChatMessage:
    if deep:
        copied_message = message.model_copy()
        copied_message.blocks = [block.model_copy () for block in message.blocks]

        return copied_message
    copied_message = message.model_copy()

    # Create a new, independent list containing the same block references
    copied_message.blocks = list(message.blocks) # or original_message.blocks[:]

    return copied_message

async def add_reflection_summary(reflection: Reflection, chat_history: List[ChatMessage]) -> List[ChatMessage]:
    """Add reflection summary and advice to help the planner understand what went wrong and what to do differently."""
    
    reflection_text = "\n### The last task failed. You have additional information about what happenend. \nThe Reflection from Previous Attempt:\n"
    
    if reflection.summary:
        reflection_text += f"**What happened:** {reflection.summary}\n\n"
    
    if reflection.advice:
        reflection_text += f"**Recommended approach for this retry:** {reflection.advice}\n"
    
    reflection_block = TextBlock(text=reflection_text)
    
    # Copy chat_history and append reflection block to the last message
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(reflection_block)
    
    return chat_history

def _format_ui_elements(ui_data, level=0) -> str:
    """Format UI elements in natural language: index. className: resourceId, text - bounds"""
    if not ui_data:
        return ""
    
    formatted_lines = []
    indent = "  " * level  # Indentation for nested elements
    
    # Handle both list and single element
    elements = ui_data if isinstance(ui_data, list) else [ui_data]
    
    for element in elements:
        if not isinstance(element, dict):
            continue
            
        # Extract element properties
        index = element.get('index', '')
        class_name = element.get('className', '')
        resource_id = element.get('resourceId', '')
        text = element.get('text', '')
        bounds = element.get('bounds', '')
        children = element.get('children', [])
        
        
        # Format the line: index. className: resourceId, text - bounds
        line_parts = []
        if index != '':
            line_parts.append(f"{index}.")
        if class_name:
            line_parts.append(class_name + ":")
        
        details = []
        if resource_id:
            details.append(f'"{resource_id}"')
        if text:
            details.append(f'"{text}"')
        if details:
            line_parts.append(", ".join(details))
        
        if bounds:
            line_parts.append(f"- ({bounds})")
        
        formatted_line = f"{indent}{' '.join(line_parts)}"
        formatted_lines.append(formatted_line)
        
        # Recursively format children with increased indentation
        if children:
            child_formatted = _format_ui_elements(children, level + 1)
            if child_formatted:
                formatted_lines.append(child_formatted)
    
    return "\n".join(formatted_lines)

async def add_ui_text_block(ui_state: str, chat_history: List[ChatMessage], copy = True) -> List[ChatMessage]:
    """Add UI elements to the chat history without modifying the original."""
    if ui_state:
        # Parse the JSON and format it in natural language
        try:
            ui_data = json.loads(ui_state) if isinstance(ui_state, str) else ui_state
            formatted_ui = _format_ui_elements(ui_data)
            ui_block = TextBlock(text=f"\nCurrent Clickable UI elements from the device in the schema 'index. className: resourceId, text - bounds(x1,y1,x2,y2)':\n{formatted_ui}\n")
        except (json.JSONDecodeError, TypeError):
            # Fallback to original format if parsing fails
            ui_block = TextBlock(text="\nCurrent Clickable UI elements from the device using the custom TopViewService:\n```json\n" + json.dumps(ui_state) + "\n```\n")
        
        if copy:
            chat_history = chat_history.copy()
            chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(ui_block)
    return chat_history

async def add_screenshot_image_block(screenshot, chat_history: List[ChatMessage], copy = True) -> None:
    if screenshot:
        image_block = ImageBlock(image=screenshot)
        if copy:
            chat_history = chat_history.copy()  # Create a copy of chat history to avoid modifying the original
            chat_history[-1] = message_copy(chat_history[-1])
        chat_history[-1].blocks.append(image_block)
    return chat_history


async def add_phone_state_block(phone_state, chat_history: List[ChatMessage]) -> List[ChatMessage]:
    
    # Format the phone state data nicely
    if isinstance(phone_state, dict) and 'error' not in phone_state:
        current_app = phone_state.get('currentApp', '')
        package_name = phone_state.get('packageName', 'Unknown')
        keyboard_visible = phone_state.get('keyboardVisible', False)
        focused_element = phone_state.get('focusedElement')
        
        # Format the focused element
        if focused_element:
            element_text = focused_element.get('text', '')
            element_class = focused_element.get('className', '')
            element_resource_id = focused_element.get('resourceId', '')
            
            # Build focused element description
            focused_desc = f"'{element_text}' {element_class}"
            if element_resource_id:
                focused_desc += f" | ID: {element_resource_id}"
        else:
            focused_desc = "None"
        
        phone_state_text = f"""
**Current Phone State:**
• **App:** {current_app} ({package_name})
• **Keyboard:** {'Visible' if keyboard_visible else 'Hidden'}
• **Focused Element:** {focused_desc}
        """
    else:
        # Handle error cases or malformed data
        if isinstance(phone_state, dict) and 'error' in phone_state:
            phone_state_text = f"\n📱 **Phone State Error:** {phone_state.get('message', 'Unknown error')}\n"
        else:
            phone_state_text = f"\n📱 **Phone State:** {phone_state}\n"
    
    ui_block = TextBlock(text=phone_state_text)
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(ui_block)
    return chat_history

async def add_packages_block(packages, chat_history: List[ChatMessage]) -> List[ChatMessage]:
    
    ui_block = TextBlock(text=f"\nInstalled packages: {packages}\n```\n")
    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(ui_block)
    return chat_history

async def add_memory_block(memory: List[str], chat_history: List[ChatMessage]) -> List[ChatMessage]:
    memory_block = "\n### Remembered Information:\n"
    for idx, item in enumerate(memory, 1):
        memory_block += f"{idx}. {item}\n"
    
    for i, msg in enumerate(chat_history):
        if msg.role == "user":
            if isinstance(msg.content, str):
                updated_content = f"{memory_block}\n\n{msg.content}"
                chat_history[i] = ChatMessage(role="user", content=updated_content)
            elif isinstance(msg.content, list):
                memory_text_block = TextBlock(text=memory_block)
                content_blocks = [memory_text_block] + msg.content
                chat_history[i] = ChatMessage(role="user", content=content_blocks)
            break
    return chat_history

async def get_reflection_block(reflections: List[Reflection]) -> ChatMessage:
    reflection_block = "\n### You also have additional Knowledge to help you guide your current task from previous expierences:\n"
    for reflection in reflections:
        reflection_block += f"**{reflection.advice}\n"
    
    return ChatMessage(role="user", content=reflection_block)
        
async def add_task_history_block(all_tasks: list[dict], chat_history: List[ChatMessage]) -> List[ChatMessage]:
    """Experimental task history with all previous tasks."""
    if not all_tasks:
        return chat_history

    lines = ["### Task Execution History (chronological):"]
    for index, task in enumerate(all_tasks, 1):
        description: str
        status_value: str

        if hasattr(task, "description") and hasattr(task, "status"):
            description = getattr(task, "description")
            status_value = getattr(task, "status") or "unknown"
        elif isinstance(task, dict):
            description = str(task.get("description", task))
            status_value = str(task.get("status", "unknown"))
        else:
            description = str(task)
            status_value = "unknown"

        indicator = f"[{status_value}]"

        lines.append(f"{index}. {indicator} {description}")

    task_block = TextBlock(text="\n".join(lines))

    chat_history = chat_history.copy()
    chat_history[-1] = message_copy(chat_history[-1])
    chat_history[-1].blocks.append(task_block)
    return chat_history

def parse_tool_descriptions(tool_list) -> str:
    """Parses the available tools and their descriptions for the system prompt."""
    logger.info("🛠️  Parsing tool descriptions...")
    tool_descriptions = []
    
    for tool in tool_list.values():
        assert callable(tool), f"Tool {tool} is not callable."
        tool_name = tool.__name__
        tool_signature = inspect.signature(tool)
        tool_docstring = tool.__doc__ or "No description available."
        formatted_signature = f"def {tool_name}{tool_signature}:\n    \"\"\"{tool_docstring}\"\"\"\n..."
        tool_descriptions.append(formatted_signature)
        logger.debug(f"  - Parsed tool: {tool_name}")
    descriptions = "\n".join(tool_descriptions)
    logger.info(f"🔩 Found {len(tool_descriptions)} tools.")
    return descriptions


def parse_persona_description(personas) -> str:
    """Parses the available agent personas and their descriptions for the system prompt."""
    logger.debug("👥 Parsing agent persona descriptions for Planner Agent...")
    
    if not personas:
        logger.warning("No agent personas provided to Planner Agent")
        return "No specialized agents available."
    
    persona_descriptions = []
    for persona in personas:
        # Format each persona with name, description, and expertise areas
        expertise_list = ", ".join(persona.expertise_areas) if persona.expertise_areas else "General tasks"
        formatted_persona = f"- **{persona.name}**: {persona.description}\n  Expertise: {expertise_list}"
        persona_descriptions.append(formatted_persona)
        logger.debug(f"  - Parsed persona: {persona.name}")
    
    # Join all persona descriptions into a single string
    descriptions = "\n".join(persona_descriptions)
    logger.debug(f"👤 Found {len(persona_descriptions)} agent personas.")
    return descriptions


def extract_code_and_thought(response_text: str) -> Tuple[Optional[str], str]:
    """
    Extracts code from Markdown blocks (```python ... ```) and the surrounding text (thought),
    handling indented code blocks.

    Returns:
        Tuple[Optional[code_string], thought_string]
    """
    logger.debug("✂️ Extracting code and thought from response...")
    code_pattern = r"^\s*```python\s*\n(.*?)\n^\s*```\s*?$"
    code_matches = list(re.finditer(code_pattern, response_text, re.DOTALL | re.MULTILINE))

    if not code_matches:
        logger.debug("  - No code block found. Entire response is thought.")
        return None, response_text.strip()

    extracted_code_parts = []
    for match in code_matches:
            code_content = match.group(1)
            extracted_code_parts.append(code_content)

    extracted_code = "\n\n".join(extracted_code_parts)


    thought_parts = []
    last_end = 0
    for match in code_matches:
        start, end = match.span(0)
        thought_parts.append(response_text[last_end:start])
        last_end = end
    thought_parts.append(response_text[last_end:])

    thought_text = "".join(thought_parts).strip()
    thought_preview = (thought_text[:100] + '...') if len(thought_text) > 100 else thought_text
    logger.debug(f"  - Extracted thought: {thought_preview}")

    return extracted_code, thought_text
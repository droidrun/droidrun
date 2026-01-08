"""Coordinate formatter - Format with normalized coordinates for coordinate click mode."""

from typing import Dict, Any, List, Optional, Tuple
from droidrun.tools.formatters.base import TreeFormatter
from droidrun.tools.coordinate import (
    ScreenSize,
    absolute_to_normalized,
    NORMALIZED_RANGE,
)


class CoordinateFormatter(TreeFormatter):
    """
    Format tree with normalized coordinates for coordinate-based clicking.
    
    This formatter outputs UI elements with normalized coordinates [0-1000]
    to enable LLMs to output precise click positions.
    """

    def __init__(self, screen_size: Optional[ScreenSize] = None):
        """
        Initialize formatter.
        
        Args:
            screen_size: Screen size for coordinate normalization.
                        If None, will use default 1080x2400.
        """
        self._screen_size = screen_size or ScreenSize(width=1080, height=2400)

    def set_screen_size(self, screen_size: ScreenSize) -> None:
        """Update screen size for coordinate normalization."""
        self._screen_size = screen_size

    def format(
        self, filtered_tree: Optional[Dict[str, Any]], phone_state: Dict[str, Any]
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Format device state with normalized coordinates.
        
        Returns:
            Tuple of (formatted_text, focused_text, a11y_tree, phone_state)
        """
        focused_text = self._get_focused_text(phone_state)

        if filtered_tree is None:
            a11y_tree = []
        else:
            a11y_tree = self._flatten_with_coordinates(filtered_tree, [1])

        phone_state_text = self._format_phone_state(phone_state)
        ui_elements_text = self._format_ui_elements_text(a11y_tree)

        formatted_text = f"{phone_state_text}\n\n{ui_elements_text}"

        return (formatted_text, focused_text, a11y_tree, phone_state)

    @staticmethod
    def _get_focused_text(phone_state: Dict[str, Any]) -> str:
        """Extract focused element text."""
        focused_element = phone_state.get("focusedElement")
        if focused_element:
            return focused_element.get("text", "")
        return ""

    @staticmethod
    def _format_phone_state(phone_state: Dict[str, Any]) -> str:
        """Format phone state."""
        if isinstance(phone_state, dict) and "error" not in phone_state:
            current_app = phone_state.get("currentApp", "")
            package_name = phone_state.get("packageName", "Unknown")
            focused_element = phone_state.get("focusedElement")
            is_editable = phone_state.get("isEditable", False)

            if focused_element and focused_element.get("text"):
                focused_desc = f"'{focused_element.get('text', '')}'"
            else:
                focused_desc = "''"

            phone_state_text = f"""**Current Phone State:**
• **App:** {current_app} ({package_name})
• **Keyboard:** {'Visible' if is_editable else 'Hidden'}
• **Focused Element:** {focused_desc}
• **Coordinate Range:** [0-{NORMALIZED_RANGE}] (normalized)"""
        else:
            if isinstance(phone_state, dict) and "error" in phone_state:
                phone_state_text = f"📱 **Phone State Error:** {phone_state.get('message', 'Unknown error')}"
            else:
                phone_state_text = f"📱 **Phone State:** {phone_state}"

        return phone_state_text

    def _format_ui_elements_text(self, a11y_tree: List[Dict[str, Any]]) -> str:
        """Format UI elements with normalized coordinates."""
        if a11y_tree:
            formatted_ui = self._format_ui_elements(a11y_tree)
            ui_elements_text = (
                "Current UI elements with normalized coordinates [0-1000]:\n"
                "Format: 'index. className: text - norm_bounds[x1,y1,x2,y2]'\n"
                f"{formatted_ui}"
            )
        else:
            ui_elements_text = (
                "Current UI elements with normalized coordinates [0-1000]:\n"
                "No UI elements found"
            )
        return ui_elements_text

    def _format_ui_elements(
        self, ui_data: List[Dict[str, Any]], level: int = 0
    ) -> str:
        """Format UI elements with normalized coordinates."""
        if not ui_data:
            return ""

        formatted_lines = []
        indent = "  " * level

        elements = ui_data if isinstance(ui_data, list) else [ui_data]

        for element in elements:
            if not isinstance(element, dict):
                continue

            index = element.get("index", "")
            class_name = element.get("className", "")
            text = element.get("text", "")
            norm_bounds = element.get("norm_bounds", "")
            children = element.get("children", [])

            line_parts = []
            if index != "":
                line_parts.append(f"{index}.")
            if class_name:
                line_parts.append(class_name + ":")

            if text:
                line_parts.append(f'"{text}"')

            if norm_bounds:
                line_parts.append(f"- [{norm_bounds}]")

            formatted_line = f"{indent}{' '.join(line_parts)}"
            formatted_lines.append(formatted_line)

            if children:
                child_formatted = self._format_ui_elements(children, level + 1)
                if child_formatted:
                    formatted_lines.append(child_formatted)

        return "\n".join(formatted_lines)

    def _flatten_with_coordinates(
        self, node: Dict[str, Any], counter: List[int]
    ) -> List[Dict[str, Any]]:
        """Recursively flatten tree with normalized coordinates."""
        results = []

        formatted = self._format_node_with_coordinates(node, counter[0])
        results.append(formatted)
        counter[0] += 1

        for child in node.get("children", []):
            results.extend(self._flatten_with_coordinates(child, counter))

        return results

    def _format_node_with_coordinates(
        self, node: Dict[str, Any], index: int
    ) -> Dict[str, Any]:
        """Format single node with normalized coordinates."""
        bounds = node.get("boundsInScreen", {})
        left = bounds.get("left", 0)
        top = bounds.get("top", 0)
        right = bounds.get("right", 0)
        bottom = bounds.get("bottom", 0)

        # Convert to normalized coordinates
        norm_x1, norm_y1 = absolute_to_normalized(left, top, self._screen_size)
        norm_x2, norm_y2 = absolute_to_normalized(right, bottom, self._screen_size)

        text = (
            node.get("text")
            or node.get("contentDescription")
            or node.get("resourceId")
            or node.get("className", "")
        )

        class_name = node.get("className", "")
        short_class = class_name.split(".")[-1] if class_name else ""

        return {
            "index": index,
            "resourceId": node.get("resourceId", ""),
            "className": short_class,
            "text": text,
            "bounds": f"{left},{top},{right},{bottom}",
            "norm_bounds": f"{norm_x1},{norm_y1},{norm_x2},{norm_y2}",
            "children": [],
        }


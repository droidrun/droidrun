"""IOSStateProvider — builds UIState from iOS portal accessibility data.

Parses the raw text-based accessibility tree returned by the iOS portal
into structured elements compatible with UIState. Screen dimensions and
focused element are provided by the portal.

Known limitations:
- Normalized coordinates untested on iOS
- No filter/formatter pipeline (iOS a11y tree is raw text, not structured JSON)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from droidrun.tools.driver.base import DeviceDriver
from droidrun.tools.ui.provider import StateProvider
from droidrun.tools.ui.state import UIState

logger = logging.getLogger("droidrun")

# Element types considered interactive on iOS.
_INTERACTIVE_TYPES = {
    "Button",
    "SearchField",
    "TextField",
    "Cell",
    "Switch",
    "Slider",
    "Stepper",
    "Picker",
    "Link",
}

_COORD_RE = re.compile(r"\{\{([0-9.]+),\s*([0-9.]+)\},\s*\{([0-9.]+),\s*([0-9.]+)\}\}")
_ELEMENT_TYPE_RE = re.compile(r"\s*(.+?),")
_LABEL_RE = re.compile(r"label:\s*'([^']*)'")
_IDENTIFIER_RE = re.compile(r"identifier:\s*'([^']*)'")
_PLACEHOLDER_RE = re.compile(r"placeholderValue:\s*'([^']*)'")
_VALUE_RE = re.compile(r"value:\s*([^,}]+)")


class IOSStateProvider(StateProvider):
    """Produces ``UIState`` from an iOS device's accessibility tree."""

    supported = {"element_index", "convert_point"}

    def __init__(self, driver: DeviceDriver, use_normalized: bool = False) -> None:
        super().__init__(driver)
        # TODO: normalized coordinates untested on iOS
        self.use_normalized = use_normalized

    async def get_state(self) -> UIState:
        raw = await self.driver.get_ui_tree()
        a11y_text = raw.get("a11y_raw", "")
        phone_state = raw.get("phone_state", {})

        elements = _parse_a11y_tree(a11y_text)

        # Screen size from portal
        screen_size = raw.get("screen_size", {})
        screen_width = int(screen_size["width"])
        screen_height = int(screen_size["height"])

        formatted_text = _format_elements(elements, screen_width, screen_height)
        focused_text = phone_state.get("focusedElement", "") or ""

        return UIState(
            elements=elements,
            formatted_text=formatted_text,
            focused_text=focused_text,
            phone_state=phone_state,
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_a11y_tree(a11y_text: str) -> List[Dict[str, Any]]:
    """Parse iOS accessibility tree text into structured elements.

    Moved verbatim from ``IOSTools._parse_ios_accessibility_tree``.
    """
    elements: List[Dict[str, Any]] = []
    element_index = 0

    for line in a11y_text.strip().split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("Attributes:")
            or stripped.startswith("Element subtree:")
            or stripped.startswith("Path to element:")
            or stripped.startswith("Query chain:")
        ):
            continue

        coord_match = _COORD_RE.search(line)
        if not coord_match:
            continue

        x, y, width, height = map(float, coord_match.groups())

        # Element type
        type_match = _ELEMENT_TYPE_RE.match(line)
        element_type = type_match.group(1).strip() if type_match else "Unknown"
        element_type = re.sub(r"^[→\s]+", "", element_type)

        # Only keep interactive elements
        if not any(it in element_type for it in _INTERACTIVE_TYPES):
            continue

        # Extract properties
        label_m = _LABEL_RE.search(line)
        label = label_m.group(1) if label_m else ""
        ident_m = _IDENTIFIER_RE.search(line)
        identifier = ident_m.group(1) if ident_m else ""
        ph_m = _PLACEHOLDER_RE.search(line)
        placeholder = ph_m.group(1) if ph_m else ""
        val_m = _VALUE_RE.search(line)
        value = val_m.group(1).strip() if val_m else ""

        text = label or identifier or placeholder or ""

        # Bounds in "left,top,right,bottom" format — compatible with UIState
        bounds_str = f"{int(x)},{int(y)},{int(x + width)},{int(y + height)}"

        elements.append(
            {
                "index": element_index,
                "type": element_type,
                "className": element_type,
                "text": text,
                "label": label,
                "identifier": identifier,
                "placeholder": placeholder,
                "value": value,
                "bounds": bounds_str,
                "rect": f"{x},{y},{width},{height}",
                "children": [],
            }
        )
        element_index += 1

    return elements


# ---------------------------------------------------------------------------
# Formatting for agent prompt
# ---------------------------------------------------------------------------


def _format_elements(
    elements: List[Dict[str, Any]],
    screen_width: int,
    screen_height: int,
) -> str:
    """Build the text representation shown to the agent."""
    schema = "'index. className: text - bounds(x1,y1,x2,y2)'"
    if not elements:
        return f"Current Clickable UI elements:\n{schema}:\nNo UI elements found"

    lines = [f"Current Clickable UI elements:\n{schema}:"]
    for el in elements:
        idx = el.get("index", 0)
        cls = el.get("className", "Unknown")
        text = el.get("text", "")
        bounds = el.get("bounds", "")

        parts = [f"{idx}. {cls}:"]
        if text:
            parts.append(text)
        if bounds:
            parts.append(f"- ({bounds})")
        lines.append(" ".join(parts))

    return "\n".join(lines)

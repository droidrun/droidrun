"""Concise filtering - all logic self-contained."""

from typing import Dict, Any, Optional
from droidrun.tools.filters.base import TreeFilter


class ConciseFilter(TreeFilter):
    """Concise tree filtering (formerly DroidRun)."""

    def filter(
        self,
        a11y_tree: Dict[str, Any],
        device_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Filter using concise logic."""
        screen_bounds = device_context.get("screen_bounds", {})
        filtering_params = device_context.get("filtering_params", {})

        return self._filter_node(a11y_tree, screen_bounds, filtering_params)

    def _filter_node(
        self,
        node: Dict[str, Any],
        screen_bounds: Dict[str, int],
        filtering_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Recursively filter node."""
        min_size = filtering_params.get("min_element_size", 5)
        screen_width = screen_bounds.get("width", 1080)
        screen_height = screen_bounds.get("height", 2400)

        filtered_children = []
        for child in node.get("children", []):
            filtered_child = self._filter_node(child, screen_bounds, filtering_params)
            if filtered_child:
                filtered_children.append(filtered_child)

        if not self._in_bounds(node, screen_width, screen_height):
            if filtered_children:
                return {**node, "children": filtered_children}
            return None

        if not self._min_size(node, min_size):
            if filtered_children:
                return {**node, "children": filtered_children}
            return None

        if not self._interactive(node):
            if filtered_children:
                return {**node, "children": filtered_children}
            return None

        return {**node, "children": filtered_children}

    @staticmethod
    def _in_bounds(node: Dict[str, Any], width: int, height: int) -> bool:
        """Check if element is within screen bounds."""
        bounds = node.get("boundsInScreen", {})
        left = bounds.get("left", 0)
        top = bounds.get("top", 0)
        right = bounds.get("right", 0)
        bottom = bounds.get("bottom", 0)

        return (left >= 0 and top >= 0 and
                right <= width and bottom <= height)

    @staticmethod
    def _min_size(node: Dict[str, Any], min_size: int) -> bool:
        """Check if element meets minimum size."""
        bounds = node.get("boundsInScreen", {})
        width = bounds.get("right", 0) - bounds.get("left", 0)
        height = bounds.get("bottom", 0) - bounds.get("top", 0)

        return width > min_size and height > min_size

    @staticmethod
    def _interactive(node: Dict[str, Any]) -> bool:
        """Check if element is interactive."""
        return True

    def get_name(self) -> str:
        """Return filter name."""
        return "concise"

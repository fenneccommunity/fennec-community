"""List Output Parser"""

import re
from typing import Callable, List, Optional

from .base_parser import BaseOutputParser, ParsingError


class ListOutputParser(BaseOutputParser):
    """Parse list output from text.

    Extracts numbered or bulleted lists from text, returning a clean
    list of strings.

    Supported formats:
        - Numbered lists : 1. Item  /  2) Item  /  (3) Item
        - Lettered lists : a. Item  /  b) Item
        - Bullet lists   : * Item  /  - Item  /  • Item  /  → Item
        - Comma-separated (inline) : "item1, item2, item3"  (fallback)

    Example:
        parser = ListOutputParser()
        text = '''
        Here are the steps:
        1. Install dependencies
        2. Configure settings
        3. Run the application
        '''
        result = parser.parse(text)
        # Returns: ['Install dependencies', 'Configure settings', 'Run the application']
    """

    # Compiled patterns shared across instances
    _NUMBERED_RE = re.compile(
        r"^\s*[\(\[]?\s*(?:\d+|[a-zA-Z])\s*[\.\)\]]\s+(.+)$"
    )
    _BULLET_RE = re.compile(
        r"^\s*[*\-–—•◦▪▸▷→›»]\s+(.+)$"
    )
    # Lines that are clearly intro/outro sentences, not list items
    _SKIP_RE = re.compile(
        r"^(here\s+(is|are)|the\s+(following|items?|list)|summary[:\s]|"
        r"result[s]?[:\s]|output[:\s]|note[:\s]|example[:\s])",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_items: int = 0,
        max_items: Optional[int] = None,
        item_transform: Optional[Callable[[str], str]] = None,
    ):
        """Initialize list parser.

        Args:
            min_items: Minimum required items (default: 0)
            max_items: Maximum allowed items (default: None for unlimited)
            item_transform: Optional callable applied to each item after extraction
                            (e.g. str.lower, str.strip)
        """
        if min_items < 0:
            raise ValueError("min_items must be >= 0")
        if max_items is not None and max_items < min_items:
            raise ValueError("max_items must be >= min_items")

        self.min_items = min_items
        self.max_items = max_items
        self.item_transform = item_transform

    def parse(self, text: str) -> List[str]:
        """Parse list from text.

        Args:
            text: Raw text containing a list

        Returns:
            List of extracted item strings

        Raises:
            ParsingError: If no valid items found or count constraints violated
        """
        if not text or not text.strip():
            raise ParsingError("Cannot parse empty text", original_text=text)

        items = self._extract_list_items(text)

        if not items:
            raise ParsingError(
                "No list items found. Expected a numbered or bulleted list.",
                original_text=text,
            )

        if self.item_transform:
            items = [self.item_transform(i) for i in items]

        if len(items) < self.min_items:
            raise ParsingError(
                f"Found {len(items)} item(s), but minimum required is {self.min_items}.",
                original_text=text,
            )

        if self.max_items is not None and len(items) > self.max_items:
            raise ParsingError(
                f"Found {len(items)} item(s), but maximum allowed is {self.max_items}.",
                original_text=text,
            )

        return items

    def get_format_instructions(self) -> str:
        """Get list format instructions."""
        instructions = "Return the result as a numbered list, one item per line.\n"

        if self.min_items > 0:
            instructions += f"Include at least {self.min_items} items.\n"
        if self.max_items is not None:
            instructions += f"Include at most {self.max_items} items.\n"

        instructions += (
            "\nExample format:\n"
            "1. First item\n"
            "2. Second item\n"
            "3. Third item"
        )
        return instructions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items using a two-pass strategy.

        Pass 1: Collect explicitly marked items (numbered / bulleted).
        Pass 2: If nothing found, attempt comma-split fallback on a single line.
        """
        items: List[str] = []
        lines = text.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            item = self._match_list_item(stripped)
            if item:
                items.append(item)

        if items:
            return items

        # Fallback: comma-separated inline list on a single non-header line
        for line in lines:
            stripped = line.strip()
            if not stripped or self._SKIP_RE.match(stripped):
                continue
            if "," in stripped and len(stripped) < 500:
                parts = [p.strip() for p in stripped.split(",") if p.strip()]
                if len(parts) >= 2:
                    return parts

        return []

    def _match_list_item(self, line: str) -> Optional[str]:
        """Return the content of a list item line, or None if not a list item."""
        # Numbered / lettered
        m = self._NUMBERED_RE.match(line)
        if m:
            return m.group(1).strip()

        # Bulleted
        m = self._BULLET_RE.match(line)
        if m:
            return m.group(1).strip()

        return None

    def __repr__(self) -> str:
        return (
            f"ListOutputParser(min_items={self.min_items}, "
            f"max_items={self.max_items})"
        )

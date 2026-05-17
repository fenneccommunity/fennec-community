"""Base Output Parser Interface"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class BaseOutputParser(ABC):
    """Abstract base class for all output parsers.

    Output parsers convert raw LLM text outputs into structured formats.

    All parsers must implement:
        - parse(): Convert raw text to structured output
        - get_format_instructions(): Provide formatting instructions for the LLM
    """

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse raw text into structured output.

        Args:
            text: Raw text output from language model

        Returns:
            Structured data in desired format

        Raises:
            ParsingError: If parsing fails
        """
        pass

    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get formatting instructions for the language model.

        Returns:
            String containing format instructions
        """
        pass

    async def async_parse(self, text: str) -> Any:
        """Async version of parse().

        Runs parse() in a thread pool to avoid blocking the event loop.

        Args:
            text: Raw text to parse

        Returns:
            Structured data in desired format
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.parse, text)

    def parse_with_prompt(self, text: str) -> tuple[Any, str]:
        """Parse text and return both result and format instructions.

        Args:
            text: Raw text to parse

        Returns:
            Tuple of (parsed_result, format_instructions)
        """
        return self.parse(text), self.get_format_instructions()

    def parse_or_default(self, text: str, default: Any = None) -> Any:
        """Parse text, returning a default value if parsing fails.

        Args:
            text: Raw text to parse
            default: Value to return on failure (default: None)

        Returns:
            Parsed result or default value
        """
        try:
            return self.parse(text)
        except (ParsingError, Exception) as e:
            logger.warning("Parsing failed, returning default. Error: %s", e)
            return default

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ParsingError(Exception):
    """Exception raised when parsing fails."""

    def __init__(
        self,
        message: str,
        original_text: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """Initialize parsing error.

        Args:
            message: Error description
            original_text: The text that failed to parse
            cause: Original exception that caused the error
        """
        super().__init__(message)
        self.original_text = original_text
        self.cause = cause

    def __str__(self) -> str:
        msg = super().__str__()
        if self.cause:
            msg += f"\nCaused by: {self.cause}"
        if self.original_text:
            preview = (
                self.original_text[:200] + "..."
                if len(self.original_text) > 200
                else self.original_text
            )
            msg += f"\nOriginal text preview: {preview}"
        return msg


class AutoFixingParser(BaseOutputParser):
    """Wraps another parser with automatic retry and fix logic.

    On parse failure, applies a fixer function (e.g. another LLM call)
    to clean up the text and retries parsing.

    Example:
        def simple_fixer(bad_text: str, instructions: str) -> str:
            # Call your LLM here to fix the output
            return cleaned_text

        base_parser = JSONOutputParser()
        parser = AutoFixingParser(base_parser, fixer=simple_fixer, max_retries=2)
        result = parser.parse(raw_llm_output)
    """

    def __init__(
        self,
        parser: BaseOutputParser,
        fixer: Optional[Callable[[str, str], str]] = None,
        max_retries: int = 2,
    ):
        """Initialize auto-fixing parser.

        Args:
            parser: Underlying parser to use
            fixer: Callable(bad_text, format_instructions) -> fixed_text
            max_retries: Maximum retry attempts (default: 2)
        """
        self.parser = parser
        self.fixer = fixer
        self.max_retries = max_retries

    def parse(self, text: str) -> Any:
        """Parse with automatic retry and fixing.

        Args:
            text: Raw text to parse

        Returns:
            Parsed result

        Raises:
            ParsingError: If all retries fail
        """
        last_error: Optional[Exception] = None
        current_text = text

        for attempt in range(self.max_retries + 1):
            try:
                return self.parser.parse(current_text)
            except (ParsingError, Exception) as e:
                last_error = e
                if attempt < self.max_retries and self.fixer:
                    logger.info(
                        "Parse attempt %d failed. Applying fixer...", attempt + 1
                    )
                    try:
                        current_text = self.fixer(
                            current_text, self.parser.get_format_instructions()
                        )
                    except Exception as fix_err:
                        logger.warning("Fixer failed: %s", fix_err)
                        break

        raise ParsingError(
            f"Failed after {self.max_retries + 1} attempts.",
            original_text=text,
            cause=last_error,
        )

    def get_format_instructions(self) -> str:
        return self.parser.get_format_instructions()

    def __repr__(self) -> str:
        return f"AutoFixingParser(parser={self.parser!r}, max_retries={self.max_retries})"


class ChainedParser(BaseOutputParser):
    """Tries multiple parsers in order, returning the first successful result.

    Useful when the LLM might return output in one of several formats.

    Example:
        parser = ChainedParser([JSONOutputParser(), ListOutputParser()])
        result = parser.parse(text)  # tries JSON first, then list
    """

    def __init__(self, parsers: list[BaseOutputParser]):
        """Initialize chained parser.

        Args:
            parsers: List of parsers to try in order
        """
        if not parsers:
            raise ValueError("parsers list cannot be empty")
        self.parsers = parsers

    def parse(self, text: str) -> Any:
        """Try each parser and return first successful result.

        Args:
            text: Raw text to parse

        Returns:
            First successfully parsed result

        Raises:
            ParsingError: If all parsers fail
        """
        errors: list[str] = []
        for parser in self.parsers:
            try:
                return parser.parse(text)
            except (ParsingError, Exception) as e:
                errors.append(f"{parser.__class__.__name__}: {e}")

        raise ParsingError(
            "All parsers in chain failed:\n" + "\n".join(errors),
            original_text=text,
        )

    def get_format_instructions(self) -> str:
        """Return combined instructions from all parsers."""
        parts = [
            f"[Option {i+1}] {p.get_format_instructions()}"
            for i, p in enumerate(self.parsers)
        ]
        return "Accept any of the following formats:\n\n" + "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"ChainedParser(parsers={self.parsers!r})"

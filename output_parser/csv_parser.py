"""CSV Output Parser"""

import csv
import io
import re
from typing import Any, Dict, List, Optional
from .base_parser import BaseOutputParser, ParsingError


class CSVOutputParser(BaseOutputParser):
    """Parse CSV-formatted output from LLM text.

    Extracts a CSV table from raw text, returning a list of dicts (one
    per data row) with keys taken from the header row.

    Features:
        - Auto-detects common delimiters (comma, semicolon, tab, pipe)
        - Extracts CSV from markdown code blocks
        - Optional column validation
        - Type coercion per column

    Example:
        parser = CSVOutputParser(
            required_columns=["name", "age", "city"],
            column_types={"age": int},
        )

        text = '''
        ```csv
        name,age,city
        Alice,30,Paris
        Bob,25,London
        ```
        '''

        rows = parser.parse(text)
        # [{'name': 'Alice', 'age': 30, 'city': 'Paris'},
        #  {'name': 'Bob',   'age': 25, 'city': 'London'}]
    """

    _DELIMITERS = [",", ";", "\t", "|"]

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, type]] = None,
        delimiter: Optional[str] = None,
    ):
        """Initialize CSV parser.

        Args:
            required_columns: Column names that must be present in the header
            column_types: Mapping of column name → Python type for coercion
                          (e.g. {"age": int, "score": float})
            delimiter: Force a specific delimiter. If None, auto-detect.
        """
        self.required_columns = [c.lower() for c in (required_columns or [])]
        self.column_types = {k.lower(): v for k, v in (column_types or {}).items()}
        self.delimiter = delimiter

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Parse CSV from text.

        Args:
            text: Raw text containing CSV data

        Returns:
            List of dicts, one per data row

        Raises:
            ParsingError: If CSV cannot be extracted or required columns are missing
        """
        if not text or not text.strip():
            raise ParsingError("Cannot parse empty text", original_text=text)

        csv_text = self._extract_csv(text)
        delimiter = self.delimiter or self._detect_delimiter(csv_text)

        try:
            reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
            rows = list(reader)
        except csv.Error as e:
            raise ParsingError(f"CSV parsing error: {e}", original_text=csv_text, cause=e)

        if not rows:
            raise ParsingError("CSV contains no data rows.", original_text=csv_text)

        # Normalize header keys (strip whitespace, lowercase)
        rows = [
            {k.strip().lower(): v.strip() for k, v in row.items() if k}
            for row in rows
        ]

        # Validate required columns
        if self.required_columns:
            actual_cols = set(rows[0].keys())
            missing = set(self.required_columns) - actual_cols
            if missing:
                raise ParsingError(
                    f"Missing required columns: {sorted(missing)}",
                    original_text=csv_text,
                )

        # Type coercion
        if self.column_types:
            rows = [self._coerce_row(row) for row in rows]

        return rows

    def get_format_instructions(self) -> str:
        lines = ["Return the result as a CSV table with a header row."]

        if self.required_columns:
            header = ",".join(self.required_columns)
            lines += [
                f"The table must include these columns: {header}",
                "",
                "Example:",
                header,
                ",".join(f"value_{c}" for c in self.required_columns),
            ]
        else:
            lines += [
                "",
                "Example:",
                "column1,column2,column3",
                "value1,value2,value3",
            ]

        lines.append(
            "\nYou may wrap the CSV in a ```csv code block for clarity."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_csv(self, text: str) -> str:
        """Extract CSV content from text, removing markdown fences if present."""
        # markdown code block: ```csv ... ``` or ``` ... ```
        match = re.search(r"```(?:csv)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _detect_delimiter(self, text: str) -> str:
        """Detect the most likely delimiter by counting occurrences in the first line."""
        first_line = text.split("\n")[0]
        counts = {d: first_line.count(d) for d in self._DELIMITERS}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","

    def _coerce_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Apply column_types coercions to a single row dict."""
        result: Dict[str, Any] = {}
        for col, raw in row.items():
            if col in self.column_types:
                try:
                    result[col] = self.column_types[col](raw)
                except (ValueError, TypeError) as e:
                    raise ParsingError(
                        f"Cannot coerce column '{col}' value {raw!r} "
                        f"to {self.column_types[col].__name__}: {e}",
                        cause=e,
                    )
            else:
                result[col] = raw
        return result

    def __repr__(self) -> str:
        return (
            f"CSVOutputParser(required_columns={self.required_columns}, "
            f"delimiter={self.delimiter!r})"
        )

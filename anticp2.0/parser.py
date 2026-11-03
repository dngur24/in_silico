"""
Parser: extracts structured data from the raw HTML returned by AntiCP 2.0.

Target: https://webs.iiitd.edu.in/raghava/anticp2/predict.php
        POST action: multiple_test3.php

The result page is expected to contain a table with:
  - Peptide name (S.No. / Name / Sequence)
  - ML Score (0–1 float)
  - Prediction label ("Anti-Cancer Peptide" / "Non Anti-Cancer Peptide")
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup

from .models import PredictionResult


class Parser:
    """
    Parses the raw HTML result page from AntiCP 2.0.

    The server returns a results table with columns such as:
      S.No. | Sequence Name | Sequence | ML Score | Prediction

    Strategy
    --------
    1. Table-based parsing: locate the result ``<table>`` whose header row
       contains score / prediction / sequence columns.
    2. Regex fallback on visible text when table parsing yields nothing.
    """

    # ── Column header keyword sets ─────────────────────────────────────────────
    _NAME_KEYWORDS   = ("name", "s.no", "no.", "id", "peptide")
    _SEQ_KEYWORDS    = ("sequence", "seq")
    _SCORE_KEYWORDS  = ("score", "ml score", "svm score", "predicted score")
    _LABEL_KEYWORDS  = ("prediction", "predicted", "result", "class")

    # ── Regex fallbacks ────────────────────────────────────────────────────────
    _SCORE_RE = re.compile(
        r"(?:score|ml\s*score|svm\s*score)\s*[:\-]?\s*(0?\.\d+|1\.0|0|1)\b",
        re.IGNORECASE,
    )
    _LABEL_RE = re.compile(
        r"\b(Anti[- ]?Cancer\s*Peptide|Non\s*Anti[- ]?Cancer\s*Peptide)\b",
        re.IGNORECASE,
    )
    _SERVER_ERROR_RE = re.compile(
        r"(The\s+length\s+of\s+sequence[^<\n]+?(?:greater\s+than\s+or\s+equal\s+to\s+\d+\.|less\s+than\s+\d+\.)"
        r"|Please\s+enter\s+valid[^<\n.]+?\.)",
        re.IGNORECASE,
    )

    # ─────────────────────────────────────────────────────────────────────────

    def parse(
        self,
        html: str,
        peptide_name: str,
        peptide: str,
        method: str,
        threshold: str,
    ) -> list[PredictionResult]:
        """
        Extract prediction results from the AntiCP 2.0 result HTML.

        The server may return results for multiple sequences at once
        (batch FASTA input). Each row in the result table maps to one
        ``PredictionResult``.

        Parameters
        ----------
        html : str
            Raw HTML of the result page returned by ``multiple_test3.php``.
        peptide_name : str
            FASTA sequence ID used as fallback when the table lacks a name column.
        peptide : str
            Amino-acid sequence used as fallback.
        method : str
            Prediction method label (stored in the result object).
        threshold : str
            Score threshold label (stored in the result object).

        Returns
        -------
        list[PredictionResult]
            One entry per result row found.  Falls back to a single regex-derived
            result if no table is found.
        """
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        server_error = self._extract_server_error(text)

        # ── 1. Table-based parsing ────────────────────────────────────────────
        results = self._parse_table(
            soup,
            method,
            threshold,
            fallback_name=peptide_name,
            fallback_seq=peptide,
            server_error=server_error,
        )
        if results:
            return results

        # ── 2. Regex fallback on visible text ────────────────────────────────
        score = self._regex_score(text)
        label = self._regex_label(text)
        error = None
        if score is None and label is None:
            error = (
                server_error
                or "AntiCP 2.0 did not return a valid prediction for this sequence and model."
            )

        return [
            PredictionResult(
                peptide_name=peptide_name,
                peptide=peptide,
                method=method,
                threshold=threshold,
                score=score,
                label=label,
                raw_html=html,
                error=error,
            )
        ]

    # ------------------------------------------------------------------
    # Table parsing
    # ------------------------------------------------------------------

    def _parse_table(
        self,
        soup: BeautifulSoup,
        method: str,
        threshold: str,
        *,
        fallback_name: str = "",
        fallback_seq: str = "",
        server_error: Optional[str] = None,
    ) -> list[PredictionResult]:
        """
        Locate the result table and extract one PredictionResult per data row.

        Returns an empty list when no suitable table is found.
        """
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            header_idx, cols = self._find_header(rows)
            if header_idx is None:
                continue

            data_rows = rows[header_idx + 1:]
            single_row = len(data_rows) == 1
            results: list[PredictionResult] = []
            for row in data_rows:
                cells = [
                    td.get_text(strip=True)
                    for td in row.find_all(["td", "th"])
                ]
                if not cells:
                    continue

                name  = self._get_cell(cells, cols.get("name"))
                seq   = self._get_cell(cells, cols.get("seq"))
                score = self._safe_float(self._get_cell(cells, cols.get("score")))
                label = self._normalize_label(
                    self._get_cell(cells, cols.get("label")) or ""
                )
                row_error = None
                if not seq or score is None or label is None:
                    score = None
                    label = None
                    row_error = (
                        server_error
                        or f"Upstream AntiCP 2.0 server returned an empty prediction for method '{method}'."
                    )

                results.append(
                    PredictionResult(
                        peptide_name=name or (fallback_name if single_row else ""),
                        peptide=seq or (fallback_seq if single_row else ""),
                        method=method,
                        threshold=threshold,
                        score=score,
                        label=label or None,
                        error=row_error,
                    )
                )

            if results:
                return results

        return []

    def _find_header(
        self, rows: list
    ) -> tuple[Optional[int], dict[str, int]]:
        """
        Search rows for a header that contains score / label columns.

        Returns
        -------
        (header_row_idx, col_map)
            ``col_map`` maps logical column names to their 0-based index.
            Returns ``(None, {})`` when no header is found.
        """
        for i, row in enumerate(rows):
            cells_lower = [
                td.get_text(strip=True).lower()
                for td in row.find_all(["td", "th"])
            ]

            has_score = any(
                any(kw in c for kw in self._SCORE_KEYWORDS) for c in cells_lower
            )
            has_label = any(
                any(kw in c for kw in self._LABEL_KEYWORDS) for c in cells_lower
            )

            if not (has_score or has_label):
                continue

            col_map: dict[str, int] = {}

            def _find_col(keywords: tuple[str, ...]) -> Optional[int]:
                return next(
                    (j for j, c in enumerate(cells_lower)
                     if any(kw in c for kw in keywords)),
                    None,
                )

            if (idx := _find_col(self._NAME_KEYWORDS))  is not None:
                col_map["name"]  = idx
            if (idx := _find_col(self._SEQ_KEYWORDS))   is not None:
                col_map["seq"]   = idx
            if (idx := _find_col(self._SCORE_KEYWORDS)) is not None:
                col_map["score"] = idx
            if (idx := _find_col(self._LABEL_KEYWORDS)) is not None:
                col_map["label"] = idx

            return i, col_map

        return None, {}

    # ------------------------------------------------------------------
    # Regex helpers
    # ------------------------------------------------------------------

    def _regex_score(self, text: str) -> Optional[float]:
        """Try to extract a numeric score from plain text."""
        m = self._SCORE_RE.search(text)
        return self._safe_float(m.group(1)) if m else None

    def _regex_label(self, text: str) -> Optional[str]:
        """Try to extract a classification label from plain text."""
        m = self._LABEL_RE.search(text)
        return self._normalize_label(m.group(0)) if m else None

    def _extract_server_error(self, text: str) -> Optional[str]:
        """Extract upstream validation/error notice (e.g. minimum length warning)."""
        m = self._SERVER_ERROR_RE.search(text)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cell(cells: list[str], idx: Optional[int]) -> Optional[str]:
        """Safely retrieve a cell by index, or None."""
        if idx is None or idx >= len(cells):
            return None
        return cells[idx] or None

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        try:
            return float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_label(label: str) -> Optional[str]:
        """Standardize label text from the server."""
        if not label:
            return None
        lower = label.lower()
        if "anti" not in lower and "cancer" not in lower:
            return None
        if "non" in lower:
            return "Non Anti-Cancer Peptide"
        return "Anti-Cancer Peptide"

"""
HTML parser for HemoPI 2.0 result pages (`disp_r1.php` and `disp_c1.php`).

Supported result table (`<table id="example">`) schemas:
  - Regression (`RFR based`):
      SeqID | Sequence | HC50(μM) | Prediction
  - Classification (`ESM2-t6 based`):
      SeqID | Sequence | ESM Score | Prediction
  - Classification (`Hybrid1 (ESM2-t6 + MERCI)`):
      SeqID | Sequence | ESM Score | MERCI Score | Hybrid Score | Prediction
  - Classification (`RF based`):
      SeqID | Sequence | ML Score | Prediction
  - Classification (`Hybrid2 (RF + MERCI)`):
      SeqID | Sequence | ML Score | MERCI Score | Hybrid Score | Prediction
"""

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag


class Parser:
    """Parses HemoPI 2.0 HTML result pages mapped by FASTA SeqID."""

    _JOB_ID_RE = re.compile(r"Job\s*ID\s*:\s*(\d+)", re.IGNORECASE)
    _COLOR_RE = re.compile(r"color\s*:\s*([a-zA-Z#0-9]+)", re.IGNORECASE)
    _FONT_SIZE_RE = re.compile(r"font-size\s*:\s*(\d+(?:\.\d+)?)\s*px", re.IGNORECASE)
    _PRIMARY_SCORE_PRIORITY = (
        "hc50(μm)",
        "hc50(um)",
        "hc50",
        "hybrid score",
        "esm score",
        "ml score",
        "score",
    )

    def extract_job_id(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = self._JOB_ID_RE.search(text)
        return match.group(1) if match else None

    def parse_batch(self, html: str) -> dict[str, dict[str, Any]]:
        """
        Parse the result table (`#example` or matching prediction table) and return
        a mapping ``{seq_id: {"sequence": ..., "score": ..., "label": ..., "properties": ...}}``.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = self._find_result_table(soup)
        if table is None:
            return {}

        headers, data_rows = self._extract_table_rows(table)
        if not headers or not data_rows:
            return {}

        lower_headers = [self._norm_header(h) for h in headers]
        seqid_col = self._find_col(lower_headers, ("seqid", "seq id", "id", "subject"))
        seq_col = self._find_col(lower_headers, ("sequence", "seq"))
        pred_col = self._find_col(lower_headers, ("prediction", "predicted", "result"))
        score_col = self._find_primary_score_col(lower_headers)

        if seqid_col is None:
            return {}

        parsed: dict[str, dict[str, Any]] = {}
        for cells in data_rows:
            if seqid_col >= len(cells):
                continue
            seq_id = cells[seqid_col].strip()
            if not seq_id or self._norm_header(seq_id) in ("seqid", "seq id"):
                continue

            sequence = cells[seq_col].strip().upper() if seq_col is not None and seq_col < len(cells) else ""
            score = (
                self._safe_float(cells[score_col])
                if score_col is not None and score_col < len(cells)
                else None
            )
            raw_label = cells[pred_col].strip() if pred_col is not None and pred_col < len(cells) else ""
            label = self._normalize_label(raw_label)

            props: dict[str, Any] = {}
            for idx, header_name in enumerate(headers):
                if idx in (seqid_col, seq_col, pred_col) or idx >= len(cells):
                    continue
                val_str = cells[idx].strip()
                if not val_str:
                    continue
                num_val = self._safe_float(val_str)
                props[header_name] = num_val if num_val is not None else val_str

            parsed[seq_id] = {
                "sequence": sequence,
                "score": score,
                "label": label,
                "properties": props or None,
            }

        return parsed

    def parse_visual_presentation(self, html: str) -> dict[str, list[dict[str, Any]]]:
        """
        Parse the `#seq_table` Visual Presentation table from HemoPI 2.0 HTML.

        Returns ``{seq_id: [{"text": "I", "color": "black", "size": 18}, ...]}``
        where color is one of ``"red"`` (increases hemolysis), ``"green"``
        (decreases hemolysis), or ``"black"`` (weak correlation), and size is
        the font size in pixels indicating correlation strength.
        """
        soup = BeautifulSoup(html, "html.parser")
        seq_table = soup.find("table", id="seq_table")
        if seq_table is None:
            return {}

        visuals: dict[str, list[dict[str, Any]]] = {}
        for tr in seq_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            seq_id = tds[0].get_text(" ", strip=True)
            if not seq_id or self._norm_header(seq_id) in ("seqid", "seq id"):
                continue

            spans = tds[1].find_all("span")
            segments: list[dict[str, Any]] = []
            if spans:
                for sp in spans:
                    txt = sp.get_text()
                    if not txt:
                        continue
                    style = sp.get("style", "") or ""
                    c_match = self._COLOR_RE.search(style)
                    s_match = self._FONT_SIZE_RE.search(style)
                    raw_color = c_match.group(1).strip().lower() if c_match else "black"
                    color = raw_color if raw_color in ("red", "green", "black") else "black"
                    size = int(round(float(s_match.group(1)))) if s_match else 18
                    segments.append({"text": txt, "color": color, "size": size})
            else:
                plain = tds[1].get_text(strip=True)
                if plain:
                    segments.append({"text": plain, "color": "black", "size": 18})

            if segments:
                visuals[seq_id] = segments

        return visuals

    def extract_server_error(self, html: str) -> Optional[str]:
        """Detect explicit error messages returned by HemoPI 2.0 when no table exists."""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        for pattern in (
            r"(Number of sequences[^.\n]+exceeds[^.\n]+\.)",
            r"(Please enter[^.\n]+\.)",
            r"(Your Sequence contains[^.\n]+\.)",
            r"(Error\s*:\s*[^.\n]+\.?)",
        ):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_result_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        # Primary target: HemoPI 2.0 renders results inside <table id="example">
        table = soup.find("table", id="example")
        if table is not None:
            return table

        # Fallback: locate any table whose header row contains both SeqID and Prediction
        for tbl in soup.find_all("table"):
            if tbl.get("id") == "seq_table":
                continue
            headers, _ = self._extract_table_rows(tbl)
            lower = [self._norm_header(h) for h in headers]
            if ("seqid" in lower or "seq id" in lower) and "prediction" in lower:
                return tbl
        return None

    @staticmethod
    def _extract_table_rows(table: Tag) -> tuple[list[str], list[list[str]]]:
        thead = table.find("thead")
        tbody = table.find("tbody")

        if thead is not None:
            header_tr = thead.find("tr")
            headers = (
                [c.get_text(" ", strip=True) for c in header_tr.find_all(["th", "td"])]
                if header_tr is not None
                else []
            )
            row_tags = tbody.find_all("tr") if tbody is not None else table.find_all("tr")[1:]
            data_rows = [
                [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                for tr in row_tags
                if tr.find_all("td")
            ]
            return headers, data_rows

        all_rows = table.find_all("tr")
        if len(all_rows) < 2:
            return [], []
        headers = [c.get_text(" ", strip=True) for c in all_rows[0].find_all(["th", "td"])]
        data_rows = [
            [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            for tr in all_rows[1:]
            if tr.find_all(["td", "th"])
        ]
        return headers, data_rows

    @staticmethod
    def _norm_header(header: str) -> str:
        return " ".join(header.strip().lower().split())

    @staticmethod
    def _find_col(lower_headers: list[str], candidates: tuple[str, ...]) -> Optional[int]:
        for cand in candidates:
            if cand in lower_headers:
                return lower_headers.index(cand)
        return None

    def _find_primary_score_col(self, lower_headers: list[str]) -> Optional[int]:
        for key in self._PRIMARY_SCORE_PRIORITY:
            for idx, hdr in enumerate(lower_headers):
                if hdr == key or key in hdr:
                    return idx
        return None

    @staticmethod
    def _normalize_label(raw: str) -> Optional[str]:
        if not raw:
            return None
        low = raw.strip().lower()
        if "non" in low and "hemolytic" in low:
            return "Non-Hemolytic"
        if "hemolytic" in low:
            return "Hemolytic"
        return raw.strip()

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val.strip())
        except (TypeError, ValueError):
            return None

"""
HTML parser for ToxinPred 3.0 result pages (`disp1.php?ran=XXXXX`).

Supported result table (`<table id="example">`) schemas:
  - ET based (`terminus="0"`):
      Subject | Sequence | ML Score | Prediction | PPV
  - Hybrid(ET+MERCI) (`terminus="1"`):
      Subject | Sequence | ML Score | MERCI Score (+ve) | MERCI Score (-ve) | Hybrid Score | Prediction | PPV
  - DL based (`terminus="2"`):
      Subject | Sequence | DL Score | Prediction | PPV
  - Hybrid(DL+MERCI) (`terminus="3"`):
      Subject | Sequence | ML Score | MERCI Score (+ve) | MERCI Score (-ve) | Hybrid Score | Prediction | PPV
"""

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup, Tag


class Parser:
    """Parses ToxinPred 3.0 HTML result tables."""

    _JOB_ID_RE = re.compile(r"Job\s*ID\s*:\s*(\d+)", re.IGNORECASE)
    _PRIMARY_SCORE_PRIORITY = (
        "hybrid score",
        "dl score",
        "ml score",
        "score",
    )

    def extract_job_id(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        match = self._JOB_ID_RE.search(text)
        return match.group(1) if match else None

    def parse_rows(self, html: str) -> list[dict[str, Any]]:
        """
        Parse `<table id="example">` in row order and return a list of dicts:
        ``[{"subject": ..., "sequence": ..., "score": ..., "label": ..., "properties": ...}, ...]``.
        """
        soup = BeautifulSoup(html, "html.parser")
        table = self._find_result_table(soup)
        if table is None:
            return []

        headers, data_rows = self._extract_table_rows(table)
        if not headers or not data_rows:
            return []

        lower_headers = [self._norm_header(h) for h in headers]
        subj_col = self._find_col(lower_headers, ("subject", "seqid", "seq id", "id"))
        seq_col = self._find_col(lower_headers, ("sequence", "seq"))
        pred_col = self._find_col(lower_headers, ("prediction", "predicted", "result"))
        score_col = self._find_primary_score_col(lower_headers)

        parsed_rows: list[dict[str, Any]] = []
        for cells in data_rows:
            if subj_col is not None and subj_col < len(cells):
                subject = cells[subj_col].strip()
                if self._norm_header(subject) in ("subject", "seqid", "seq id"):
                    continue
            else:
                subject = ""

            sequence = (
                cells[seq_col].strip().upper()
                if seq_col is not None and seq_col < len(cells)
                else ""
            )
            score = (
                self._safe_float(cells[score_col])
                if score_col is not None and score_col < len(cells)
                else None
            )
            raw_label = (
                cells[pred_col].strip()
                if pred_col is not None and pred_col < len(cells)
                else ""
            )
            label = self._normalize_label(raw_label)

            props: dict[str, Any] = {}
            for idx, header_name in enumerate(headers):
                if idx in (subj_col, seq_col, pred_col) or idx >= len(cells):
                    continue
                val_str = cells[idx].strip()
                if not val_str:
                    continue
                num_val = self._safe_float(val_str)
                props[header_name] = num_val if num_val is not None else val_str

            parsed_rows.append(
                {
                    "subject": subject,
                    "sequence": sequence,
                    "score": score,
                    "label": label,
                    "properties": props or None,
                }
            )

        return parsed_rows

    def parse_batch(self, html: str) -> dict[str, dict[str, Any]]:
        """Return rows keyed by Subject ID (e.g. 'Seq_1')."""
        return {
            row["subject"]: row
            for row in self.parse_rows(html)
            if row.get("subject")
        }

    def extract_server_error(self, html: str) -> Optional[str]:
        """Detect explicit error messages returned by ToxinPred 3.0 when no table exists."""
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        for pattern in (
            r"(The number of sequences exceeds[^.\n]+\.)",
            r"(exceeds the maximum length[^.\n]+\.)",
            r"(Please Enter[^.\n!]+[!.]?)",
            r"(Your Sequence contains[^.\n]+\.?)",
            r"(Error\s*:\s*[^.\n]+\.?)",
        ):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_result_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        table = soup.find("table", id="example")
        if table is not None:
            return table

        for tbl in soup.find_all("table"):
            headers, _ = self._extract_table_rows(tbl)
            lower = [self._norm_header(h) for h in headers]
            if ("subject" in lower or "seqid" in lower) and "prediction" in lower:
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
        if "non" in low and "toxin" in low:
            return "Non-Toxin"
        if "toxin" in low:
            return "Toxin"
        return raw.strip()

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val.strip())
        except (TypeError, ValueError):
            return None

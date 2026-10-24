"""
HTML parser for SEP-AlgPro job_search result pages.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from bs4 import BeautifulSoup

_JOB_ID_RE = re.compile(r"jobId=([A-Za-z0-9_-]{16,64})|/download/([A-Za-z0-9_-]{16,64})/")


class Parser:
    """Extracts job status, job ID, and per-sequence prediction rows from SEP-AlgPro HTML."""

    def parse_job_page(self, html: str) -> tuple[Optional[str], Optional[str], dict[str, dict[str, Any]]]:
        soup = BeautifulSoup(html, "html.parser")
        job_id: Optional[str] = None
        status: Optional[str] = None
        predictions: dict[str, dict[str, Any]] = {}

        m = _JOB_ID_RE.search(html)
        if m:
            job_id = m.group(1) or m.group(2)

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            headers = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
            if "job id" in headers and "status" in headers:
                jid_col = headers.index("job id")
                st_col = headers.index("status")
                data_cells = [c.get_text(" ", strip=True) for c in rows[1].find_all(["td", "th"])]
                if len(data_cells) > max(jid_col, st_col):
                    job_id = data_cells[jid_col].strip() or job_id
                    status = data_cells[st_col].strip().upper()
                continue

            if "name" in headers and "class" in headers:
                name_col = headers.index("name")
                cls_col = headers.index("class")
                prob_col = headers.index("probability") if "probability" in headers else None

                for tr in rows[1:]:
                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                    if len(cells) <= max(name_col, cls_col):
                        continue
                    name = cells[name_col].strip()
                    if not name:
                        continue
                    label = cells[cls_col].strip() or None
                    score = (
                        self._safe_float(cells[prob_col])
                        if prob_col is not None and prob_col < len(cells)
                        else None
                    )
                    predictions[name] = {"label": label, "score": score}

        return job_id, status, predictions

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

"""
HTML parser for AlgPred 2.0 batch_action.php output tables.
"""

from __future__ import annotations

from typing import Any, Optional

from bs4 import BeautifulSoup


class Parser:
    """Parses AlgPred 2.0 tabular HTML output mapped by FASTA subject ID."""

    def parse_batch(self, html: str) -> dict[str, dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        parsed: dict[str, dict[str, Any]] = {}

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            header_idx = None
            headers: list[str] = []
            for idx, tr in enumerate(rows):
                cols = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                lower = [c.lower() for c in cols]
                if "subject" in lower and "prediction" in lower:
                    header_idx = idx
                    headers = cols
                    break

            if header_idx is None:
                continue

            lower_headers = [h.lower() for h in headers]
            subj_col = lower_headers.index("subject")
            pred_col = lower_headers.index("prediction")
            hybrid_col = lower_headers.index("hybrid score") if "hybrid score" in lower_headers else None
            ml_col = lower_headers.index("ml score") if "ml score" in lower_headers else None

            for tr in rows[header_idx + 1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) <= max(subj_col, pred_col):
                    continue
                subj = cells[subj_col].strip()
                if not subj or subj.lower() == "subject":
                    continue

                score_col = hybrid_col if hybrid_col is not None else ml_col
                score = self._safe_float(cells[score_col]) if score_col is not None and score_col < len(cells) else None
                label = cells[pred_col].strip() or None

                props: dict[str, Any] = {}
                for col_i, h_name in enumerate(headers):
                    if col_i in (subj_col, pred_col) or col_i >= len(cells):
                        continue
                    num_v = self._safe_float(cells[col_i])
                    props[h_name] = num_v if num_v is not None else cells[col_i]

                parsed[subj] = {
                    "score": score,
                    "label": label,
                    "properties": props or None,
                }

        return parsed

    @staticmethod
    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

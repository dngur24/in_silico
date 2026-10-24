"""
Predictor & parser for AntiCP 3.0 (https://webs.iiitd.edu.in/raghava/anticp3/predict.html).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.security import build_allowed_hosts, safe_request_text

logger = logging.getLogger(__name__)

_BASE_URL = "https://webs.iiitd.edu.in/raghava/anticp3"
_SUBMIT_URL = f"{_BASE_URL}/predict_action.php"
_MIN_LEN = 5

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{_BASE_URL}/predict.html",
    "Origin": "https://webs.iiitd.edu.in",
}


class Method(str, Enum):
    HYBRID_ESM2_BLAST = "1"  # Hybrid Model (Finetuned ESM2-t33 + BLAST)
    ESM2_ONLY = "2"          # Finetuned ESM2-t33

    @classmethod
    def resolve(cls, val: Optional[str]) -> "Method":
        if isinstance(val, cls):
            return val
        if not val:
            return cls.HYBRID_ESM2_BLAST
        norm = str(val).strip().lower()
        if norm in ("2", "esm2", "esm2_only", "finetuned esm2-t33"):
            return cls.ESM2_ONLY
        return cls.HYBRID_ESM2_BLAST


@dataclass
class PredictionResult:
    peptide: str
    method: str
    score: Optional[float] = None
    label: Optional[str] = None
    properties: Optional[dict[str, Any]] = None
    raw_html: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class Predictor:
    """Submits FASTA sequences to AntiCP 3.0 and parses #myTable results."""

    def __init__(
        self,
        method: Method | str = Method.HYBRID_ESM2_BLAST,
        threshold: float = 0.5,
        fields: Optional[list[str]] = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
    ) -> None:
        self.method = Method.resolve(method)
        self.threshold = f"{max(0.1, min(1.0, round(float(threshold if threshold is not None else 0.5), 1))):.1f}"
        self.fields = fields if fields is not None else ["4", "7", "9", "11", "13"]
        self.timeout = timeout
        self._allowed_hosts = build_allowed_hosts(_BASE_URL)
        self._session = self._build_session(max_retries, retry_backoff)

    @property
    def method_label(self) -> str:
        return "ESM2+BLAST" if self.method == Method.HYBRID_ESM2_BLAST else "ESM2"

    def predict(self, peptide: str) -> PredictionResult:
        return self.predict_batch([peptide])[0]

    def predict_batch(self, sequences: list[str]) -> list[PredictionResult]:
        cleaned = [self._clean(s) for s in sequences]
        results: list[Optional[PredictionResult]] = [None] * len(cleaned)
        valid_indices: list[int] = []

        for idx, seq in enumerate(cleaned):
            err = self._validate(seq)
            if err:
                results[idx] = PredictionResult(peptide=seq, method=self.method_label, error=err)
            else:
                valid_indices.append(idx)

        if not valid_indices:
            return [r for r in results if r is not None]

        fasta_lines = []
        id_map: dict[str, tuple[int, str]] = {}
        for pos, orig_idx in enumerate(valid_indices, start=1):
            tag = f"pep_{pos:04d}"
            seq = cleaned[orig_idx]
            fasta_lines.append(f">{tag}\n{seq}")
            id_map[tag] = (orig_idx, seq)

        try:
            html = safe_request_text(
                self._session,
                "POST",
                _SUBMIT_URL,
                allowed_hosts=self._allowed_hosts,
                timeout=self.timeout,
                data={
                    "seq": "\n".join(fasta_lines),
                    "method": self.method.value,
                    "thva1": self.threshold,
                    "field[]": self.fields,
                    "title": "",
                    "email": "",
                },
                files={"uploadedfile": ("", b"", "application/octet-stream")},
            )
            parsed = self._parse_table(html)
            if not parsed:
                upstream_err = "Upstream AntiCP 3.0 server returned an empty result table (server-side model unavailable)."
                for _tag, (orig_idx, seq) in id_map.items():
                    results[orig_idx] = PredictionResult(
                        peptide=seq,
                        method=self.method_label,
                        raw_html=html,
                        error=upstream_err,
                    )
            else:
                for tag, (orig_idx, seq) in id_map.items():
                    row = parsed.get(tag)
                    if not row:
                        results[orig_idx] = PredictionResult(
                            peptide=seq,
                            method=self.method_label,
                            raw_html=html,
                            error="Sequence missing from AntiCP 3.0 output table.",
                        )
                    else:
                        results[orig_idx] = PredictionResult(
                            peptide=seq,
                            method=self.method_label,
                            score=row["score"],
                            label=row["label"],
                            properties=row["properties"],
                            raw_html=html,
                        )
        except Exception as exc:  # noqa: BLE001
            for _tag, (orig_idx, seq) in id_map.items():
                results[orig_idx] = PredictionResult(
                    peptide=seq,
                    method=self.method_label,
                    error=str(exc),
                )

        return [r for r in results if r is not None]

    @staticmethod
    def _parse_table(html: str) -> dict[str, dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"id": "myTable"})
        if not table:
            return {}
        rows = table.find_all("tr")
        if len(rows) < 2:
            return {}
        headers = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        if not any(headers):
            return {}

        lower = [h.lower() for h in headers]
        id_col = 0
        pred_col = next((i for i, h in enumerate(lower) if "predict" in h or "class" in h), None)
        score_col = next((i for i, h in enumerate(lower) if "score" in h or "prob" in h), None)

        out: dict[str, dict[str, Any]] = {}
        for tr in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells or len(cells) < 2:
                continue
            subj = cells[id_col].strip().lstrip(">")
            score = None
            if score_col is not None and score_col < len(cells):
                try:
                    score = float(cells[score_col])
                except ValueError:
                    score = None
            label = cells[pred_col].strip() if pred_col is not None and pred_col < len(cells) else None
            props = {
                headers[i]: cells[i]
                for i in range(len(headers))
                if i not in {id_col, pred_col, score_col} and i < len(cells) and headers[i]
            }
            out[subj] = {"score": score, "label": label, "properties": props or None}
        return out

    @staticmethod
    def _clean(raw: str) -> str:
        lines = [
            ln.strip()
            for ln in raw.strip().splitlines()
            if ln.strip() and not ln.strip().startswith(">")
        ]
        return "".join(lines).upper()

    @staticmethod
    def _validate(seq: str) -> Optional[str]:
        if not seq:
            return "Peptide sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", seq):
            return f"Sequence contains invalid characters: '{seq}'."
        if len(seq) < _MIN_LEN:
            return f"Sequence length {len(seq)} is too short (minimum {_MIN_LEN} residues)."
        return None

    @staticmethod
    def _build_session(max_retries: int, backoff: float) -> requests.Session:
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

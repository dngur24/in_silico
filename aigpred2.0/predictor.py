"""
Predictor for AlgPred 2.0 (https://webs.iiitd.edu.in/raghava/algpred2/batch.html).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.security import build_allowed_hosts, safe_request_text

from .models import Method, PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)

_BASE_URL = "https://webs.iiitd.edu.in/raghava/algpred2"
_SUBMIT_URL = f"{_BASE_URL}/batch_action.php"
_MIN_LEN = 5
_CHUNK_SIZE = 50

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": f"{_BASE_URL}/batch.html",
    "Origin": "https://webs.iiitd.edu.in",
}


class Predictor:
    """Submits peptide sequences to AlgPred 2.0 and parses allergenicity predictions."""

    def __init__(
        self,
        method: Method | str = Method.HYBRID,
        threshold: float = 0.3,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.method = Method.resolve(method)
        self.threshold = self._snap_threshold(threshold)
        self.timeout = timeout
        self._allowed_hosts = build_allowed_hosts(_BASE_URL)
        self._parser = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    @property
    def method_label(self) -> str:
        return "Hybrid" if self.method == Method.HYBRID else "AAC_RF"

    def predict(self, peptide: str) -> PredictionResult:
        return self.predict_batch([peptide])[0]

    def predict_batch(self, sequences: list[str]) -> list[PredictionResult]:
        cleaned = [self._clean(s) for s in sequences]
        results: list[Optional[PredictionResult]] = [None] * len(cleaned)
        valid_indices: list[int] = []

        for idx, seq in enumerate(cleaned):
            err = self._validate(seq)
            if err:
                results[idx] = PredictionResult(
                    peptide=seq,
                    method=self.method_label,
                    error=err,
                )
            else:
                valid_indices.append(idx)

        for start in range(0, len(valid_indices), _CHUNK_SIZE):
            chunk_idxs = valid_indices[start:start + _CHUNK_SIZE]
            fasta_lines = []
            id_map: dict[str, tuple[int, str]] = {}
            for pos, orig_idx in enumerate(chunk_idxs, start=1):
                tag = f"pep_{pos:04d}"
                seq = cleaned[orig_idx]
                fasta_lines.append(f">{tag}\n{seq}")
                id_map[tag] = (orig_idx, seq)

            fasta_block = "\n".join(fasta_lines)
            try:
                html = self._submit(fasta_block)
                parsed_map = self._parser.parse_batch(html)
                for tag, (orig_idx, seq) in id_map.items():
                    row = parsed_map.get(tag)
                    if not row:
                        results[orig_idx] = PredictionResult(
                            peptide=seq,
                            method=self.method_label,
                            raw_html=html,
                            error="AlgPred 2.0 did not return a result row for this sequence.",
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
                logger.error("AlgPred 2.0 batch submission failed: %s", exc)
                for _tag, (orig_idx, seq) in id_map.items():
                    results[orig_idx] = PredictionResult(
                        peptide=seq,
                        method=self.method_label,
                        error=str(exc),
                    )

        return [r for r in results if r is not None]

    def _submit(self, fasta_block: str) -> str:
        data = {
            "name": "BatchJob",
            "seq": fasta_block,
            "terminus": self.method.value,
            "svm_th": self.threshold,
            "email": "",
            "submit": "Submit",
        }
        files = {
            "uploadedfile": ("", b"", "application/octet-stream"),
        }
        return safe_request_text(
            self._session,
            "POST",
            _SUBMIT_URL,
            allowed_hosts=self._allowed_hosts,
            timeout=self.timeout,
            data=data,
            files=files,
        )

    @staticmethod
    def _snap_threshold(th: float) -> str:
        val = max(-0.5, min(2.0, round(float(th if th is not None else 0.3), 1)))
        if val == 0:
            val = 0.0
        return f"{val:.1f}"

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

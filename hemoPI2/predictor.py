"""
Predictor: submits peptide sequences to the HemoPI 2.0 web server
and returns structured PredictionResult objects.

Target form page: https://webs.iiitd.edu.in/raghava/hemopi2/prediction.html
Form action     : https://webs.iiitd.edu.in/raghava/hemopi2/prediction_action.php
Result pages    : https://webs.iiitd.edu.in/raghava/hemopi2/disp_r1.php?ran=XXXXX  (Regression)
                  https://webs.iiitd.edu.in/raghava/hemopi2/disp_c1.php?ran=XXXXX  (Classification)

Form fields (enctype="multipart/form-data"):
  - selectedMode  : "regression" | "classification"
  - seq           : FASTA-formatted peptide sequences (max 1000 per submission)
  - MAX_FILE_SIZE : ""
  - uploadedfile  : optional FASTA file upload
  - mode          : "regression" | "classification"
  - terminus      : "0" (RFR based — Regression)
                    "1" (ESM2-t6 based — Classification)
                    "2" (Hybrid1 (ESM2-t6 + MERCI) — Classification)
                    "3" (RF based — Classification)
                    "4" (Hybrid2 (RF + MERCI) — Classification)
  - th            : threshold for Classification ("0.0" .. "1.0", incl. "0.45", "0.58")
  - email         : optional email (Regression)
  - email2        : optional email (Classification)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional, Sequence, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.security import (
    build_allowed_hosts,
    resolve_safe_redirect,
    safe_request_text,
)

from .models import (
    MLTechnique,
    PredictionMethod,
    PredictionResult,
    ThresholdValue,
)
from .parser import Parser

logger = logging.getLogger(__name__)

_BASE_URL = "https://webs.iiitd.edu.in/raghava/hemopi2"
_FORM_URL = f"{_BASE_URL}/prediction.html"
_SUBMIT_URL = f"{_BASE_URL}/prediction_action.php"

_CANONICAL_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_META_REFRESH_RE = re.compile(
    r"content\s*=\s*['\"]\s*\d+\s*;\s*url\s*=\s*([^'\"]+)['\"]",
    re.IGNORECASE,
)

_MIN_LEN = 3
_MAX_SERVER_BATCH = 1000
_DEFAULT_CHUNK_SIZE = 100

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": _FORM_URL,
    "Origin": "https://webs.iiitd.edu.in",
}


class Predictor:
    """
    Submits peptide sequences to HemoPI 2.0 via HTTP (`requests`) and returns
    structured :class:`PredictionResult` objects.

    Parameters
    ----------
    method : PredictionMethod | str
        Prediction mode ('Regression' or 'Classification'). Default: 'Regression'.
    technique : MLTechnique | str | None
        ML technique used for developing the model:
          - For Regression     : 'RFR based' (default)
          - For Classification : 'Hybrid1 (ESM2-t6 + MERCI)' (default),
                                 'ESM2-t6 based', 'RF based', 'Hybrid2 (RF + MERCI)'
    threshold : ThresholdValue | float | str | None
        Score threshold for Classification ('0.0'–'1.0', including '0.45' and '0.58').
        Ignored in Regression mode. When omitted in Classification mode, uses the
        web server's per-technique default ('0.58' for ESM2-t6/Hybrid1, '0.45' for RF/Hybrid2).
    timeout : float
        HTTP request timeout in seconds (default: 120.0).
    max_retries : int
        Retry count for transient HTTP/network errors (default: 3).
    retry_backoff : float
        Exponential backoff factor for retries (default: 1.0).
    chunk_size : int
        Maximum number of peptides submitted per HTTP batch request (1–1000, default: 100).
    inter_request_delay : float
        Delay in seconds between consecutive chunk submissions (default: 0.5).
    """

    def __init__(
        self,
        method: PredictionMethod | str = PredictionMethod.REGRESSION,
        technique: MLTechnique | str | None = None,
        threshold: ThresholdValue | float | str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        inter_request_delay: float = 0.5,
    ) -> None:
        self.method = PredictionMethod.resolve(method)
        self.technique = MLTechnique.resolve(technique, method=self.method)

        if self.method is PredictionMethod.CLASSIFICATION:
            default_th = self.technique.default_threshold or ThresholdValue.T_0_58
            self.threshold: Optional[ThresholdValue] = ThresholdValue.resolve(
                threshold, default=default_th
            )
        else:
            self.threshold = None

        if not (1 <= chunk_size <= _MAX_SERVER_BATCH):
            raise ValueError(
                f"chunk_size must be between 1 and {_MAX_SERVER_BATCH}, got {chunk_size}."
            )

        self.timeout = timeout
        self.chunk_size = chunk_size
        self.inter_request_delay = inter_request_delay
        self._allowed_hosts = build_allowed_hosts(_BASE_URL)
        self._parser = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    @property
    def method_label(self) -> str:
        return f"{self.method.display_name} ({self.technique.display_name})"

    def predict(self, peptide: str, name: str = "Seq_1") -> PredictionResult:
        """Predict hemolytic activity for a single peptide sequence."""
        return self.predict_batch([(name, peptide)])[0]

    def predict_fasta(self, fasta: str) -> list[PredictionResult]:
        """Parse a multi-sequence FASTA string and predict all sequences."""
        items = self._parse_fasta_items(fasta)
        if not items:
            return [
                PredictionResult(
                    peptide="",
                    method=self.method_label,
                    technique=self.technique.display_name,
                    threshold=self.threshold.value if self.threshold else None,
                    error="FASTA input must not be empty.",
                )
            ]
        return self.predict_batch(items)

    def predict_batch(
        self,
        sequences: Sequence[Union[str, tuple[str, str]]],
    ) -> list[PredictionResult]:
        """
        Predict hemolytic activity for a batch of peptides.

        Accepts either plain peptide strings ``["SEQ1", "SEQ2"]`` or
        ``(name, sequence)`` tuples ``[("Seq_1", "SEQ1"), ...]``.
        Submits valid sequences in chunks to minimize HTTP round trips.
        """
        normalized = self._normalize_items(sequences)
        results: list[Optional[PredictionResult]] = [None] * len(normalized)
        valid_indices: list[int] = []

        for idx, (orig_name, seq) in enumerate(normalized):
            err = self._validate(seq)
            if err:
                results[idx] = self._make_error(orig_name, seq, err)
            else:
                valid_indices.append(idx)

        for chunk_num, start in enumerate(range(0, len(valid_indices), self.chunk_size)):
            if chunk_num > 0 and self.inter_request_delay > 0:
                time.sleep(self.inter_request_delay)

            chunk_idxs = valid_indices[start : start + self.chunk_size]
            fasta_lines: list[str] = []
            id_map: dict[str, tuple[int, str, str]] = {}

            for pos, orig_idx in enumerate(chunk_idxs, start=1):
                orig_name, seq = normalized[orig_idx]
                tag = f"pep_{pos:04d}"
                fasta_lines.append(f">{tag}\n{seq}")
                id_map[tag] = (orig_idx, orig_name, seq)

            fasta_block = "\n".join(fasta_lines)
            try:
                html = self._submit(fasta_block)
                parsed_map = self._parser.parse_batch(html)
                visual_map = self._parser.parse_visual_presentation(html)
                job_id = self._parser.extract_job_id(html)
                server_error = self._parser.extract_server_error(html) if not parsed_map else None

                for tag, (orig_idx, orig_name, seq) in id_map.items():
                    row = parsed_map.get(tag)
                    if not row:
                        msg = (
                            server_error
                            or "HemoPI 2.0 did not return a result row for this sequence."
                        )
                        results[orig_idx] = self._make_error(
                            orig_name, seq, msg, raw_html=html, job_id=job_id
                        )
                    else:
                        props = dict(row["properties"]) if row.get("properties") else {}
                        props.setdefault("Mode", self.method.display_name)
                        props.setdefault("Technique", self.technique.display_name)
                        if self.threshold is not None:
                            props.setdefault("Threshold", float(self.threshold.value))
                        if job_id:
                            props.setdefault("Job ID", job_id)

                        results[orig_idx] = PredictionResult(
                            peptide=seq,
                            peptide_name=orig_name,
                            method=self.method_label,
                            technique=self.technique.display_name,
                            threshold=self.threshold.value if self.threshold else None,
                            score=row["score"],
                            label=row["label"],
                            properties=props or None,
                            visual_presentation=visual_map.get(tag),
                            job_id=job_id,
                            raw_html=html,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error("HemoPI 2.0 batch submission failed: %s", exc)
                for _tag, (orig_idx, orig_name, seq) in id_map.items():
                    results[orig_idx] = self._make_error(orig_name, seq, str(exc))

        return [r for r in results if r is not None]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _submit(self, fasta_block: str) -> str:
        """
        POST FASTA sequences to `prediction_action.php` and follow the
        validated `<meta http-equiv='refresh'>` redirect to `disp_r1.php` / `disp_c1.php`.
        """
        th_value = self.threshold.value if self.threshold is not None else ThresholdValue.T_0_58.value

        data = {
            "selectedMode": self.method.value,
            "seq": fasta_block,
            "MAX_FILE_SIZE": "",
            "mode": self.method.value,
            "terminus": self.technique.value,
            "th": th_value,
            "email": "",
            "email2": "",
        }
        files = {
            "uploadedfile": ("", b"", "application/octet-stream"),
        }

        logger.debug(
            "POST %s  mode=%s  terminus=%s (%s)  th=%s",
            _SUBMIT_URL,
            self.method.value,
            self.technique.value,
            self.technique.display_name,
            th_value,
        )

        first_html = safe_request_text(
            self._session,
            "POST",
            _SUBMIT_URL,
            allowed_hosts=self._allowed_hosts,
            timeout=self.timeout,
            data=data,
            files=files,
        )

        match = _META_REFRESH_RE.search(first_html)
        if not match:
            logger.debug("No meta-refresh redirect in HemoPI 2.0 response; returning directly.")
            return first_html

        redirect_target = match.group(1).strip()
        result_url = resolve_safe_redirect(redirect_target, _SUBMIT_URL, self._allowed_hosts)
        logger.debug("Following validated meta-refresh → %s", result_url)

        return safe_request_text(
            self._session,
            "GET",
            result_url,
            allowed_hosts=self._allowed_hosts,
            timeout=self.timeout,
        )

    def _make_error(
        self,
        name: str,
        seq: str,
        error: str,
        *,
        raw_html: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> PredictionResult:
        return PredictionResult(
            peptide=seq,
            peptide_name=name,
            method=self.method_label,
            technique=self.technique.display_name,
            threshold=self.threshold.value if self.threshold else None,
            job_id=job_id,
            raw_html=raw_html,
            error=error,
        )

    @classmethod
    def _normalize_items(
        cls,
        sequences: Sequence[Union[str, tuple[str, str]]],
    ) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for idx, item in enumerate(sequences, start=1):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name = str(item[0]).strip() or f"Seq_{idx}"
                seq = cls._clean(str(item[1]))
            else:
                raw = str(item)
                name, seq = cls._split_single_fasta(raw, default_name=f"Seq_{idx}")
            normalized.append((name, seq))
        return normalized

    @classmethod
    def _parse_fasta_items(cls, fasta: str) -> list[tuple[str, str]]:
        text = fasta.strip()
        if not text:
            return []
        if ">" not in text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return [(f"Seq_{i}", cls._clean(ln)) for i, ln in enumerate(lines, start=1)]

        items: list[tuple[str, str]] = []
        cur_name: Optional[str] = None
        cur_seq_parts: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if cur_name is not None:
                    items.append((cur_name, "".join(cur_seq_parts).upper()))
                cur_name = stripped[1:].strip() or f"Seq_{len(items) + 1}"
                cur_seq_parts = []
            else:
                if cur_name is None:
                    cur_name = f"Seq_{len(items) + 1}"
                cur_seq_parts.append(stripped)
        if cur_name is not None:
            items.append((cur_name, "".join(cur_seq_parts).upper()))
        return items

    @classmethod
    def _split_single_fasta(cls, raw: str, default_name: str) -> tuple[str, str]:
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            return default_name, ""
        if lines[0].startswith(">"):
            name = lines[0][1:].strip() or default_name
            seq = "".join(ln for ln in lines[1:] if not ln.startswith(">")).upper()
            return name, seq
        return default_name, "".join(lines).upper()

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
        if len(seq) < _MIN_LEN:
            return (
                f"Sequence length {len(seq)} is too short "
                f"(minimum {_MIN_LEN} residues)."
            )
        if not _CANONICAL_AA_RE.fullmatch(seq):
            invalid = sorted(set(re.findall(r"[^ACDEFGHIKLMNPQRSTVWY]", seq)))
            return (
                f"Sequence contains non-natural or invalid characters "
                f"{invalid}: '{seq}'."
            )
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

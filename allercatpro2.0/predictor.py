"""
Predictor: submits peptide/protein sequences to AllerCatPro 2.0 and
returns standardized prediction results.

AllerCatPro 2.0 endpoint:
    POST https://allercatpro.bii.a-star.edu.sg/cgi-bin/allergy2newblastD.pl
    Content-Type: multipart/form-data

Form fields:
    seq      – FASTA-formatted sequence(s) pasted as text
    seqfile  – optional file upload (we always send an empty part)

The server supports up to 50 sequences per submission, with a minimum
sequence length of 8 amino acids.

Usage
-----
Single peptide:

    >>> predictor = Predictor()
    >>> result = predictor.predict("ACDEFGHIKLMNPQRSTVWY", name="my_peptide")
    >>> print(result.result, result.is_allergenic)

Batch – submit all at once, filter out allergens:

    >>> peptides = ["ACDEFGHIKLM", "MKTFLILALLA", ...]
    >>> safe = predictor.filter_safe(peptides)

Batch – one-by-one with retry / delay:

    >>> results = predictor.predict_batch(peptides)
    >>> PredictionResult.to_csv(results, "results.csv")
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import PredictionResult
from .parser import Parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL    = "https://allercatpro.bii.a-star.edu.sg"
_SUBMIT_URL  = f"{_BASE_URL}/cgi-bin/allergy2newblastD.pl"

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Referer": f"{_BASE_URL}/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# AllerCatPro server limits
MAX_SEQUENCES_PER_BATCH = 50
MIN_SEQUENCE_LENGTH     = 8


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """
    Wrapper around the AllerCatPro 2.0 web server.

    Parameters
    ----------
    timeout : float
        HTTP request timeout in seconds (default 60).
    max_retries : int
        Number of automatic retries on server errors (default 3).
    retry_backoff : float
        Exponential backoff factor between retries (default 2.0).
    inter_request_delay : float
        Seconds to wait between consecutive batch requests (default 2.0).
        Be polite to the public server.

    Examples
    --------
    Single prediction::

        predictor = Predictor()
        result = predictor.predict("ACDEFGHIKLM", name="pep_001")
        print(result.result)          # 'no evidence'
        print(result.is_allergenic)   # False

    Batch – submit all at once (≤50 sequences)::

        peptides = ["ACDEFGHIKLM", "MKTFLILALLA"]
        results = predictor.predict_batch_bulk(peptides)
        PredictionResult.to_csv(results, "results.csv")

    Filter safe peptides::

        safe = predictor.filter_safe(peptides)
    """

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        inter_request_delay: float = 2.0,
    ) -> None:
        self.timeout = timeout
        self.inter_request_delay = inter_request_delay
        self._parser = Parser()
        self._session = self._build_session(max_retries, retry_backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        peptide: str,
        *,
        name: str = "query",
    ) -> PredictionResult:
        """
        Predict allergenicity for a single peptide.

        Parameters
        ----------
        peptide : str
            Amino-acid sequence (≥8 residues).
        name : str
            Label for the FASTA header (default ``'query'``).

        Returns
        -------
        PredictionResult
            ``.success`` is ``False`` and ``.error`` is set on failure.
        """
        peptide = peptide.strip().upper()

        err = self._validate(peptide)
        if err:
            logger.warning("Validation failed for '%s': %s", name, err)
            return PredictionResult(peptide=peptide, name=name, error=err)

        fasta = self._to_fasta(name, peptide)
        try:
            html = self._submit(fasta)
        except Exception as exc:  # noqa: BLE001
            logger.error("Request failed for '%s': %s", name, exc)
            return PredictionResult(peptide=peptide, name=name, error=str(exc))

        parsed = self._parser.parse(html)
        if not parsed:
            return PredictionResult(
                peptide=peptide,
                name=name,
                error="Parser returned no results.",
                raw_html=html,
            )

        result = parsed[0]
        result.peptide = peptide
        result.raw_html = html
        logger.info("Predicted '%s' → %s", name, result.result)
        return result

    def predict_batch_bulk(
        self,
        peptides: list[str],
        *,
        names: Optional[list[str]] = None,
        chunk_size: int = MAX_SEQUENCES_PER_BATCH,
    ) -> list[PredictionResult]:
        """
        Submit multiple sequences in bulk (up to ``chunk_size`` per request).

        This is the most efficient approach because AllerCatPro accepts up to
        50 sequences in a single POST, avoiding repeated round-trips.

        Parameters
        ----------
        peptides : list[str]
            Amino-acid sequences to evaluate.
        names : list[str], optional
            FASTA labels for each peptide. Auto-generated when omitted.
        chunk_size : int
            Number of sequences per HTTP request (max 50).

        Returns
        -------
        list[PredictionResult]
            One result per input peptide, preserving order.
        """
        if names is None:
            names = [f"pep_{i + 1:04d}" for i in range(len(peptides))]

        if len(names) != len(peptides):
            raise ValueError("`names` must have the same length as `peptides`.")

        chunk_size = min(chunk_size, MAX_SEQUENCES_PER_BATCH)
        all_results: list[PredictionResult] = []

        # Split into chunks of ≤ chunk_size
        chunks = [
            (peptides[i: i + chunk_size], names[i: i + chunk_size])
            for i in range(0, len(peptides), chunk_size)
        ]

        for chunk_idx, (chunk_peps, chunk_names) in enumerate(chunks):
            if chunk_idx > 0:
                logger.debug("Waiting %.1fs before next chunk …", self.inter_request_delay)
                time.sleep(self.inter_request_delay)

            chunk_results = self._predict_chunk(chunk_peps, chunk_names)
            all_results.extend(chunk_results)
            logger.info(
                "Chunk %d/%d done (%d sequences).",
                chunk_idx + 1,
                len(chunks),
                len(chunk_peps),
            )

        return all_results

    def predict_batch(
        self,
        peptides: list[str],
        *,
        names: Optional[list[str]] = None,
        on_error: str = "continue",
    ) -> list[PredictionResult]:
        """
        Predict a list of peptides **one by one** (sequential single requests).

        Use :meth:`predict_batch_bulk` when you want to batch multiple
        sequences into fewer HTTP calls. Use this method when you need
        per-peptide granular error handling.

        Parameters
        ----------
        peptides : list[str]
        names : list[str], optional
        on_error : {'continue', 'raise'}

        Returns
        -------
        list[PredictionResult]
        """
        if names is None:
            names = [f"pep_{i + 1:04d}" for i in range(len(peptides))]

        results: list[PredictionResult] = []
        for i, (peptide, name) in enumerate(zip(peptides, names)):
            if i > 0:
                time.sleep(self.inter_request_delay)

            result = self.predict(peptide, name=name)

            if not result.success and on_error == "raise":
                raise RuntimeError(
                    f"Prediction failed for '{name}': {result.error}"
                )

            results.append(result)
            logger.debug(
                "[%d/%d] %s → %s",
                i + 1,
                len(peptides),
                name,
                result.result,
            )

        return results

    def filter_safe(
        self,
        peptides: list[str],
        *,
        names: Optional[list[str]] = None,
        use_bulk: bool = True,
    ) -> list[str]:
        """
        Return only the peptides that show **no allergenic evidence**.

        This is the primary use-case: you pass in a set of candidate
        peptides and receive back only those that AllerCatPro classifies
        as ``'no evidence'``.

        Implementation note
        -------------------
        Instead of a ``for``-loop comparison against a separate allergen
        list, we leverage a ``set`` of allergenic sequences for O(1)
        membership testing:

            safe = {r.peptide for r in results if r.is_safe}
            return [p for p in peptides if p.upper() in safe]

        This preserves the original submission order and handles
        duplicates correctly (a peptide that appears twice and is safe
        appears twice in the output).

        Parameters
        ----------
        peptides : list[str]
            Input sequences.
        names : list[str], optional
            FASTA labels (auto-generated when omitted).
        use_bulk : bool
            If ``True`` (default), use :meth:`predict_batch_bulk` for
            efficiency. Set to ``False`` to use one-by-one submission.

        Returns
        -------
        list[str]
            Safe (non-allergenic) peptides in their original order.
        """
        if use_bulk:
            results = self.predict_batch_bulk(peptides, names=names)
        else:
            results = self.predict_batch(peptides, names=names)

        # Build a set of allergenic sequences for O(1) lookup
        allergenic: set[str] = {
            r.peptide.upper()
            for r in results
            if r.success and r.is_allergenic
        }

        safe_peptides = [p for p in peptides if p.strip().upper() not in allergenic]
        logger.info(
            "filter_safe: %d input → %d safe, %d allergenic",
            len(peptides),
            len(safe_peptides),
            len(allergenic),
        )
        return safe_peptides

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_chunk(
        self,
        peptides: list[str],
        names: list[str],
    ) -> list[PredictionResult]:
        """Submit a single chunk (≤50 sequences) and return parsed results."""
        # Build validated list; mark validation errors immediately
        valid_entries: list[tuple[str, str]] = []       # (name, upper_seq)
        pre_results: list[PredictionResult] = []        # slot per input peptide

        for peptide, name in zip(peptides, names):
            peptide_up = peptide.strip().upper()
            err = self._validate(peptide_up)
            if err:
                logger.warning("Validation failed for '%s': %s", name, err)
                pre_results.append(
                    PredictionResult(peptide=peptide_up, name=name, error=err)
                )
            else:
                pre_results.append(None)  # placeholder
                valid_entries.append((name, peptide_up))

        if not valid_entries:
            return [r for r in pre_results if r is not None]

        # Build FASTA block for valid sequences only
        fasta_block = "\n".join(
            self._to_fasta(name, seq) for name, seq in valid_entries
        )

        try:
            html = self._submit(fasta_block)
        except Exception as exc:  # noqa: BLE001
            logger.error("Chunk request failed: %s", exc)
            error_result = [
                PredictionResult(peptide=seq, name=name, error=str(exc))
                for name, seq in valid_entries
            ]
            # Merge back in order
            valid_iter = iter(error_result)
            return [
                r if r is not None else next(valid_iter)
                for r in pre_results
            ]

        parsed = self._parser.parse(html)

        # Align parsed results with valid_entries (server returns in same order)
        parsed_iter = iter(parsed)
        aligned_parsed: list[PredictionResult] = []
        for name, seq in valid_entries:
            p = next(parsed_iter, None)
            if p is None:
                aligned_parsed.append(
                    PredictionResult(
                        peptide=seq,
                        name=name,
                        error="Parser returned fewer results than expected.",
                        raw_html=html,
                    )
                )
            else:
                p.peptide = seq
                if not p.name:
                    p.name = name
                p.raw_html = html
                aligned_parsed.append(p)

        # Merge validation errors and parsed results back into original order
        aligned_iter = iter(aligned_parsed)
        return [
            r if r is not None else next(aligned_iter)
            for r in pre_results
        ]

    def _submit(self, fasta_block: str) -> str:
        """POST FASTA to AllerCatPro and return raw HTML.

        Submission strategy
        -------------------
        AllerCatPro 2.0's server-side Perl CGI.pm does not parse the
        ``seq`` textarea field correctly when multiple FASTA sequences
        are submitted via Python's multipart encoder: the boundary
        delimiter leaks into the second (and subsequent) sequence
        values, causing inflated ``seq_len`` readings and wrong
        predictions.

        Uploading the FASTA content as the ``seqfile`` file-upload
        field avoids this issue entirely because the server's file-
        handling code correctly delimits the file content from the rest
        of the multipart body.
        """
        from utils.security import build_allowed_hosts, safe_request_text

        allowed_hosts = build_allowed_hosts(_BASE_URL)
        logger.debug("POST %s  (%d chars FASTA)", _SUBMIT_URL, len(fasta_block))
        return safe_request_text(
            self._session,
            "POST",
            _SUBMIT_URL,
            allowed_hosts=allowed_hosts,
            timeout=self.timeout,
            data={"seq": ""},           # empty textarea (required field)
            files={
                "seqfile": (
                    "sequences.fasta",
                    fasta_block.encode("utf-8"),
                    "text/plain",
                ),
            },
        )


    @staticmethod
    def _validate(peptide: str) -> Optional[str]:
        """
        Basic client-side validation.
        Returns an error string, or ``None`` if valid.
        """
        import re
        if not peptide:
            return "Sequence must not be empty."
        if not re.fullmatch(r"[A-Z]+", peptide):
            return f"Sequence contains invalid characters: '{peptide}'."
        if len(peptide) < MIN_SEQUENCE_LENGTH:
            return (
                f"Sequence length {len(peptide)} is below the minimum "
                f"of {MIN_SEQUENCE_LENGTH} residues."
            )
        return None

    @staticmethod
    def _to_fasta(name: str, sequence: str) -> str:
        """Format a single sequence as a FASTA entry."""
        return f">{name}\n{sequence}"

    @staticmethod
    def _build_session(max_retries: int, backoff: float) -> requests.Session:
        """Create a requests Session with retry logic."""
        session = requests.Session()
        session.headers.update(_DEFAULT_HEADERS)

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

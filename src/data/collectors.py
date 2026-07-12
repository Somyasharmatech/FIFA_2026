"""Automated dataset collection.

Downloads public football datasets defined in ``config/config.yaml``
with retry logic and exponential backoff, caches them under
``data/raw/``, and exposes them as pandas DataFrames.

No predictions are hardcoded anywhere in this platform; collectors only
fetch raw historical facts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.core.config import AppConfig, DatasetConfig

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded after all retries."""


class DatasetCollector:
    """Downloads and caches the datasets declared in the configuration."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.http.user_agent})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Download (or load from cache) every configured dataset.

        Args:
            force_refresh: When ``True``, ignore cached files and re-download.

        Returns:
            Mapping of dataset name to its DataFrame.
        """
        self._config.ensure_directories()
        frames: dict[str, pd.DataFrame] = {}
        for name, dataset in self._config.datasets.items():
            frames[name] = self.collect(dataset, force_refresh=force_refresh)
        logger.info("Collected %d datasets", len(frames))
        return frames

    def collect(self, dataset: DatasetConfig, force_refresh: bool = False) -> pd.DataFrame:
        """Return a single dataset, using the local cache when available."""
        target = self._config.raw_dir / dataset.filename
        if target.exists() and not force_refresh:
            logger.info("Cache hit for '%s' (%s)", dataset.name, target)
            return pd.read_csv(target)

        self._download(dataset.url, target)
        frame = pd.read_csv(target)
        logger.info("Downloaded '%s': %d rows, %d columns", dataset.name, *frame.shape)
        return frame

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _download(self, url: str, target: Path) -> None:
        """Fetch ``url`` to ``target`` with retries and exponential backoff."""
        http = self._config.http
        last_error: Exception | None = None
        for attempt in range(1, http.max_retries + 1):
            try:
                response = self._session.get(url, timeout=http.timeout_seconds)
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                return
            except requests.RequestException as exc:  # network or HTTP error
                last_error = exc
                wait = http.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d failed for %s (%s); retrying in %.1fs",
                    attempt, http.max_retries, url, exc, wait,
                )
                time.sleep(wait)
        raise DownloadError(f"Failed to download {url} after {http.max_retries} attempts") from last_error

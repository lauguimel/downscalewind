"""AEMET OpenData client with two-step fetch, cache, and strict rate limit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

BASE_URL = "https://opendata.aemet.es/opendata/api"


class AemetHTTPError(RuntimeError):
    """Permanent AEMET HTTP/envelope error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AemetTransientError(AemetHTTPError):
    """Retryable AEMET error such as 429/5xx or connection failures."""


class AemetCacheMiss(RuntimeError):
    """Raised when cache-only mode cannot satisfy a request."""


def _dms_to_deg(s: str) -> float:
    """Parse AEMET compact DMS coordinates like ``410756N`` or ``033456W``."""
    text = str(s).strip().upper()
    if len(text) < 6:
        raise ValueError(f"invalid DMS coordinate: {s!r}")
    hemi = text[-1]
    if hemi not in {"N", "S", "E", "W"}:
        raise ValueError(f"invalid DMS hemisphere: {s!r}")
    digits = text[:-1]
    if not digits.isdigit() or len(digits) < 5:
        raise ValueError(f"invalid DMS digits: {s!r}")
    deg = int(digits[:-4])
    minutes = int(digits[-4:-2])
    seconds = int(digits[-2:])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid DMS minutes/seconds: {s!r}")
    value = deg + minutes / 60.0 + seconds / 3600.0
    return -value if hemi in {"S", "W"} else value


def wind_speed_dir_to_uv(
    speed: np.ndarray,
    direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert wind speed and meteorological direction to east/north components."""
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u.astype(np.float32), v.astype(np.float32)


class AemetClient:
    """Small client for AEMET OpenData endpoints used by the ES OBS ingester."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path = "tmp/aemet_cache",
        *,
        base_url: str = BASE_URL,
        min_interval_s: float = 1.05,
        timeout_s: float = 60.0,
        logger: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.log = logger
        self._last_call_start: float | None = None

    def station_inventory(self, *, cache_only: bool = False) -> list[dict[str, Any]]:
        payload = self.get_json(
            "/valores/climatologicos/inventarioestaciones/todasestaciones",
            cache_only=cache_only,
        )
        return payload if isinstance(payload, list) else []

    def hourly_archive(
        self,
        idema: str,
        year: int,
        *,
        cache_only: bool = False,
    ) -> list[dict[str, Any]]:
        payload = self.get_json(
            f"/valores/climatologicos/horarios/datos/anyo/{year}/estacion/{idema}",
            cache_only=cache_only,
        )
        return _as_records(payload)

    def daily_archive(
        self,
        idema: str,
        ini: str,
        fin: str,
        *,
        cache_only: bool = False,
    ) -> list[dict[str, Any]]:
        payload = self.get_json(
            "/valores/climatologicos/diarios/datos/"
            f"fechaini/{ini}/fechafin/{fin}/estacion/{idema}",
            cache_only=cache_only,
        )
        return _as_records(payload)

    def get_json(self, path_or_url: str, *, cache_only: bool = False) -> Any:
        """Fetch an AEMET endpoint envelope, then its signed ``datos`` URL."""
        url = self._url(path_or_url)
        cache_path = self._cache_path("GET", url)
        cached = self._read_cache(cache_path)
        if cached is not None:
            self._log_get(url, True)
            return cached
        if cache_only:
            raise AemetCacheMiss(f"cache miss for {url}")

        envelope = self._request_json(url, headers={"api_key": self.api_key})
        estado = _to_int(envelope.get("estado")) if isinstance(envelope, dict) else None
        if estado != 200:
            desc = envelope.get("descripcion", "") if isinstance(envelope, dict) else ""
            msg = f"AEMET envelope failed estado={estado}: {desc}"
            if estado in {429, 500, 502, 503, 504}:
                raise AemetTransientError(msg, estado)
            raise AemetHTTPError(msg, estado)
        data_url = envelope.get("datos")
        if not data_url:
            raise AemetHTTPError(f"AEMET envelope missing datos URL: {url}", estado)

        payload = self._request_json(str(data_url), headers=None)
        self._write_cache(cache_path, payload, endpoint_url=url, data_url=str(data_url))
        return payload

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.base_url}/{path_or_url.lstrip('/')}"

    def _cache_path(self, method: str, url: str) -> Path:
        key = hashlib.sha256(f"{method.upper()}{url}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and "_aemet_cache_payload" in cached:
            return cached["_aemet_cache_payload"]
        return cached

    def _write_cache(self, path: Path, payload: Any, *, endpoint_url: str, data_url: str) -> None:
        tmp = path.with_suffix(".tmp")
        content = {
            "_aemet_cache_payload": payload,
            "endpoint_url": endpoint_url,
            "data_url": data_url,
            "cached_at_unix": time.time(),
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False)
        tmp.replace(path)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(AemetTransientError),
        reraise=True,
    )
    def _request_json(self, url: str, headers: dict[str, str] | None) -> Any:
        self._sleep_for_rate_limit()
        try:
            response = self.session.get(url, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise AemetTransientError(f"AEMET request failed: {exc}") from exc
        self._log_get(url, False)
        if response.status_code == 429 or response.status_code >= 500:
            raise AemetTransientError(
                f"AEMET transient HTTP {response.status_code}: {url}",
                response.status_code,
            )
        if response.status_code >= 400:
            raise AemetHTTPError(
                f"AEMET HTTP {response.status_code}: {response.text[:200]}",
                response.status_code,
            )
        response.encoding = response.encoding or "utf-8"
        try:
            return response.json()
        except ValueError as exc:
            text = response.text.lstrip("\ufeff").strip()
            try:
                return json.loads(text)
            except ValueError:
                raise AemetHTTPError(f"AEMET response is not JSON: {url}") from exc

    def _sleep_for_rate_limit(self) -> None:
        now = time.monotonic()
        if self._last_call_start is not None:
            wait_s = self.min_interval_s - (now - self._last_call_start)
            if wait_s > 0:
                time.sleep(wait_s)
        self._last_call_start = time.monotonic()

    def _log_get(self, url: str, cache: bool) -> None:
        if self.log is None:
            return
        ts_unix = time.time()
        if not cache and self._last_call_start is not None:
            ts_unix -= time.monotonic() - self._last_call_start
        extra = {"event": "aemet_get", "url": url, "t_ms": int(ts_unix * 1000), "cache": cache}
        if cache:
            self.log.debug("aemet_get", extra=extra)
            return
        self.log.info(
            "aemet_get",
            extra={**extra, "unix_ts": ts_unix},
        )


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("datos", "data", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

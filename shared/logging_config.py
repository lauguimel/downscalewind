"""
shared.logging_config — Logger structuré JSON pour tous les services DownscaleWind.

Usage :
    from shared.logging_config import get_logger
    log = get_logger("ingest_era5")
    log.info("Téléchargement commencé", extra={"month": "2017-05", "size_mb": 42})
    log.error("Erreur CDS", extra={"attempt": 2, "wait_s": 60}, exc_info=True)

Format de sortie :
    {"ts": "2024-02-01T12:00:00+00:00", "level": "INFO", "logger": "ingest_era5",
     "msg": "Téléchargement commencé", "month": "2017-05", "size_mb": 42}
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Formatter qui sérialise chaque log record en JSON sur une ligne."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Champs supplémentaires passés via extra={"_key": value}
        # Convention : les clés passées dans extra= sont directement incluses
        # sauf les clés internes de logging (levelname, msg, etc.)
        _internal = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
            "message",
        }
        for k, v in record.__dict__.items():
            if k not in _internal and not k.startswith("_"):
                try:
                    json.dumps(v)  # vérifier que la valeur est JSON-sérialisable
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Retourne un logger JSON structuré.

    Args:
        name:  Nom du logger (typiquement __name__ ou le nom du service).
        level: Niveau de log (défaut : variable d'env LOG_LEVEL ou INFO).

    Returns:
        logging.Logger configuré avec le formatter JSON.
    """
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    logger = logging.getLogger(name)

    # Éviter de doubler les handlers si get_logger est appelé plusieurs fois
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)

    logger.setLevel(level)
    # Ne pas propager vers le root logger (évite les doublons)
    logger.propagate = False

    return logger

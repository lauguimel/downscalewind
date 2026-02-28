"""
utils.checkpointing — Sentinelles de téléchargement et hashes SHA256.

Permet de reprendre un téléchargement interrompu sans retraiter les mois déjà
téléchargés. Chaque sentinel contient le SHA256 du fichier brut pour vérification
de l'intégrité à la réutilisation.

Usage :
    from utils.checkpointing import Checkpointer

    cp = Checkpointer("data/raw/.checkpoints")
    key = "era5_2017_05"
    if cp.is_done(key):
        log.info("Mois déjà téléchargé, passage au suivant", extra={"key": key})
    else:
        # ... téléchargement ...
        cp.mark_done(key, sha256="abc123...", extra_meta={"n_times": 120})
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Checkpointer:
    """Gère les sentinelles de téléchargement dans un répertoire dédié."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _sentinel_path(self, key: str) -> Path:
        # Sanitiser la clé pour un nom de fichier valide
        safe_key = key.replace("/", "_").replace(" ", "_")
        return self.dir / f"{safe_key}.done"

    def is_done(self, key: str) -> bool:
        """Vérifie si la clé a déjà été traitée avec succès."""
        return self._sentinel_path(key).exists()

    def mark_done(
        self,
        key: str,
        sha256: str | None = None,
        extra_meta: dict | None = None,
    ) -> None:
        """
        Marque une clé comme traitée avec succès.

        Args:
            key:        Identifiant unique (ex: "era5_2017_05")
            sha256:     Hash SHA256 du fichier téléchargé (pour intégrité)
            extra_meta: Métadonnées additionnelles (taille, nombre de pas, etc.)
        """
        sentinel = self._sentinel_path(key)
        content = {
            "key":        key,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "sha256":     sha256,
        }
        if extra_meta:
            content.update(extra_meta)
        sentinel.write_text(json.dumps(content, indent=2, ensure_ascii=False))

    def get_meta(self, key: str) -> dict | None:
        """Lit les métadonnées d'un sentinel existant, ou None si absent."""
        sentinel = self._sentinel_path(key)
        if not sentinel.exists():
            return None
        return json.loads(sentinel.read_text())

    def list_done(self) -> list[str]:
        """Liste toutes les clés marquées comme traitées."""
        return [p.stem for p in sorted(self.dir.glob("*.done"))]

    def clear(self, key: str) -> None:
        """Supprime le sentinel d'une clé (pour forcer un re-téléchargement)."""
        sentinel = self._sentinel_path(key)
        if sentinel.exists():
            sentinel.unlink()

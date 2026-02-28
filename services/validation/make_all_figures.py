"""
make_all_figures.py — Regenerate all Module 2 figures from existing data

Usage
-----
    python make_all_figures.py \
        --srtm   data/raw/srtm_perdigao_30m.tif \
        --cfd-db data/cfd-database/perdigao \
        --output figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_all_figures(
    srtm_tif: Path | None,
    cfd_db: Path,
    output_dir: Path,
) -> None:
    """Regenerate all figures."""
    from services.validation.plot_terrain_refinement import plot_terrain_refinement

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Fig 1: Terrain refinement 4-panel ===")
    plot_terrain_refinement(
        srtm_tif=srtm_tif,
        output_path=output_dir / "terrain_refinement_4panel.png",
    )

    logger.info("All figures generated in %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Regenerate all Module 2 figures")
    parser.add_argument("--srtm",   default=None)
    parser.add_argument("--cfd-db", default="data/cfd-database/perdigao")
    parser.add_argument("--output", default="figures/")
    args = parser.parse_args()

    make_all_figures(
        srtm_tif=Path(args.srtm) if args.srtm else None,
        cfd_db=Path(args.cfd_db),
        output_dir=Path(args.output),
    )

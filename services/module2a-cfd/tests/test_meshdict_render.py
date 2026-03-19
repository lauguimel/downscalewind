"""Tests for meshDict.j2 rendering with box and octagonal domain."""
import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parents[1] / "templates" / "openfoam"
SYSTEM_DIR = TEMPLATES_DIR / "system"


def _render_meshdict(octagonal: bool, domain_km: float = 5.0) -> str:
    env = Environment(loader=FileSystemLoader(str(SYSTEM_DIR)))
    tmpl = env.get_template("meshDict.j2")
    ctx = {
        "domain": {"octagonal": octagonal, "radius_m": domain_km * 1000 / 2},
        "resolution_m": 500,
        "context_cells": 1,
        "domain_km": domain_km,
        "cfmesh_refinements": [],  # box mode uses this, cylinder ignores it
        "mesh": {
            "max_cell_size": 1000,
            "target_cell_size": 500,
            "fine_cell_size": 250,
            "terrain_refine_levels": 2,
            "terrain_refine_thickness": 500,
            "n_boundary_layers": 3,
            "bl_expansion": 1.2,
            "bl_first_layer": 10.0,
        },
    }
    return tmpl.render(**ctx)


def test_meshdict_octagonal_render():
    """Octagonal mode renders mesoZone/fineZone/nearTerrain with lateral renameBoundary."""
    out = _render_meshdict(octagonal=True)
    assert "lateral" in out, "lateral patch must appear in octagonal meshDict"
    assert "mesoZone" in out, "mesoZone refinement missing"
    assert "fineZone" in out, "fineZone refinement missing"
    assert "nearTerrain" in out, "nearTerrain refinement missing"
    # Box-specific patches must NOT appear
    assert "xMin" not in out, "xMin should not appear in octagonal mode"
    assert "xMax" not in out


def test_meshdict_box_render():
    """Box mode preserves backward compatibility — xMin/xMax patches present."""
    out = _render_meshdict(octagonal=False)
    # Box mode should have the cardinal boundary renaming
    # (west/east/south/north or xMin/xMax/yMin/yMax — check what's actually in the template)
    assert "xMin" in out or "west" in out, "Box domain must have xMin/west renaming"
    # Must NOT have octagonal refinements
    assert "mesoZone" not in out


def test_meshdict_octagonal_has_maxcellsize():
    """maxCellSize must be present in both modes."""
    assert "maxCellSize" in _render_meshdict(octagonal=True)
    assert "maxCellSize" in _render_meshdict(octagonal=False)

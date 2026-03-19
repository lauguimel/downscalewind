"""Tests for make_octagon_stl() and terrain leveling in generate_mesh.py."""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from generate_mesh import make_octagon_stl  # noqa: E402


def _parse_stl_vertices(stl_str: str, solid_name: str) -> list[tuple]:
    """Extract all (x,y,z) vertex tuples from a named solid."""
    # Find the solid block
    pattern = rf"solid {solid_name}(.*?)endsolid {solid_name}"
    match = re.search(pattern, stl_str, re.DOTALL)
    assert match, f"Solid '{solid_name}' not found in STL"
    block = match.group(1)
    vertices = []
    for m in re.finditer(r"vertex\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", block):
        vertices.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return vertices


def _count_facets(stl_str: str, solid_name: str) -> int:
    pattern = rf"solid {solid_name}(.*?)endsolid {solid_name}"
    match = re.search(pattern, stl_str, re.DOTALL)
    assert match
    return stl_str.count("facet normal", match.start(), match.end())


def test_make_octagon_stl_geometry():
    """Octagon STL has correct structure: 16 lateral facets, 8 top facets, correct radius."""
    cx, cy = 0.0, 0.0
    radius = 14000.0
    height = 3000.0
    stl = make_octagon_stl(cx, cy, radius, height, n_sides=8)

    assert "solid lateral" in stl
    assert "endsolid lateral" in stl
    assert "solid top" in stl
    assert "endsolid top" in stl

    # 8 panels × 2 triangles = 16 lateral facets
    assert _count_facets(stl, "lateral") == 16

    # 8 top triangles
    assert _count_facets(stl, "top") == 8

    # All lateral vertices at radius ± 1 m (in XY)
    verts = _parse_stl_vertices(stl, "lateral")
    for x, y, z in verts:
        r = math.sqrt((x - cx)**2 + (y - cy)**2)
        assert abs(r - radius) < 1.0, f"Vertex at unexpected radius {r:.1f} != {radius}"
        assert z in (0.0, height) or abs(z) < 1.0 or abs(z - height) < 1.0


def test_make_octagon_stl_non_square():
    """Octagon STL works with off-center coordinates."""
    stl = make_octagon_stl(100_000.0, 200_000.0, 5000.0, 2000.0)
    assert "solid lateral" in stl
    assert "solid top" in stl
    assert len(stl) > 100


def test_level_terrain_blending():
    """Terrain leveling: center unchanged, boundary ≈ 0, monotonically decreasing."""
    from generate_mesh import _level_terrain

    R = 10000.0
    # Grid: 0 to R in 11 steps
    x_vals = np.linspace(0, R, 11)
    z_vals = np.ones(11) * 100.0  # flat 100m terrain

    # Center point (r=0): should be unchanged
    z_leveled = _level_terrain(z_vals, x_vals, np.zeros(11), 0.0, 0.0, R)
    assert abs(z_leveled[0] - 100.0) < 0.1, f"Center should be ~100m, got {z_leveled[0]:.2f}"

    # Boundary point (r=R): should be near 0
    assert z_leveled[-1] < 5.0, f"Boundary should be ~0m, got {z_leveled[-1]:.2f}"

    # Monotonically decreasing (leveling increases with r)
    assert np.all(np.diff(z_leveled) <= 0.0 + 1e-10), "Leveling should decrease with r"

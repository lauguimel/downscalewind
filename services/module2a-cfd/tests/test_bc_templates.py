"""Tests for BC template rendering with box and octagonal domain."""
import sys
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

TEMPLATES_DIR = Path(__file__).parents[1] / "templates" / "openfoam"
ZERO_DIR = TEMPLATES_DIR / "0"


def _make_env():
    return Environment(loader=FileSystemLoader(str(ZERO_DIR)))


def _base_ctx(octagonal: bool) -> dict:
    """Minimal context for rendering BC templates."""
    return {
        "domain": {"octagonal": octagonal},
        "inflow": {
            "u_hub": 10.0,
            "wind_dir": 231.0,
            "flowDir_x": -0.5,
            "flowDir_y": -0.866,
            "u_star": 0.4,
            "z0": 0.05,
            "T_ref": 288.15,
            "T_grad_Km": 0.0,
            "Ri_b": 0.0,
            "L_mo": 0,
            "kappa": 0.41,
            "z_ref": 10.0,
        },
        "physics": {
            "p_rgh_top": 100000.0,
        },
        "thermal": True,
        "coriolis": False,
        "solver": {"name": "buoyantBoussinesqSimpleFoam"},
    }


def _has_patch(out: str, patch_name: str) -> bool:
    """Return True if `patch_name { ... }` appears as an OF boundary patch block."""
    import re
    return bool(re.search(r'^\s+' + re.escape(patch_name) + r'\s*\n\s*\{', out, re.MULTILINE))


@pytest.mark.parametrize("template_name", ["U.j2", "k.j2", "p_rgh.j2", "nut.j2"])
def test_octagonal_has_lateral_no_cardinal(template_name):
    """In octagonal mode, 'lateral' is present; west/east/south/north are absent."""
    env = _make_env()
    tmpl = env.get_template(template_name)
    out = tmpl.render(**_base_ctx(octagonal=True))
    assert _has_patch(out, "lateral"), f"{template_name}: 'lateral' patch missing in octagonal mode"
    for face in ["west", "east", "south", "north"]:
        assert not _has_patch(out, face), f"{template_name}: '{face}' should not appear in octagonal mode"


@pytest.mark.parametrize("template_name", ["U.j2", "k.j2", "p_rgh.j2", "nut.j2"])
def test_box_has_cardinal_no_lateral(template_name):
    """In box mode, west/east/south/north are present; lateral is absent."""
    env = _make_env()
    tmpl = env.get_template(template_name)
    out = tmpl.render(**_base_ctx(octagonal=False))
    assert _has_patch(out, "west"), f"{template_name}: 'west' patch missing in box mode"
    assert _has_patch(out, "north"), f"{template_name}: 'north' patch missing in box mode"
    assert not _has_patch(out, "lateral"), f"{template_name}: 'lateral' should not appear in box mode"


def test_top_terrain_always_present():
    """top and terrain patches must be present in both modes."""
    env = _make_env()
    for tmpl_name in ["U.j2", "k.j2"]:
        tmpl = env.get_template(tmpl_name)
        for octagonal in [True, False]:
            out = tmpl.render(**_base_ctx(octagonal=octagonal))
            assert _has_patch(out, "top"), f"{tmpl_name} octagonal={octagonal}: 'top' missing"
            assert _has_patch(out, "terrain"), f"{tmpl_name} octagonal={octagonal}: 'terrain' missing"

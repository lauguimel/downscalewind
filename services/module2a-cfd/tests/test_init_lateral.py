"""Tests for detect_lateral_patches in init_from_era5.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from init_from_era5 import detect_lateral_patches


def test_detect_lateral_patches_octagonal():
    """Cylindrical domain: returns {'lateral'}."""
    result = detect_lateral_patches({"lateral": {}, "top": {}, "terrain": {}})
    assert result == {"lateral"}


def test_detect_lateral_patches_box():
    """Box domain: returns the four cardinal patches."""
    result = detect_lateral_patches({
        "west": {}, "east": {}, "south": {}, "north": {},
        "top": {}, "terrain": {}
    })
    assert result == {"west", "east", "south", "north"}


def test_detect_lateral_patches_mixed():
    """When 'lateral' is present alongside cardinal patches, lateral wins."""
    result = detect_lateral_patches({
        "lateral": {}, "west": {}, "east": {}, "top": {}
    })
    assert result == {"lateral"}


def test_detect_lateral_patches_empty():
    """Empty dict -> falls back to box cardinal patches."""
    result = detect_lateral_patches({})
    assert result == {"west", "east", "south", "north"}

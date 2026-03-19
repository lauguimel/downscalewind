# Plan: Phase 1 — Cylindrical CFD domain infrastructure
Generated: 2026-03-19
Test command: conda run -n downscalewind pytest services/module2a-cfd/tests/ -v

## Phase 1 — Octagon STL generator + terrain leveling

**Files to create/modify:**
- `services/module2a-cfd/generate_mesh.py` (add `make_octagon_stl()` + terrain leveling)
- `services/module2a-cfd/tests/test_octagon_stl.py` (new test file)

**What to implement:**
Add `make_octagon_stl(center_lat, center_lon, radius_m, height_m, n_sides=8)` that builds a regular octagonal prism as ASCII multi-solid STL with solids `lateral` (8 side panels, each 2 triangles) and `top` (fan of 8 triangles from centre). Add terrain leveling `z × (1 − tanh((1.6r/R)^8))` as an optional step in `dem_to_stl()` via new `level_terrain=True` and `domain_radius_m` parameters applied after reprojection, before triangulation. No change to the existing box path — new code is additive.

**Success test:** `services/module2a-cfd/tests/test_octagon_stl.py::test_make_octagon_stl_geometry` — parse the returned STL string, verify 8×2 triangles in solid `lateral`, all vertices at the expected radius ± 1 m, `top` solid present. `test_level_terrain_blending` — verify leveling returns Z unchanged at r=0, ≈0 at r=R.

**Status:** done


## Phase 2 — meshDict.j2: 3-ring octagonal refinement

**Files to create/modify:**
- `services/module2a-cfd/templates/openfoam/system/meshDict.j2`
- `services/module2a-cfd/generate_mesh.py` (add `domain_type` parameter + octagon FMS builder call)
- `services/module2a-cfd/tests/test_meshdict_render.py` (new test file)

**What to implement:**
Add `domain_type` parameter (`"box"` default, `"cylinder"`) to `generate_mesh()`. When `"cylinder"`: call `make_octagon_stl()` → write `constant/triSurface/domain_octagon.stl`, update Jinja context with `domain.octagonal=True`. Update `meshDict.j2` with a Jinja conditional: when octagonal, emit 3-ring objectRefinements (`mesoZone` 200m / `fineZone` 100m / `nearTerrain` 50m) and a `renameBoundary` block collapsing sides to single `lateral` patch. Box path (default) is completely unchanged.

**Success test:** `services/module2a-cfd/tests/test_meshdict_render.py::test_meshdict_octagonal_render` — render with `domain.octagonal=True`, assert `lateral` in renameBoundary, `mesoZone`/`fineZone`/`nearTerrain` present, `xMin`/`xMax` absent. `test_meshdict_box_render` — render with `domain.octagonal=False`, assert old box renaming intact.

**Status:** done


## Phase 3 — BC templates: 4-face loop → single lateral patch

**Files to modify:**
- `services/module2a-cfd/templates/openfoam/0/U.j2`
- `services/module2a-cfd/templates/openfoam/0/k.j2`
- `services/module2a-cfd/templates/openfoam/0/epsilon.j2`
- `services/module2a-cfd/templates/openfoam/0/T.j2`
- `services/module2a-cfd/templates/openfoam/0/p_rgh.j2`
- `services/module2a-cfd/templates/openfoam/0/nut.j2`
- `services/module2a-cfd/templates/openfoam/0/alphat.j2` (if it exists)
- `services/module2a-cfd/tests/test_bc_templates.py` (new test file)

**What to implement:**
In each template, add a Jinja conditional on `domain.octagonal`. When `True`: emit a single `lateral { ... }` block (same BC type as the 4-face loop). When `False` (default): keep the existing `{% for face in ['west', 'east', 'south', 'north'] %}` loop unchanged. `top`, `terrain`, and `bottom` blocks are unchanged in both branches.

**Success test:** `services/module2a-cfd/tests/test_bc_templates.py::test_U_octagonal` — render `U.j2` with `domain.octagonal=True`, assert `lateral` present exactly once, `west`/`east`/`south`/`north` absent. `test_U_box` — render with `domain.octagonal=False`, assert `west` and `north` present. Parametrize over `k`, `p_rgh`, `nut`.

**Status:** done


## Phase 4 — init_from_era5.py: lateral patch auto-detection

**Files to modify:**
- `services/module2a-cfd/init_from_era5.py`
- `services/module2a-cfd/tests/test_init_lateral.py` (new test file)

**What to implement:**
Replace the hardcoded `BOUNDARY_DATA_PATCHES = {"west", "east", "south", "north"}` constant with a function `detect_lateral_patches(boundary_faces: dict) -> set[str]` that returns `{"lateral"}` if `"lateral"` is a key in `boundary_faces`, otherwise returns `{"west", "east", "south", "north"}` (box fallback). Replace the patch-gating guard in `init_from_era5()` with a call to this function. Existing CLI and function signatures unchanged.

**Success test:** `services/module2a-cfd/tests/test_init_lateral.py::test_detect_lateral_patches_octagonal` — call with `{"lateral": ..., "top": ..., "terrain": ...}`, assert returns `{"lateral"}`. `test_detect_lateral_patches_box` — call with box dict, assert returns `{"west","east","south","north"}`.

**Status:** done

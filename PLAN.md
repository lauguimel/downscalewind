# Plan: Reboot Module 2A for OF2412 ESI + kraken-sim
Generated: 2026-03-14
Test command: cd services/module2a-cfd && python -c "from generate_mesh import generate_mesh; print('OK')" && python -c "from generate_campaign import generate_campaign; print('OK')"

## Phase 1 — OF2412 ESI template headers + cleanup
**Files to modify:** `0/U.j2`, `0/k.j2`, `0/epsilon.j2`, `0/p.j2`, `0/nut.j2`, `system/fvSolution.j2`, `constant/fvOptions.j2`, `system/surfaceFeatureExtractDict.j2`
**Files to delete:** `system/surfaceFeaturesDict.j2`, `constant/turbulenceProperties` (static duplicate)
**What to implement:** Update all Foundation-style OF headers (`Version: 10`, `www.openfoam.org`) to ESI style (`v2412`, `www.openfoam.com`). Remove `{{ of_version }}` template variable from fvOptions.j2 header. Delete surfaceFeaturesDict.j2 (Foundation `surfaceFeatures` command; ESI uses `surfaceFeatureExtract` with surfaceFeatureExtractDict.j2 which already exists). Delete the static `constant/turbulenceProperties` file (will be replaced by the renamed .j2 in Phase 2).
**Headers already ESI (no change needed):** controlDict.j2, T.j2, alphat.j2, p_rgh.j2
**Success test:** `grep -rL "www.openfoam.com" services/module2a-cfd/templates/openfoam/{0,system,constant}/*.j2` returns empty (all .j2 files have ESI header). `ls services/module2a-cfd/templates/openfoam/system/surfaceFeaturesDict.j2` fails (deleted). `ls services/module2a-cfd/templates/openfoam/constant/turbulenceProperties` fails (deleted).
**Status:** done

## Phase 2 — Fix momentumTransport.j2 → turbulenceProperties.j2 + controlDict cleanup
**Files to modify:** `constant/momentumTransport.j2` (rename to `constant/turbulenceProperties.j2` + rewrite content), `system/controlDict.j2`, `Allrun.j2`
**What to implement:**
1. Rename `momentumTransport.j2` → `turbulenceProperties.j2`. Rewrite: remove `of_version` conditional, hardcode `object turbulenceProperties;`, use `RASModel kEpsilon;` (ESI syntax instead of Foundation `model kEpsilon;`), ESI header.
2. In `controlDict.j2`: remove the entire `functions {}` block (lines 70-115) — kraken-sim injects its own functionObjects for monitoring.
3. In `Allrun.j2`: change `runApplication surfaceFeatures` → `runApplication surfaceFeatureExtract` (ESI command name). The `runApplication checkMesh -latestTime` line already exists (line 45), keep it.
**Success test:** `python -c "from jinja2 import Environment, FileSystemLoader; e=Environment(loader=FileSystemLoader('services/module2a-cfd/templates/openfoam')); t=e.get_template('constant/turbulenceProperties.j2'); print(t.render())"` renders without error and contains `RASModel`. `grep -c 'functions' services/module2a-cfd/templates/openfoam/system/controlDict.j2` returns 0. `grep 'surfaceFeatureExtract' services/module2a-cfd/templates/openfoam/Allrun.j2` matches.
**Status:** done

## Phase 3 — Adapt generate_mesh.py (remove OF version logic)
**Files to modify:** `services/module2a-cfd/generate_mesh.py`
**What to implement:** Remove `of_version` parameter from `generate_mesh()` signature and from `jinja_ctx` dict. Remove the OF9 compatibility block (lines 598-603) that renames momentumTransport → turbulenceProperties. Remove `--of-version` CLI argument (line 679). Remove `of_version=args.of_version` from CLI call (line 718). The template is now `turbulenceProperties.j2` (renamed in Phase 2) and always renders as `turbulenceProperties` — no post-render rename needed.
**Success test:** `cd services/module2a-cfd && python -c "from generate_mesh import generate_mesh; print('OK')"` succeeds. `grep -c 'of_version' services/module2a-cfd/generate_mesh.py` returns 0.
**Status:** done

## Phase 4 — Adapt generate_campaign.py for kraken-sim + create run.pbs.j2
**Files to create:** `services/module2a-cfd/templates/openfoam/run.pbs.j2`
**Files to modify:** `services/module2a-cfd/generate_campaign.py`
**What to implement:**
1. Create `run.pbs.j2` — PBS job script template for kraken-sim. Variables: `case_id`, `solver.name`, `solver.n_cores`, `walltime`, `queue`. Loads OpenFOAM module, sources bashrc, runs `./Allrun $NCPUS`.
2. In `generate_campaign.py`: remove `of_version=9` from `generate_mesh()` call (line 333). Remove `_register_in_hpcsim()` function and all hpc-sim references (imports, `--db` CLI arg, db_path parameter). Replace JSON manifest with kraken-sim `campaign.yaml` output (YAML with case list + PBS template path). Each case entry has: case_id, solver, parameters, path.
**Success test:** `cd services/module2a-cfd && python -c "from generate_campaign import generate_campaign; print('OK')"` succeeds. `grep -c 'hpc.sim\|hpcsim\|of_version' services/module2a-cfd/generate_campaign.py` returns 0. `test -f services/module2a-cfd/templates/openfoam/run.pbs.j2`.
**Status:** done

## Phase 5 — Simplify workflow_module2.py + update docs
**Files to modify:** `notebooks/workflow_module2.py`, `CLAUDE.md`
**What to implement:**
1. Rewrite `workflow_module2.py`: keep only case generation + campaign.yaml output. Remove sections 0 (baseline ERA5), 5 (OpenFOAM Docker run), 6 (export), 7 (convergence subprocess), 8 (CFD vs obs), 9 (QC), 10 (batch PoC). Keep sections 1-4 (canonical case selection, terrain viz, mesh generation, inlet profile) and add a new final section that calls `generate_campaign()` to produce campaign.yaml. Remove `OpenFOAMRunner` imports.
2. Update `CLAUDE.md`: change "OF10 Foundation" references to "OF2412 ESI", remove hpc-sim mentions, add kraken-sim reference, update pipeline description.
**Success test:** `python -c "import ast; ast.parse(open('notebooks/workflow_module2.py').read()); print('OK')"` (valid Python). `grep -c 'OpenFOAMRunner\|run_cfd_batch\|export_cfd\|check_coherence' notebooks/workflow_module2.py` returns 0. `grep 'OF2412\|kraken' CLAUDE.md` matches.
**Status:** done

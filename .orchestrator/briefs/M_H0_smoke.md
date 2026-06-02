# Engineer brief — M_H0_smoke

## Mission
Validate the **fine-tune pipeline** for surrogate v2 ViT with an additional **OBS anchoring input channel** (option E.2, stage 1 = "E.1"). Fork architecture + dataset + training script, fine-tune **1 epoch** on **5–10 CFD cases** on Aqua H100, and run a **toggle test** proving the model uses the new channel.

The base checkpoint (`surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`, val_mse 0.1212) is **already excellent on CFD val**. The mission is not to beat it — only to prove the pipeline runs cleanly and the model can integrate the OBS channel without collapse. Full training is M_H1 (separate mission, gated on this smoke).

## Required reading (read in order)

1. `.orchestrator/mandate.md` §0, §1, §3 (constraints), §5 M_H0_smoke (lines 83–114)
2. `.orchestrator/memory/engineer.md` — esp. §"Surrogate v2 input dim formula", §"Zarr 3.1.5 create_dataset", §"Apptainer/weka paths", §"ERA5 d2m required"
3. On Aqua (read-only, do **not** modify):
   - `~/dsw/services/module2b-surrogate/src/model_vit_v2.py` (160 LOC, the architecture you fork)
   - `~/dsw/services/module2b-surrogate/src/model_vit.py` (shared blocks: `PatchEmbed2D`, `TransformerBlock`, `CrossAttentionBlock`, `ERA5TokenEncoder`, `_init_weights`)
   - `~/dsw/services/module2b-surrogate/src/dataset_v2_vit.py` (185 LOC, the dataset you fork)
   - `~/dsw/services/module2b-surrogate/src/dataset_v2.py` (helpers `DEFAULT_NORM`, `NI=NJ=180`, `NK=40`, `build_era5_baseline_tensor`, `parse_agl_levels`, `resample_volume_to_agl_levels`)
   - `~/dsw/services/module2b-surrogate/train_v2_vit.py` (the training driver you fork)
   - `~/dsw/configs/hpc/train_v2_vit_base_resid_s4_geo_agl100_k24_surface.pbs` (the reference PBS)
4. Local: `/Users/guillaume/Documents/Recherche/downscalewind/configs/hpc/train_v2_vit_base_resid_s4_geo_agl100_k24_surface.pbs` (mirror of the PBS for editing)

## Key facts to internalize before coding

- **Surrogate v2 grid**: `(180, 180, 40)` native, 33.333 m horizontal, 24 AGL levels in [0, 100] m for the surface checkpoint (`--target-agl-levels agl_0_100_24`).
- **ERA5 flat dim** = `4·N_p·9 + 4·9 + N_p + 2 = 408` for `N_p=10`. **The OBS channel is independent of `era5_flat_dim`** — keep ERA5 dim unchanged.
- **Surface checkpoint config** (from the reference PBS): `preset=base`, `use_geo=True`, `include_slopes=True` (so `terrain_in_channels=4`: terrain, slope_x, slope_y, z0_map), `use_residual=True`, `residual_baseline_mode=surface`, `agl_weight_alpha=2.0`, `agl_weight_height=50.0`, `target_agl_levels=agl_0_100_24` (nz=24).
- **Splits YAML**: `/scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_splits.yaml`. First train sites: `ct_c_morpho_0000`...`ct_c_morpho_0004`. Each site has `case_ts000`..`case_ts014` (up to 15 cases).
- **OBS observable**: 10 m AGL horizontal wind speed (`u10_obs = sqrt(u^2+v^2)` at AGL≈10 m). In the surface checkpoint with `agl_0_100_24`, the AGL coordinate is fixed: 24 levels uniformly in [0, 100] m → **level closest to 10 m is index ≈ 2** (verify via `parse_agl_levels('agl_0_100_24')`). Use the **horizontal wind speed in the normalized target** as the OBS value: `obs_value = sqrt(target[0]^2 + target[1]^2)` at `(i, j, k_10m)` (still in *normalized* space — keep everything in the model's input convention).
- **Zarr 3.1.5 + Apptainer**: use `realpath` (`os.path.realpath`) for `/scratch/maitreje/...` paths. Use `create_array` not `create_dataset`. (Not load-bearing here since we only read.)
- **No mpirun on login node**. No `qsub` >2h walltime. No CDS download. No `git commit` / `git push`.

## Engineer scope — deliverables

### 1. Forked architecture: `services/module2b-surrogate/src/model_vit_v2_e2.py` (NEW FILE)

- Copy `TerrainViT_V2_S3` from `model_vit_v2.py` into a new class `TerrainViT_V2_E2`.
- Add an OBS input channel. **You choose ONE of the following strategies** (A / B / C / D) and **justify the choice in ≤200 words** in the docstring of `TerrainViT_V2_E2`:
  - **A — Sparse additive token embedding (RECOMMENDED default)**: build a small MLP `phi(value, mask) → embed_dim`, identify the patch index containing the OBS pixel `(i, j)`, and add `phi(obs)` to that patch token after `patch_embed`. When mask=0, `phi` should output ≈0 (init the last linear layer's weight at small scale, or gate by the mask multiplicatively).
  - **B — Cross-attention injection**: append the OBS embedding as an extra key/value token alongside the ERA5 tokens in the cross-attention blocks.
  - **C — Concat + LayerNorm**: spatially broadcast (value, mask) as 2 extra channels of the terrain input (180×180) before `patch_embed` (set everywhere to NaN/0 except at `(i, j)`); requires bumping `terrain_in_channels` by 2.
  - **D — Other principled (FiLM gating, etc.)**: must be implementable in ≤80 LOC.
- The new channel weights must be **zero-init or small-init** so that, at init, `TerrainViT_V2_E2(...)` with `dropout=1` (no OBS) reproduces `TerrainViT_V2_S3(...)` (within float noise). Verify this in your toggle test.
- Constructor signature: keep the parent signature, add `obs_value_dim=1` (just speed for now), `obs_mask_dim=1`, plus whatever args your strategy needs.
- Forward signature: `forward(terrain, era5, geo=None, obs_value=None, obs_mask=None, obs_ij=None)` where `obs_ij: (B, 2) int` is the pixel coordinate.

### 2. Forked dataset: `services/module2b-surrogate/src/dataset_v2_vit_e2.py` (NEW FILE)

- Copy `WindV2DatasetViT` into a new class `WindV2DatasetViT_E2` (subclass or copy — your call, but **never edit the original**).
- Add to `__init__`: `obs_dropout=0.5`, `obs_height_m=10.0`, `obs_agl_level_idx=None` (auto-resolve from `target_agl_levels` and `obs_height_m`).
- In `__getitem__`, after computing the normalised `target`:
  - If `dropout` triggers (`np.random.rand() < self.obs_dropout`): emit `obs_value=0.0`, `obs_mask=0`, `obs_ij=(0,0)`.
  - Else: pick uniform random `(i, j) ∈ [0, 180)²`, compute `obs_value = sqrt(target[0, i, j, k_10m]² + target[1, i, j, k_10m]²)`, `obs_mask=1`, `obs_ij=(i, j)`.
- Return tuple includes the new fields in a deterministic order. Document the order in the docstring.
- The position k_10m is resolved once in `__init__` and stored as `self.obs_k`.

### 3. Training script: `services/module2b-surrogate/train_v2_e2.py` (NEW FILE)

- Fork `train_v2_vit.py`. Use `WindV2DatasetViT_E2` and `build_vit_v2_e2` (you may add a builder in `model_vit_v2_e2.py`).
- Accept new CLI args: `--resume <base.pt>` (mandatory — loads the surface checkpoint), `--obs-dropout` (default 0.5), `--smoke` (alias for `--max-train-cases 8 --max-val-cases 4 --epochs 1 --batch-size 2 --num-workers 0`).
- **Weight loading**: reuse the existing `load_resume_weights(model, ck, partial=True)` pattern. The new channel's weights have no match in the base checkpoint → they stay at their zero/small init. Log explicitly `loaded=X skipped=Y` so we can audit.
- MLflow logging: `mlflow.set_tracking_uri("file://" + str(out_dir / ".." / "mlruns"))`. If MLflow unavailable, fall back to history.yaml as the original. **Do not block on MLflow** — wrap in `try/except`.
- Save best checkpoint to `data/models/surrogate_v2_e2_stage1_smoke/best.pt`.

### 4. Toggle test script: `services/module2b-surrogate/eval_e2_smoke.py` (NEW FILE)

- Load `data/models/surrogate_v2_e2_stage1_smoke/best.pt`.
- Build the val dataset with the same 4 cases used for smoke val.
- Run inference TWICE on each val case:
  - **dropout=0** (canal always provided, OBS = synthetic from CFD truth at random pixel)
  - **dropout=1** (canal always masked)
- Report:
  - `MSE(pred_drop0, target)`, `MSE(pred_drop1, target)`. Expect `MSE_drop0 < MSE_drop1`.
  - `mean |pred_drop0 - pred_drop1|` (in normalized units). Expect > 1e-3 (proves outputs differ).
  - The same pixel where OBS was injected: print `pred_drop0[i,j,k_10m]` vs `pred_drop1[i,j,k_10m]` vs `target[i,j,k_10m]`. Expect pred_drop0 to be visibly closer to target than pred_drop1.
- Write a tiny summary YAML to `data/models/surrogate_v2_e2_stage1_smoke/toggle_test.yaml`.

### 5. PBS smoke script: `configs/hpc/finetune_e2_smoke.pbs` (NEW FILE)

- Walltime: **45 min** (≤2h hard limit). 1 H100, ncpus=8, mem=48GB.
- Mirror the `train_v2_vit_base_resid_s4_geo_agl100_k24_surface.pbs` env setup (Miniconda3 → conda activate fuxicfd → LD_LIBRARY_PATH nvidia libs).
- Call `python -u train_v2_e2.py ... --smoke --resume "$BASE"` with `$BASE` = surface checkpoint, all the surface-checkpoint config flags (`--use-geo --include-slopes --use-residual --residual-baseline-mode surface --agl-weight-alpha 2.0 --agl-weight-height 50.0 --target-agl-levels agl_0_100_24 --preset base --loss-type s4 --w-amp 0.1 --w-div 0.05`).
- After training, run `python eval_e2_smoke.py --checkpoint $OUT/best.pt --toggle-dropout` from the PBS.

### 6. Smoke training config: `configs/training/finetune_e2_stage1_smoke.yaml` (NEW FILE)

A YAML mirror of the CLI args (so the user can read the run config at a glance). Not consumed by the script unless trivial. Document the choice of OBS injection strategy here too.

## Allowed edit zones (STRICT — do not touch anything else)

- `services/module2b-surrogate/src/model_vit_v2_e2.py` (NEW)
- `services/module2b-surrogate/src/dataset_v2_vit_e2.py` (NEW)
- `services/module2b-surrogate/train_v2_e2.py` (NEW)
- `services/module2b-surrogate/eval_e2_smoke.py` (NEW)
- `configs/training/finetune_e2_stage1_smoke.yaml` (NEW)
- `configs/hpc/finetune_e2_smoke.pbs` (NEW)
- `data/models/surrogate_v2_e2_stage1_smoke/` (output, smoke checkpoint)
- `test/scratch/`, `scratch/`, `tmp/` (free zone for debug)

## Forbidden actions (STRICT)

- **No destructive modification of `model_vit_v2.py`, `model_vit.py`, `dataset_v2_vit.py`, `dataset_v2.py`, `train_v2_vit.py`** — fork only by copy + rename.
- No full training on the 9252 cases. Smoke is hard-capped at **≤10 train cases + ≤4 val cases**.
- No `git commit`, no `git push`, no `git tag`.
- No `qsub` with walltime > 2h.
- No `mpirun` on the login node.
- No CDS download (ERA5 is already in the grid.zarr stores).
- No `rm` on any untracked file without first running `grep -r '<filename>' services/ configs/ shared/` to verify nothing imports it.

## Validation workflow — TWO STAGES

**Stage A (your job)**: implement, scp to Aqua, run static checks, submit PBS, return immediately.

**Stage B (Department reviewer's job, NOT yours)**: poll qstat, retrieve logs, verify exit criterion, run toggle test if PBS didn't.

### Stage A commands (you run these — must complete in ≤25 min total)

```bash
# 1. Local: confirm files exist and parse
cd /Users/guillaume/Documents/Recherche/downscalewind
ls services/module2b-surrogate/src/model_vit_v2_e2.py \
   services/module2b-surrogate/src/dataset_v2_vit_e2.py \
   services/module2b-surrogate/train_v2_e2.py \
   services/module2b-surrogate/eval_e2_smoke.py \
   configs/hpc/finetune_e2_smoke.pbs \
   configs/training/finetune_e2_stage1_smoke.yaml
python -m py_compile services/module2b-surrogate/src/model_vit_v2_e2.py \
                     services/module2b-surrogate/src/dataset_v2_vit_e2.py \
                     services/module2b-surrogate/train_v2_e2.py \
                     services/module2b-surrogate/eval_e2_smoke.py

# 2. scp to Aqua
scp services/module2b-surrogate/src/model_vit_v2_e2.py \
    services/module2b-surrogate/src/dataset_v2_vit_e2.py \
    maitreje@aqua:~/dsw/services/module2b-surrogate/src/
scp services/module2b-surrogate/train_v2_e2.py \
    services/module2b-surrogate/eval_e2_smoke.py \
    maitreje@aqua:~/dsw/services/module2b-surrogate/
scp configs/hpc/finetune_e2_smoke.pbs \
    maitreje@aqua:~/dsw/configs/hpc/
scp configs/training/finetune_e2_stage1_smoke.yaml \
    maitreje@aqua:~/dsw/configs/training/

# 3. Static checks on Aqua (no GPU needed; use login node Python OR submit a 1-min test job)
ssh maitreje@aqua "cd ~/dsw/services/module2b-surrogate && python3 -c \"
import ast
for f in ['src/model_vit_v2_e2.py','src/dataset_v2_vit_e2.py','train_v2_e2.py','eval_e2_smoke.py']:
    ast.parse(open(f).read()); print(f, 'parses OK')
\""

# 4. Submit smoke job
ssh maitreje@aqua "cd ~/dsw && mkdir -p data/models/surrogate_v2_e2_stage1_smoke && qsub configs/hpc/finetune_e2_smoke.pbs"
# capture the job id (e.g. 20712345.aqua)

# 5. Confirm queued
ssh maitreje@aqua "qstat -u maitreje | tail -5"

# 6. Local diff cleanliness
cd /Users/guillaume/Documents/Recherche/downscalewind
git status --short    # only the allowed edit zones must appear
git diff --check
```

**STOP HERE.** Do NOT wait for the PBS to finish. Print the report and exit. The Department will poll the job and report the verdict.

## Exit criterion (Engineer's contract — Stage A only)

1. All 6 new files exist locally and on Aqua, all parse with `py_compile` / `ast.parse`.
2. PBS job submitted successfully (job id captured in your report).
3. The PBS chains training **and** the toggle test (`python eval_e2_smoke.py ...`) so the Department only needs to retrieve logs.
4. `git diff --check` clean.
5. No untracked file outside the allowed edit zones.

The toggle test PASS/FAIL verdict belongs to the Department after the PBS completes — **not your responsibility**.

## Expected report ≤200 words

Format your final stdout report as:

```
== M_H0_smoke ENGINEER REPORT (Stage A) ==
Strategy: <A/B/C/D> — <1 sentence rationale>
LOC delta: <total_new_loc> across 6 new files
  - model_vit_v2_e2.py: <N>
  - dataset_v2_vit_e2.py: <N>
  - train_v2_e2.py: <N>
  - eval_e2_smoke.py: <N>
  - finetune_e2_smoke.pbs: <N>
  - finetune_e2_stage1_smoke.yaml: <N>
Static checks: OK / FAIL <details>
PBS job: <jobid>.aqua submitted at <HH:MM>
Init check: model loads <X> weights from base, skips <Y> (new OBS channel params)
Caveats: <≤3 bullets>
```

Keep it tight. Do NOT include smoke metrics or toggle test — those come from the Department after PBS completion.

## Notes from memory (cross-cutting)

- The original surface checkpoint config (verbatim from PBS) is the contract — every flag matters; do not silently drop one.
- The dataset returns tuples whose order depends on flags (`return_geo`, `return_weight`). In the E2 dataset, **stick to the parent's order** for the original fields, then **append OBS fields at the end** before the `case_dir.name`. Update `unpack_batch` in `train_v2_e2.py` accordingly.
- Conda env on Aqua: `fuxicfd` (NOT `downscalewind` — the local env is irrelevant; everything runs on Aqua).
- LD_LIBRARY_PATH dance for CUDA libs is mandatory inside the PBS (cf. reference PBS).
- The fork files live under `src/` (matching the layout of `src/model_vit_v2.py`); the training driver lives at module root (matching `train_v2_vit.py`).

## When you're done

Print your ≤200-word report. Do NOT commit. The Department reviews and reports to the Boss.

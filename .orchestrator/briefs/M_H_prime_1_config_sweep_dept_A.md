# Engineer brief — M_H'1 config sweep (Department A, empirical)

## Mission ID

`M_H'1_config_sweep_dept_A` — adversarial dual-spawn with Department B (theoretical).
Do NOT coordinate with B. You report directly to Department A boss; Department A reports to project Boss.

## Context (condensed — full mandate at `.orchestrator/mandate.md` §5 M_H'1)

Phase H' = DEVINE-style training : small ANN corrects ERA5 input → frozen surrogate v2 forward → loss at central pixel (station). Smoke v3 (200 stations × winter2223 × WC enabled × 6 epochs, killed at walltime) GREEN baseline :

```
ep=0  train_mae=1.498 | val_mae=1.405  val_bias=+0.449 | RAW mae=1.602  bias=-0.796 | Δmae=-0.196
ep=1  train_mae=1.432 | val_mae=1.486  val_bias=+0.534 |                            | Δmae=-0.116   ← OVERSHOOT
ep=2  train_mae=1.421 | val_mae=1.407  val_bias=+0.533 |                            | Δmae=-0.195
ep=3  train_mae=1.413 | val_mae=1.407  val_bias=+0.520 |                            | Δmae=-0.194
ep=4  train_mae=1.408 | val_mae=1.412  val_bias=+0.518 |                            | Δmae=-0.190
ep=5  train_mae=1.399 | val_mae=1.417  val_bias=+0.495 |                            | Δmae=-0.185
```

Two suspected issues to probe :

1. **Epoch 1 val_mae overshoot** : 1.405 → 1.486 (+0.081 ≈ 3.2σ vs nominal noise). Hypothesis : `lr=1e-3` too aggressive for a 26k-param ANN with zero-init last layer ; the first ~6k updates step the model into a region far from the zero-delta basin before settling.
2. **val_bias overshoot** : raw −0.796 → corrected +0.495. ANN over-corrects upward. Hypothesis : τ asymmetric 0.6/0.4 (penalty stronger when `obs <= pred`) pushes the model too far above the OBS distribution mean.

## Mission scope — sweep matrix

Run **4 variants** on the **same val set** as smoke v3 so Δmae is comparable :

| Variant | `lr` schedule | τ (under/over) | grad_clip | Hypothesis |
|---|---|---|---|---|
| **V0_baseline** | constant 1e-3 | 0.6 / 0.4 | none | sanity check ; reproduce v3 exactly with cache reuse |
| **V1_warmup** | linear warmup 1e-4 → 1e-3 over **first epoch** then cosine to 1e-4 over remaining 4 epochs | 0.6 / 0.4 | none | fix epoch 1 overshoot |
| **V2_low_lr_clip** | constant 5e-4 | 0.6 / 0.4 | **1.0** | gentler optim — lr halved + grad clipping |
| **V3_tau_sym** | constant 1e-3 | **0.55 / 0.45** | none | reduce upward over-correction (intermediate vs paper asymmetric vs neutral 0.5/0.5) |

Total compute : 4 × 5 epochs × ~33 min/epoch ≈ 11h sériel ; **submit 4 jobs in parallel on Aqua H100** → wall ≈ 3.5h.

## Inputs

- Surrogate v2 frozen ckpt : `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
- Pairings parquet : `~/dsw/data/inference/noaa_winter2223.parquet`
- ERA5 store : `~/dsw/data/raw/era5_europe_winter2223.zarr`
- DEM tiles : `~/dsw/data/raw/srtm_tiles/`
- WorldCover tiles : `~/dsw/data/raw/worldcover_esa/`
- Norm YAML : `/scratch/maitreje/dsw/complex_terrain_v1/manifests/dataset_v2_norm.yaml`
- Cache (reusable) : `/scratch/maitreje/dsw/phase_H_prime_smoke_cache` — populated by smoke v3 ; reuse via `overwrite_cache: false` to save ~1h materialisation per variant
- Current trainer : `services/module2b-surrogate/train_v2_devine_style.py`
- Current ANN+loss : `services/module2b-surrogate/src/ann_correction.py`
- Current smoke YAML : `configs/training/devine_style_smoke.yaml`
- Current smoke PBS : `configs/hpc/devine_style_smoke.pbs`

## Required reading (mandatory before any code change)

1. `.orchestrator/mandate.md` — focus §0 (état), §5 (M_H'1), §4 (ADRs)
2. `.orchestrator/memory/boss.md` — `## M_H'0 smoke v3 GREEN 2026-05-29` section
3. `.orchestrator/memory/department.md` — `## Dry-run 1-batch CPU avant qsub Aqua (M_H'0 lesson)` — **MANDATORY pattern for any code change**
4. `.orchestrator/memory/engineer.md` — `qstat -x Time Use = CPU time` pattern + Aqua paths
5. `services/module2b-surrogate/train_v2_devine_style.py` — full file ; focus on `main()` optimizer build (line ~358), `_step` function, `devine_speed_loss`
6. `services/module2b-surrogate/src/ann_correction.py` — `devine_speed_loss` signature ; pay attention to the `tau_under` / `tau_over` kwargs (already parametrisable)
7. `configs/training/devine_style_smoke.yaml` — current config to clone
8. Aqua smoke v3 train.log : `ssh maitreje@aqua "grep 'ep=' ~/dsw/data/models/surrogate_v2_devine_smoke_v2/train.log"` — already extracted above for reference

## Tasks

### Task 1 — Minimal trainer extension (optional, only if needed)

The current trainer uses `torch.optim.Adam(ann.parameters(), lr=cfg['lr'])` with no scheduler and no grad clip ; `devine_speed_loss(tau_under, tau_over)` is already parametrisable but the trainer hardcodes the call with defaults.

**Add three optional config knobs in `train_v2_devine_style.py`** (backward-compatible — defaults reproduce current behaviour exactly) :

1. `lr_schedule`: one of `"constant"` (default), `"warmup_cosine"`
   - When `"warmup_cosine"` : `warmup_epochs` (default 1) linear from `lr_min_warmup=1e-4` to `lr`, then cosine decay to `lr_final=1e-4` over remaining epochs.
   - Use `torch.optim.lr_scheduler.SequentialLR` with `LinearLR` then `CosineAnnealingLR` ; step per batch.
2. `grad_clip_norm`: float or null (default `null`) → if set, `torch.nn.utils.clip_grad_norm_(ann.parameters(), grad_clip_norm)` after `loss.backward()` before `optimizer.step()`
3. `tau_under` / `tau_over`: floats (defaults `0.6` / `0.4`) → passed through to `devine_speed_loss` in `_step`

Constraints :
- Keep the existing `_step` signature ; thread tau through via partial or kwargs ; do NOT break the cached `ann.train()/eval()` logic.
- Log per-epoch the current `lr` (read `optimizer.param_groups[0]['lr']` AFTER scheduler.step) and append to `history.yaml` as `lr_end_epoch`.
- Total LOC delta target ≤ 60.
- Verify with **dry-run CPU 1-batch** (cf. department.md lesson) **before any qsub** : load V1 config, build datasets (using a tiny `max_train_pairings=8`, `max_val_pairings=4` override via a CLI flag OR a separate `_dryrun.yaml` config), run 1 train + 1 val step, assert no crash, assert scheduler advanced lr. Document the dry-run command in your report.

### Task 2 — 4 variant configs + 4 PBS scripts

Create :
- `configs/training/devine_style_smoke_A_V0_baseline.yaml`
- `configs/training/devine_style_smoke_A_V1_warmup.yaml`
- `configs/training/devine_style_smoke_A_V2_low_lr_clip.yaml`
- `configs/training/devine_style_smoke_A_V3_tau_sym.yaml`
- `configs/hpc/devine_style_smoke_A_V0_baseline.pbs`
- `configs/hpc/devine_style_smoke_A_V1_warmup.pbs`
- `configs/hpc/devine_style_smoke_A_V2_low_lr_clip.pbs`
- `configs/hpc/devine_style_smoke_A_V3_tau_sym.pbs`

Each YAML should clone `devine_style_smoke.yaml` then override :
- `output_dir: /home/maitreje/dsw/data/models/surrogate_v2_devine_A_V<X>_<name>`
- `cache_dir: /scratch/maitreje/dsw/phase_H_prime_smoke_cache` (shared — DO NOT change this path ; all variants must hit the same cache to keep the val set identical and save 4× materialisation)
- `overwrite_cache: false` (reuse smoke v3 cache)
- `epochs: 5` (capped per budget)
- `max_train_pairings: 25000`, `max_val_pairings: 6000`, `max_stations: 200` — **DO NOT CHANGE** so val set is comparable
- Variant-specific overrides per matrix above

PBS scripts : clone `devine_style_smoke.pbs`, change `#PBS -N`, change `CFG` path, change `OUT` path. **walltime=04:00:00** (margin vs 3.5h target).

### Task 3 — Submit 4 jobs in parallel

```bash
ssh maitreje@aqua "
cd ~/dsw
qsub configs/hpc/devine_style_smoke_A_V0_baseline.pbs
qsub configs/hpc/devine_style_smoke_A_V1_warmup.pbs
qsub configs/hpc/devine_style_smoke_A_V2_low_lr_clip.pbs
qsub configs/hpc/devine_style_smoke_A_V3_tau_sym.pbs
"
```

Then monitor via `qstat -u maitreje` periodically (do NOT busy-poll : 30 min interval is enough). Estimated total wall ≈ 3.5h.

### Task 4 — Collect + analyse

For each variant, once `train.log` is available :

```bash
ssh maitreje@aqua "grep 'ep=' ~/dsw/data/models/surrogate_v2_devine_A_V<X>_*/train.log"
ssh maitreje@aqua "cat  ~/dsw/data/models/surrogate_v2_devine_A_V<X>_*/history.yaml"
```

Build a comparison table with these columns per variant :
- `final_val_mae` = `min(val_mae)` across epochs (best)
- `Δmae` = `final_val_mae - val_mae_raw` (val_mae_raw should be 1.602 if same val set)
- `val_bias_final` = bias at the best epoch
- `val_mae_std` = std dev of val_mae across all 5 epochs (stability proxy)
- `overshoot_ep1` = `val_mae[1] - val_mae[0]` (positive = overshoot)
- `n_epochs_ran` (in case any variant hit walltime kill)

Rank variants by composite score :
1. Best Δmae (most negative wins)
2. Tie-break by `val_mae_std` (smaller = more stable)
3. Tie-break by `|val_bias_final|` (smaller absolute = less bias)

### Task 5 — Recommend a config for M_H'1 full

Based on ranking, pick ONE variant and write a **recommended config fragment** for M_H'1 full :

```yaml
# Recommended for M_H'1 full (264k pairings, 10 epochs)
lr: <value>
lr_schedule: <constant|warmup_cosine>
warmup_epochs: <int>
lr_min_warmup: <value>
lr_final: <value>
grad_clip_norm: <value or null>
tau_under: <value>
tau_over: <value>
batch_size: <recommend value for A100/H100 ; baseline used 4>
epochs: 10
```

Justify each non-default value with a numerical comparison from the sweep table. Mention any caveat (e.g. "V2 had best Δmae but lowest stability — recommend V1 if reproducibility critical").

## Allowed edit zones

- `configs/training/devine_style_smoke_A_*.yaml` (4 new files)
- `configs/hpc/devine_style_smoke_A_*.pbs` (4 new files)
- `services/module2b-surrogate/train_v2_devine_style.py` (only the 3 optional knobs — defaults must reproduce baseline)
- `data/models/surrogate_v2_devine_A_*/` (output dirs, on Aqua)
- `data/validation/phase_H_prime_config_sweep/` (synthesis report only — no figures unless trivial)
- `test/scratch/`, `scratch/`, `tmp/` if needed for dryrun
- `.orchestrator/briefs/M_H_prime_1_config_sweep_dept_A_dryrun_log.md` (optional — log dry-run output)

## Forbidden actions

- NO modification of `model_vit_v2.py`, `dataset_v2_vit.py`, `dataset_v2_obs_centered.py`, `ann_correction.py`
- NO modification of surrogate v2 base (`best.pt`, model code)
- NO M_H'1 full training during this sweep (sweep only on smoke)
- NO commit / push without Department/Boss approval — leave changes uncommitted, report list of modified files
- NO qsub walltime > 5h without escalation
- NO changes to `max_stations`, `max_train_pairings`, `max_val_pairings`, `seed`, `val_frac` — these define the val set ; if you change them Δmae is no longer comparable
- NO changes to `overwrite_cache` to true (would invalidate the 1h saving and risk cache corruption across parallel jobs)
- NO changes to `cache_dir` per variant — all 4 must SHARE `/scratch/maitreje/dsw/phase_H_prime_smoke_cache`

## Exit criteria

1. Dry-run CPU 1-batch passes for at least one variant (logged)
2. 4 variants submitted and tracked on Aqua (capture job IDs)
3. 4 variants complete (or have a documented RED reason if not — e.g. walltime, OOM)
4. Per-variant metrics tabulated as described in Task 4
5. Ranking computed with composite score
6. Single config recommendation with numerical justification (YAML fragment)
7. Department report ≤ 300 words to Department A boss (you write to me)

## Report format (≤ 300 words)

Send back a structured markdown report containing :

- **Mission ID** : M_H'1 config sweep (Dept A)
- **Status** : GREEN / YELLOW / RED
- **Code change summary** : LOC delta in `train_v2_devine_style.py`, dry-run command + status
- **Variants table** : 4 rows with the 6 metrics from Task 4
- **Best variant** with Δmae, bias, overshoot_ep1
- **Recommended config for M_H'1 full** : YAML fragment (≤ 15 lines)
- **Caveats observed** : anything weird (NaN, plateau, OOM, walltime, etc.)
- **Memory candidates** (≤ 3) : patterns worth persisting for future engineers (e.g. "warmup scheduler reduced overshoot from X to Y on Phase H' DEVINE setup")

## Compute / budget

- ETA wall ~3.5h if 4 jobs run in parallel on Aqua H100, ~14h if serial
- File budget : ≤ 60 LOC added to trainer + 8 small config files ≈ 200 LOC total
- Walltime per job : 4h (margin)
- No commit ; no push ; uncommitted changes will be staged for Department review

## Communication

FR with user. EN code/commits/messages. NEVER mention Claude/AI/LLM in commits or PRs (will not commit anyway per scope).

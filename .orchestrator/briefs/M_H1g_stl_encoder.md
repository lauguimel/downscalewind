# DEPARTMENT BRIEF — M_H'1g-stl : STL-encodeur CNN dans l'ANN (gate Perdigão), JJA

## 0. CONTEXTE — le levier principal identifié
M_H'1f a prouvé empiriquement que les features SCALAIRES plafonnent sur Perdigão
(over-correction 2.27→2.15 seulement, vs raw 1.37). Diagnostic : le surrogate frozen
sous-estime structurellement les crêtes (terrain raide jamais vu, Pop B exclue RANS), et un
MLP qui ne voit que des résumés scalaires (mean/std/grad_T) ne peut PAS apprendre une
correction POSITION-dépendante pour compenser. Idée user = donner à l'ANN le TERRAIN COMPLET
180×180 via un encodeur CNN → il apprend (par backprop à travers le surrogate frozen) à
pré-amplifier l'ERA5 là où le relief est raide. Ce Department teste ce levier EN ISOLATION
sur JJA (cache existant, features M_H'1f), EN PARALLÈLE de M_H'1g-ingest (ssrd/tcc).

## 1. Objectif
Ajouter un encodeur CNN du `terrain_2d (4,180,180)` à `ANNCorrection`, dont la sortie latente
est concaténée aux (era5_flat + topo_features). Re-train JJA (cache 60971 existant), re-éval
Perdigão. Chiffre clef = MAE centre Perdigão vs 2.27 (M_H'1f) / 1.37 (raw).

## 2. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
  (Codex SANS SSH Aqua → Department scp+qsub ; PBS post-train eval cache aussi le dataset ;
   override store ERA5 par période ; dry-run avant qsub)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/src/ann_correction.py`
  (ANNCorrection actuel = MLP pur era5_flat+topo → à étendre AVEC encodeur CNN, rétrocompat)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/src/dataset_v2_obs_centered.py`
  (le dataset charge DÉJÀ terrain_2d (4,180,180) pour le surrogate via load_grid_inputs L110 ;
   il faut le faire passer AUSSI à l'ANN — vérifier le collate `_collate` L~378 qui renvoie
   `terrain, era5, geo, topo, speed, k_obs, meta`)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/train_v2_devine_style.py`
  (le `_step` L213 appelle ann(era5_flat, topo) puis surrogate ; il faut passer terrain à l'ann
   quand l'encodeur est activé ; instanciation ANNCorrection ~L260-300)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/eval_devine_loso.py`
  + `services/validation/audit_devine_perdigao.py` (doivent passer terrain à l'ann aussi)
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/training/devine_style_M_H1f.yaml`
  (config M_H'1f à cloner ; topo_dim:12, enable_phys_features:true)
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/hpc/devine_style_M_H1f.pbs`
  (PBS retrain à cloner)

## 3. Design encodeur (le cœur)
- `ANNCorrection` gagne un flag `use_terrain_encoder: bool=False` (rétrocompat : défaut False →
  M_H'1f/M_H'1a best.pt restent chargeables).
- Quand True : petit CNN (ex 3-4 conv stride-2 : 4→16→32→64 ch + global avg pool → latent ~32-64d)
  sur terrain_2d (4,180,180). GARDER léger (~50k-200k params) pour limiter overfitting (cf.
  caveat boss.md : on quitte le tiny MLP). latent concaténé à [era5_flat, topo_features].
- `forward(era5_flat, topo_features, terrain=None)` : si encodeur actif, terrain requis.
- Zero-init de la dernière couche conservé (delta=0 au départ → identité, trajectoire DEVINE).
- Surveiller OVERFITTING : reporter gap train↔val MAE (si val >> train → encodeur trop gros).

## 4. Strategy
- Config `devine_style_M_H1g.yaml` = clone M_H1f + `use_terrain_encoder: true`,
  `terrain_latent_dim: 48` (ou justifié), output_dir `surrogate_v2_devine_M_H1g_stl`.
  Reste identique (lr 5e-4, grad_clip 1.0, τ 0.6/0.4, epochs 8, bs 4, topo_dim 12, phys feats,
  val_frac 0.2, seed 42, exclude perdigao, cache JJA, overwrite_cache false).
- PBS `devine_style_M_H1g.pbs` (clone, H100, walltime 16h — encodeur = train + lent).
- dry-run CPU 1-batch : terrain (B,4,180,180) → latent → ann forward OK, pas NaN.
- re-train JJA → re-éval LOSO (val mae, doit rester ≤ ~1.25) + re-éval Perdigão (MAE centre).

## 5. Allowed edit zones
- `services/module2b-surrogate/src/ann_correction.py` (modify : encodeur CNN, additif flag)
- `services/module2b-surrogate/train_v2_devine_style.py` (modify : passer terrain à l'ann)
- `services/module2b-surrogate/eval_devine_loso.py`, `services/validation/audit_devine_perdigao.py`
  (modify : passer terrain à l'ann)
- `configs/training/devine_style_M_H1g.yaml`, `configs/hpc/devine_style_M_H1g.pbs`,
  `configs/hpc/eval_perdigao_M_H1g.pbs` (new)
- `data/models/surrogate_v2_devine_M_H1g_stl/`, `data/validation/phase_H_prime_perdigao_M_H1g/`
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`
- Aqua : scp scripts/configs + écriture models/validation

## 6. Forbidden
- ZÉRO re-matérialisation (cache 60971 ; terrain déjà dans grid.zarr).
- Rétrocompat : défaut use_terrain_encoder=False → M_H'1f/M_H'1a best.pt chargeables.
- PAS de modif model_vit_v2.py / dataset_v2_vit.py / surrogate v2 best.pt / M_H'1f best.pt /
  M_H'1a best.pt / train split logic.
- PAS de ssrd/tcc ici (c'est M_H'1g-ingest en parallèle ; ce gate isole l'encodeur seul).
- PAS de commit/push. qsub ≤16h.
- Perdigão = test set immuable, READ-ONLY.

## 7. Exit criterion
- ANN avec encodeur : dry-run CPU OK (terrain (B,4,180,180)→latent→forward, pas NaN).
- re-train JJA exit 0, best.pt produit ; val mae_corrected ≤ ~1.25 (NOAA pas dégradé) ;
  reporter gap train↔val (overfitting check).
- **re-éval Perdigão : MAE centre vs 2.27 (M_H'1f) et vs 1.37 (raw)** → baisse nette = encodeur
  est le levier ; effet faible = escalate (l'étage B surrogate est le vrai mur).
- cache 60971 inchangé. Diff zones autorisées only.

## 8. Runner + ops
`codex` via run-engineer.sh, single-spawn. Codex SANS SSH Aqua → Engineer livre code + dry-run
CPU local ; Department scp + qsub (~10-14h, ne pas attendre in-turn) + récup. Course A100+H100
si file congestionnée (kill loser on start, MÊME output dir). Write `.engineer_brief.md` PUIS
run-engineer.sh (séparé). SSH conda : `module load Miniconda3/24.9.2-0; eval "$(conda shell.bash
hook)"; conda activate fuxicfd`. Shell stdout vide → rediriger /tmp + Read. train.log NUL-padded
→ `grep -a | tr -d '\000'`.

## 9. Report (≤300 mots)
- Archi encodeur (couches, latent dim, params totaux ANN), rétrocompat OK.
- dry-run : terrain→latent→forward OK.
- Si run fini : val-43 mae_corrected (vs 1.213 M_H'1f, NOAA ok ?), gap train↔val (overfit ?),
  **Perdigão MAE centre (vs 2.27 / 1.37)** → verdict encodeur GO/NO-GO.
- Sinon : "code done + scp'd + job <ID>" pour Boss hpc-watch + Perdigão re-éval pending.
- cache 60971. Job ID. proposals (≤3) + memory candidates (≤3). Pas de commit.

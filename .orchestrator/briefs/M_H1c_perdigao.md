# DEPARTMENT BRIEF — M_H'1c : propagation spatiale Perdigão IOP (voisins intra-patch)

## 0. CONTEXTE
M_H'1a validé : vitesse −32.5% (val 1.223 vs raw 1.812, bat ERA5 1.315, honnête station-disjoint
LOSO) ; direction non dégradée (améliorée, 36° vs raw 41°, ≈ERA5 en vent fort). Dernier verrou
avant déploiement : la correction, calibrée sur des stations ISOLÉES (1 pixel central par patch),
se propage-t-elle de façon PHYSIQUE aux pixels voisins, ou crée-t-elle un pic artificiel au seul
pixel calibré ? Perdigão IOP (41 tours dans 6×6 km) est le seul dataset qui le teste.

**Décision design user 2026-06-02** :
- Méthode = **voisins intra-patch** : pour chaque station, comparer pred corrigée au pixel
  central (90,90) vs pixels voisins (90±k) du MÊME patch → la correction doit varier de façon
  LISSE, pas un pic 1-pixel. (PAS de cross-station inter-patch pour cette mission.)
- Période = **IOP mai-juin 2017 complet** (test set immuable du projet).

## 1. Mission ID + type
- **ID**: M_H'1c · **Type**: Validate (forward + analyse spatiale, ~court GPU)

## 2. Prérequis VÉRIFIÉS par le Boss (ne pas re-checker)
- `perdigao_obs.zarr` : **LOCAL** (12 MB, 41 tours), **ABSENT d'Aqua** → le Department doit le
  scp : `scp -r /Users/guillaume/Documents/Recherche/downscalewind/data/raw/perdigao_obs.zarr
  maitreje@aqua:~/dsw/data/raw/perdigao_obs.zarr`.
- ERA5 `era5_europe_spring2017_v2.zarr` sur Aqua : couvre 2017-05-01→06-30 (IOP inclus), d2m
  présent ✅. ⚠️ **Δt=360min (6h)** → predictions constantes par blocs 6h (caveat M_G7). OK pour
  test SPATIAL (on compare pixels au même timestamp), mais l'audit ne doit PAS interpréter la
  variabilité temporelle. Tracer via era5_time_delta.
- DEM `srtm_tiles/` 186 tiles ✅, `worldcover_esa/` ✅.
- ANN M_H'1a : `~/dsw/data/models/surrogate_v2_devine_M_H1_jja/best.pt` (epoch 5).
- surrogate frozen : `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`.
- `data/inference/stations_perdigao/` (local) contient déjà des inputs M_G6 extraits (rne01…)
  — vérifier s'ils sont réutilisables OU re-extraire via le pipeline standard.

## 3. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
  (lessons : Codex sandbox SANS SSH Aqua → Department fait scp+qsub ; dry-run avant qsub ;
   un seul appariement obs ; perdigao exclu du training via exclude_substrings)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/eval_devine_loso.py`
  (réutiliser le forward ANN+surrogate, extraction pixel central — adapter pour extraire AUSSI
   les voisins 90±k au lieu du seul (90,90))
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/extract_v2_input_at_coords.py`
  + `utils/inference_input.py` (pipeline (lat,lon,ts)→grid.zarr/input, M_G6)
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/hpc/eval_devine_loso_h100.pbs`
  (PBS H100 récent qui marche — cloner pour Perdigão)
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/training/devine_style_full_M_H1.yaml`
  (résoudre surrogate_preset, norm_yaml, target_agl_levels, n_pressure_levels)

## 4. Deliverable
Script `services/validation/audit_devine_perdigao.py` (new) qui :
1. Charge ANN M_H'1a + surrogate frozen (terrain_in_channels=**4**).
2. Pour les 41 tours Perdigão × timestamps IOP : construit l'input grid.zarr patch centré
   (via le pipeline M_G6), forward ANN→surrogate frozen → champ corrigé ET raw (use_ann=False).
3. **Extraction multi-pixel** : au lieu du seul (90,90), extraire un voisinage (90±k, k∈{0,1,2,3,5})
   au niveau 10 m AGL → `speed_corr[i,j]`, `speed_raw[i,j]`, `u,v` idem.
4. **Métrique propagation (intra-patch, le cœur)** :
   - `delta_correction[i,j] = speed_corr[i,j] − speed_raw[i,j]` sur le voisinage.
   - mesurer la **régularité spatiale** : gradient/écart-type de `delta_correction` autour du
     centre. Une correction physique = lisse (Δ varie peu sur 90±3). Un pic artificiel =
     |Δ(centre) − Δ(voisins)| grand.
   - ratio `|delta(90±k)| / |delta(centre)|` par anneau k → doit rester ~O(1), pas s'effondrer.
5. **Justesse au centre** : MAE/biais vitesse corrigée vs obs au pixel station (sanity : la
   correction doit aussi améliorer Perdigão, pas seulement NOAA — domaine différent).
6. Sorties dans `data/validation/phase_H_prime_perdigao/` :
   - `perdigao_propagation.csv` : par (station, timestamp) → delta_centre, delta_ring_k,
     smoothness_metric, speed_corr_centre, speed_raw_centre, speed_obs.
   - `perdigao_summary.json` : agrégé — MAE centre corr vs raw vs obs ; distribution de la
     smoothness ; % cas où correction = lisse vs pic.
   - 1 figure : heatmap `delta_correction` moyenne sur le voisinage (visuel propagation).

## 5. Allowed edit zones
- `services/validation/audit_devine_perdigao.py` (new)
- `configs/hpc/eval_perdigao.pbs` (new, clone H100)
- `data/validation/phase_H_prime_perdigao/` (sorties)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`
- Aqua : scp perdigao_obs.zarr + script + PBS ; écriture sous data/validation/phase_H_prime_perdigao/

## 6. Forbidden
- PAS de modif model_vit_v2.py / dataset_v2_vit.py / surrogate best.pt / M_H'1a best.pt.
- PAS de re-train. PAS de qsub >2h. PAS de commit/push.
- Perdigão = test set immuable → READ-ONLY, ne jamais l'inclure dans un quelconque training.
- Si grid.zarr inputs Perdigão doivent être matérialisés : OK (c'est un nouveau test set, pas
  le cache JJA) mais dans un dossier dédié `/scratch/maitreje/dsw/phase_H_prime_perdigao_cache`,
  NE PAS toucher au cache JJA (60971).

## 7. Exit criterion
- `perdigao_propagation.csv` (non vide, 41 stations × IOP timestamps, colonnes voisinage).
- `perdigao_summary.json` : MAE centre corr/raw/obs + métrique smoothness agrégée + % lisse.
- Verdict propagation : la correction est-elle LISSE autour du pixel station (physique) ou
  un PIC 1-pixel (overfit du pixel calibré) ?
- Diff zones autorisées only. Cache JJA inchangé (60971).

## 8. Runner + validation
`codex` via run-engineer.sh, single-spawn. ⚠️ Codex sandbox SANS SSH Aqua → Engineer livre le
CODE + dry-run CPU local ; **le Department fait scp (obs+script+PBS) + qsub + récup résultats**.
Dry-run CPU 1-batch (1 station, 1 ts) avant qsub. Process : Write `.engineer_brief.md` PUIS
run-engineer.sh (séparé). SSH conda : `module load Miniconda3/24.9.2-0; eval "$(conda shell.bash
hook)"; conda activate fuxicfd`. Shell stdout vide → rediriger /tmp + Read.

## 9. Report (≤300 mots)
- Setup : obs scp OK, inputs Perdigão (réutilisés M_G6 ou re-extraits), Δt=6h noté.
- PROPAGATION (cœur) : la correction est-elle lisse autour du pixel station ? métrique
  smoothness agrégée, ratio delta(ring_k)/delta(centre), % cas lisses vs pics.
- JUSTESSE centre : MAE corr vs raw vs obs au pixel station Perdigão (la correction tient-elle
  sur ce domaine hors-NOAA ?).
- VERDICT : (a) propagation lisse → correction physiquement cohérente → M_H'1a déployable /
  GO M_H'1b 4-saisons ; (b) pic 1-pixel → la correction overfit le pixel calibré → escalate
  Boss pour rescope (ex : loss multi-pixel, régularisation spatiale).
- Job ID. proposals (≤3) + memory candidates (≤3).

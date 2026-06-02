# DEPARTMENT BRIEF — M_H'1d″ : eval GPU corrigé (vitesse + direction relief-aware)

## 0. CONTEXTE (faits mesurés par le Boss — ne pas re-deviner)
- Cache `/scratch/maitreje/dsw/phase_H_prime_M_H1_jja_cache` = **60971 grid.zarr / 63022
  pairings = 97%**. 214/214 stations couvertes (val-43 : 12061 cachés). Les 2051 manques =
  dates bordure mai/sept hors JJA. Clé cache = `f"{station_id}_{ts:%Y%m%dT%H%M}"`.
- Parquet `noaa_jja2023_v2.parquet` (63022 rows) = sortie **RAW** surrogate v2 (M_G7), PAS
  corrigé ANN. Colonnes : `station_id,timestamp,source,lat,lon,elev,height_obs,u_obs,v_obs,
  speed_obs,u_pred,v_pred,w_pred,speed_pred,u10_era5_baseline,v10_era5_baseline,
  speed_era5_baseline,era5_time_delta_minutes,obs_zarr`. PAS de `wind_dir` natif → la
  direction obs se calcule via `atan2(-u_obs,-v_obs)` (convention from-dir).
- **Direction RAW mesurée (val-43 vs OBS)** : surrogate mae_dir 43.3° biais +16.4° ;
  ERA5 34.4° → raw dégrade la direction vs ERA5. 22/43 stations >30°.
- **NUANCE user (à respecter dans l'analyse)** : "différent d'ERA5 ≠ faux". La déflexion
  relief-aware du surrogate est attendue. SEULE la comparaison **vs OBS** tranche. Il faut
  donc mesurer la direction du modèle CORRIGÉ vs OBS, ET caractériser la déflexion
  (corrigé−ERA5, corrigé−raw) stratifiée par terrain pour distinguer correction utile vs bruit.

## 1. Objectif
Produire les métriques du modèle **CORRIGÉ M_H'1a** (best.pt, ANN devant surrogate frozen)
sur les pairings cachés, en VITESSE et en DIRECTION, vs OBS et vs références (raw, ERA5).
Réparer le garde-fou cache qui a fait échouer le run précédent.

## 2. Mission ID + type
- **ID**: M_H'1d″ · **Type**: Implement (fix cache-filter ~30 LOC) + Validate (1 run A100 forward-only ~20min)

## 3. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/eval_devine_loso.py`
  (à corriger : `_assert_cache_ready` ~L145 hard-fail → filtre ; `main` ~L450)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/src/dataset_v2_obs_centered.py`
  (réutiliser la MÊME fonction de construction de clé/chemin de cache ; vérifier comment
   éviter toute matérialisation — flag additif `require_cached`/`materialise=False` si besoin)
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/training/devine_style_full_M_H1.yaml`
- `/Users/guillaume/Documents/Recherche/downscalewind/configs/hpc/eval_devine_loso.pbs` (déjà A100, 2h)

## 4. Artefacts Aqua
- ANN corrigé : `/home/maitreje/dsw/data/models/surrogate_v2_devine_M_H1_jja/best.pt` (epoch 5)
- surrogate frozen : `/home/maitreje/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
- parquet : `/home/maitreje/dsw/data/inference/noaa_jja2023_v2.parquet`
- cache : `/scratch/maitreje/dsw/phase_H_prime_M_H1_jja_cache` (60971 dirs — NE PAS modifier)
- env `fuxicfd`. SSH conda : `module load Miniconda3/24.9.2-0; eval "$(conda shell.bash hook)"; conda activate fuxicfd`.

## 5. Fix + deliverables
**Fix** : remplacer `_assert_cache_ready` (hard-fail) par `filter_to_cached_pairings(df,cache_dir)` :
garde les lignes dont `<cache>/<key>/grid.zarr` existe (clé via la fonction du dataset, NE PAS
hardcoder) ; log total/kept/dropped/par-mois/couverture-par-split ; BLOCK seulement si une
station val tombe à 0 pairing OU kept<1000 (ici 97% → PROCEED). Garantir **zéro
matérialisation** (passer la liste cached-only ; flag additif si nécessaire). Vérifier
`ls <cache>|wc -l` == 60971 avant ET après.

**Deliverables** (forward corrected + raw déjà extraits ; ajouter le volet déflexion) :
1. `per_station_loso.csv` (~214 lignes, split taggé) : `mae_corrected, mae_raw, mae_era5,
   bias_corrected, bias_raw, mae_dir_corrected, mae_dir_raw, mae_dir_era5, bias_dir_corrected`.
   Direction obs = `atan2(-u_obs,-v_obs)` ; pred corrigé via u/v du champ corrigé au pixel
   central ; raw via `u_pred,v_pred` du parquet ; ERA5 via `u10/v10_era5_baseline`.
2. `speed_mae_by_sector.csv` : 8 secteurs ERA5 × {mae vitesse corr/raw, mae_dir corr/raw}.
3. **`direction_deflection.csv`** (le volet "relief-aware") : par station, déflexion
   `mean(corrected_dir − era5_dir)` et `mean(corrected_dir − raw_dir)`, croisée avec une proxy
   terrain (elev, et si dispo z0_eff/relief depuis le grid.zarr ou topo features). But : voir si
   la déviation corrigé↔ERA5 corrèle avec le relief (→ correction physique) ou est dispersée (→ bruit).
4. `loso_summary.json` : bloc `coverage` (kept/dropped/par-mois/par-split) + bloc `speed`
   (val/train mae_corrected/raw/era5, bias, par classe vent low/mid/high) + bloc `direction`
   (val/train mae_dir corrected/raw/era5, bias, # stations >30°, buckets 0-15/15-30/30-45/45-90/>90).

## 6. Allowed edit zones
- `services/module2b-surrogate/eval_devine_loso.py` (modify)
- `services/module2b-surrogate/src/dataset_v2_obs_centered.py` (modify : flag additif require_cached SEULEMENT si nécessaire)
- `data/validation/phase_H_prime_loso/` (sorties)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`
- Aqua : scp script(s) + écriture sous `data/validation/phase_H_prime_loso/`

## 7. Forbidden
- AUCUNE matérialisation (cache reste 60971 dirs). AUCUN re-train. AUCUN qsub >2h.
- PAS de modif model_vit_v2.py / dataset_v2_vit.py / surrogate best.pt / M_H'1a best.pt / split-training logic.
- PAS de commit/push. Modif dataset = additive only (M_H'1a reproductible).

## 8. Exit criterion
- 3 CSV + json produits ; `per_station_loso.csv` ~214 lignes avec colonnes vitesse+direction(corr/raw/era5).
- val-43 `mae_corrected` calculé (interpréter vs raw 1.60 et vs mandate 1.222) AVEC channels=4.
- val-43 `mae_dir_corrected` vs `mae_dir_raw` (43.3°) vs `mae_dir_era5` (34.4°) → l'ANN aide ou empire la direction ?
- Cache == 60971 avant ET après. Diff zones autorisées only.

## 9. Runner + validation
`codex` via run-engineer.sh, single-spawn. Validation : dry-run CPU 1-batch ; `wc -l` cache
avant ; qsub A100 ; après : `wc -l` cache == 60971 + exit criterion. Process : Write
`.engineer_brief.md` PUIS run-engineer.sh (étapes séparées, pas batché). Shell stdout rend
souvent vide → rediriger SSH vers /tmp et Read.

## 10. Report (≤300 mots, report_templates.md §Department)
- COUVERTURE : kept/dropped, skew mois, par split.
- VITESSE : val-43 mae_corrected vs raw vs era5, gap train↔val, par classe vent, top-5 dures.
- DIRECTION : val-43 mae_dir corrected vs raw(43.3°) vs era5(34.4°), biais, # stations >30°,
  buckets. **DÉFLEXION** : corrigé−ERA5 corrèle-t-il avec relief (utile) ou dispersé (bruit) ?
- VERDICT direction : l'ANN speed-only améliore / neutre / dégrade la direction →
  (a) si dégrade → justifie mission ANN_direction (DEVINE 2-réseaux) ;
  (b) si neutre/améliore relief-aware → speed-only OK → next M_H'1c Perdigão IOP.
- Cache inchangé (60971). Job ID. proposals (≤3) + memory candidates (≤3).

# DEPARTMENT BRIEF — M_H'1g-ingest : ssrd + tcc (radiation/nuages) via GEE, alignés grid.zarr

## 0. CONTEXTE
M_H'1f a montré que les features scalaires de stabilité (grad_T, RH, q) réduisent peu la
sur-correction Perdigão (2.27→2.15). On enrichit avec deux proxies du forçage diurne
(chauffage de surface) qui pilotent le vent thermique — régime où l'ANN sur-corrige. CAPE
absent de GEE (CDS-only) → on prend à la place ssrd + tcc, CONFIRMÉS présents dans GEE
`ECMWF/ERA5_HOURLY` : `surface_solar_radiation_downwards` + `total_cloud_cover`.
Cette mission tourne EN PARALLÈLE de M_H'1g-stl (STL-encodeur). Pas de dépendance.

## 1. Objectif
Produire un store ssrd+tcc horaire JJA 2023 Europe (même bbox/période que
`era5_europe_hourly_jja2023.zarr`), exploitable comme features auxiliaires station par le
loader `dataset_v2_obs_centered.py`. Voie = GEE download+transfert (user a accès local GEE,
projet `ee-guillaumemaitrejean`), PAS CDS.

## 2. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
- `/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_era5land_fwi_gee.py`
  (modèle de pattern GEE : auth ee.Initialize project ee-guillaumemaitrejean, ImageCollection
   ECMWF, export/download, transfert) — réutiliser le pattern, PAS le copier aveugle
- `/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_era5_europe_hourly.py`
  (bbox/période/grille de référence du store JJA existant à matcher)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/src/dataset_v2_obs_centered.py`
  (compute_topo_features L52 + load_grid_inputs L110 — comment les features aux sont assemblées ;
   le but est que ssrd/tcc deviennent lisibles à l'extraction station)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/extract_v2_input_at_coords.py`
  + `utils/inference_input.py` (comment l'input station est extrait des sources gridded)

## 3. Décision de design à prendre par le Department
Deux options pour rendre ssrd/tcc consommables station, CHOISIR la moins invasive :
- **(A) store annexe** `data/raw/era5_radcloud_jja2023.zarr` {ssrd, tcc} horaire grille Europe,
  joint à la lecture par (lat,lon,ts) — comme l'ERA5 surface. PRÉFÉRÉ (n'invalide pas le cache
  grid.zarr existant 60971, pas de re-matérialisation).
- (B) ré-injecter dans chaque grid.zarr → re-matérialisation 60971 = ~14h, à ÉVITER.
Choisir (A) sauf raison technique forte. Documenter le choix.

## 4. Deliverables
- `services/data-ingestion/ingest_era5_radcloud_gee.py` (new) : download GEE ssrd+tcc, période
  JJA 2023 (2023-06-01→08-31, + marges si besoin), bbox Europe = celui du store JJA existant,
  écrit `data/raw/era5_radcloud_jja2023.zarr` (schema cohérent avec les stores ERA5 du repo :
  coords/time int64 ns, lat, lon ; vars ssrd, tcc float32).
- Smoke : vérifier 1 point (lat,lon,ts) → valeurs ssrd (W/m² ou J/m² accumulé — DOCUMENTER
  l'unité GEE et la convention horaire/accumulée) + tcc (0-1) physiques.
- Helper de lecture station `radcloud_at(lat, lon, ts)` (dans le nouveau script ou
  inference_input.py) → 2 scalaires, prêt à être branché par M_H'1g-stl plus tard.
- Rapport : unité/convention ssrd, taille store, couverture temporelle/spatiale, comment
  M_H'1g-stl le consommera (chemin + helper).

## 5. Allowed edit zones
- `services/data-ingestion/ingest_era5_radcloud_gee.py` (new)
- `services/module2b-surrogate/utils/inference_input.py` (modify : helper radcloud_at, additif)
- `configs/hpc/` (si un PBS de download nécessaire — sinon download local)
- `data/raw/era5_radcloud_jja2023.zarr` (output)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`

## 6. Forbidden
- PAS de re-matérialisation du cache grid.zarr (60971 intact) — option (A) store annexe.
- PAS de modif des stores ERA5 existants, ni du surrogate/ANN/best.pt.
- PAS de commit/push. PAS de CDS (GEE only).
- Si l'auth GEE échoue / quota → reporter BLOCKED (ne pas basculer CDS sans Boss).

## 7. Exit criterion
- `data/raw/era5_radcloud_jja2023.zarr` existe, {ssrd,tcc} présents, couvre JJA 2023, bbox
  Europe ≈ store JJA, valeurs physiques (smoke 1 point OK, unité documentée).
- helper `radcloud_at` testé (renvoie 2 scalaires finis sur Perdigão + 1 station NOAA).
- Diff zones autorisées only. Cache 60971 inchangé.

## 8. Runner + ops
`custom`/`claude-subagent` selon dispo GEE — le download GEE tourne EN LOCAL (user a accès,
Codex sandbox probablement pas d'auth GEE → si Engineer Codex ne peut s'authentifier, livrer le
script + le Department lance le download localement). Auth : `ee.Initialize(project=
'ee-guillaumemaitrejean')`. Shell stdout vide possible → rediriger /tmp + Read.

## 9. Report (≤300 mots)
- Option retenue (A store annexe vs B), justification.
- ssrd unité/convention (accumulé J/m² vs flux W/m², horaire), tcc range.
- store path/taille/couverture, helper radcloud_at testé (valeurs Perdigão + NOAA).
- Comment M_H'1g-stl branchera ssrd/tcc (2 features aux → topo_dim 12→14).
- proposals (≤3) + memory candidates (≤3). Pas de commit.

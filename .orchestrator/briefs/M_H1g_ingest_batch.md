# DEPARTMENT BRIEF — M_H'1g-ingest-batch : batcher le download GEE ssrd/tcc (25h → minutes)

## 0. CONTEXTE — fix perf
Le script `ingest_era5_radcloud_gee.py` (déjà livré) télécharge HEURE PAR HEURE via
`getDownloadURL(NPY)` par timestamp = **2208 requêtes HTTP GEE → ~25h**. Le download lent a
été TUÉ par le Boss. Il faut le réécrire en mode GROUPÉ. C'est un fix perf ciblé, pas une
nouvelle ingestion.

## 1. Le levier (mesuré par le Boss)
Grille bbox Europe `36,-10,52,10` @0.25° = **~65 lat × 80 lon = ~5200 pixels**. JJA = ~2208 h.
`ee.ImageCollection.getRegion(geometry, scale)` rend une série temporelle COMPLÈTE (tous
timestamps × tous pixels du rectangle) en UN appel, plafonné ~1M éléments/appel.
→ max ~192 h/appel → **chunker par ~8 jours → ~12 appels au lieu de 2208**. De 25h à minutes.

## 2. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
- `/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_era5_radcloud_gee.py`
  (le script à modifier : remplacer `download_with_cache` boucle horaire L259-306 par un
   download chunké via getRegion ; GARDER write_radcloud_zarr / verify_store / coords logic /
   bbox / unités ssrd inchangés — le schema de sortie ne change PAS)
- `/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_era5land_fwi_gee.py`
  (réf pattern GEE auth/collection)

## 3. Le fix (précis)
- Nouvelle fonction `download_batched(collection, region, times, lats, lons, scale, chunk_days=8)` :
  - pour chaque chunk temporel [t0, t1) de ~chunk_days :
    `collection.filterDate(t0,t1).getRegion(region, scale).getInfo()` → liste de lignes
    `[id, longitude, latitude, time, ssrd, tcc]` (header en ligne 0).
  - parser en numpy : reconstruire (n_times_chunk, n_lat, n_lon) en indexant via les coords
    lon/lat retournées (mapper chaque ligne sur (i_lat, j_lon, k_time)). ATTENTION à la
    convention lat N→S (cohérente avec fetch_grid_and_coords existant) et à l'alignement temps.
  - le `time` getRegion est en **ms epoch** → convertir en datetime64[ns].
  - empiler les chunks → (n_times_total, n_lat, n_lon) ssrd + tcc.
- Garder un cache par CHUNK (npz par chunk, pas par heure) pour reprise si un chunk échoue.
- Si getRegion dépasse la limite (erreur "Too many values") → réduire chunk_days dynamiquement
  (fallback : diviser le chunk par 2 et retry).
- `download_with_cache` (horaire) peut rester comme fallback `--mode hourly`, mais le DÉFAUT
  devient `--mode batched`.
- Réutiliser les coords de `fetch_grid_and_coords` (déjà OK) pour l'alignement spatial.

## 4. Validation
- **Smoke batched** : `--smoke` (1 chunk, ex 24-48h) → store écrit, shapes correctes,
  valeurs ssrd/tcc finies physiques, lat N→S. Comparer 1 pixel/heure au cache .npz horaire
  EXISTANT (`data/raw/_cache_radcloud/`, 48 fichiers déjà téléchargés) pour vérifier que le
  batch donne les MÊMES valeurs que l'horaire (preuve de non-régression du parsing).
- Puis run complet JJA batché (local, conda downscalewind, GEE auth ee-guillaumemaitrejean).
  Doit finir en MINUTES, pas heures.

## 5. Allowed edit zones
- `services/data-ingestion/ingest_era5_radcloud_gee.py` (modify : download batché)
- `services/module2b-surrogate/utils/inference_input.py` (si helper radcloud_at pas encore livré)
- `data/raw/era5_radcloud_jja2023.zarr`, `data/raw/_cache_radcloud/` (output)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`

## 6. Forbidden
- PAS de changement du schema de sortie (write_radcloud_zarr, coords, unités ssrd J/m²) — juste
  la MÉTHODE de download.
- PAS de re-matérialisation grid.zarr (cache 60971 intact). PAS de modif surrogate/ANN/best.pt.
- PAS de commit/push. PAS de CDS. Si getRegion auth/quota échoue → BLOCKED.

## 7. Exit criterion
- `era5_radcloud_jja2023.zarr` écrit via download batché, couvre JJA 2023, {ssrd,tcc} finis,
  lat N→S, smoke batched == cache horaire existant (mêmes valeurs sur pixels communs).
- Run complet en minutes (reporter le wall réel vs 25h horaire).
- helper radcloud_at testé (Perdigão + NOAA, 2 scalaires finis).
- Diff zones autorisées only. Cache 60971 inchangé.

## 8. Runner + ops
`custom`/`claude-subagent` — le download GEE tourne EN LOCAL (Codex sandbox sans auth GEE
probable → Engineer livre le code, Department lance le download localement : conda env
downscalewind, `ee.Initialize(project='ee-guillaumemaitrejean')`). Lancer en background +
surveiller `ls data/raw/_cache_radcloud | wc -l` ou le log. Shell stdout vide → /tmp + Read.

## 9. Report (≤300 mots)
- Méthode batchée (getRegion, chunk_days, # appels réels), wall réel vs 25h.
- Smoke : batch == horaire sur pixels communs (non-régression parsing).
- store path/taille/couverture, helper radcloud_at testé (Perdigão + NOAA).
- proposals (≤3) + memory candidates (≤3). Pas de commit.

# Engineer brief — M_H+ Axe 1 : DEM tiles PT + sud ES

## Mission

Télécharger les tiles Copernicus DSM 30m couvrant PT (lat 37-42°N, lon -10 à -6°E) + sud ES (lat 36-39°N, lon -8 à -1°E) sur Aqua, et étendre l'auto-discovery du DEM dans `inference_input.py` pour les détecter.

État de départ confirmé :
- 86 tiles déjà présentes sur Aqua : `/home/maitreje/dsw/data/raw/srtm_tiles/` couvrant FR + nord ES (`Copernicus_DSM_COG_10_N<NN>_00_<E|W><EEE>_00_DEM.tif`).
- Tiles PT/sud ES manquantes : seules N36/N37 W002/W003 présentes ; il manque ~22-24 tiles.
- 40 stations NOAA PT + 41 stations NOAA ES_sud (<39N) sont actuellement bloquées faute de tiles.

## Lectures requises (avant tout code)

1. `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/mandate.md` §0, §3, §5 M_H+
2. `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/engineer.md` §`_resolve_dem_path` (cf. `services/module2b-surrogate/utils/inference_input.py:67-94`)
3. `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/boss.md` §pre-rm grep
4. `services/module2b-surrogate/utils/inference_input.py` (la fonction `_resolve_dem_path` à étendre)

## Tâches précises

### 1. Calculer la liste exacte des tiles manquantes

Grille cible (24 tiles environ) :
- **PT continental** : lat N37→N41, lon W007→W010 → ~5×4 = 20 tiles candidates
- **Sud ES** : lat N36→N38, lon W001→W007 et E000→E001 → tiles déjà partiellement présentes (cf. N36_W002, N36_W003, N37_W002, N37_W003 OK).
- Croiser avec la liste actuelle `ssh maitreje@aqua "ls /home/maitreje/dsw/data/raw/srtm_tiles/"` pour identifier seulement les **manquantes**.

Conseil : écrire un petit helper Python `services/data-ingestion/ingest_dem_copernicus.py` qui prend une bbox (S,W,N,E) en CLI, énumère les tiles 1°×1° couvrant cette bbox, et télécharge celles manquantes seulement. Pattern de nom : `Copernicus_DSM_COG_10_N<NN>_00_<E|W><EEE>_00_DEM.tif`.

### 2. Source de téléchargement

Endpoint à utiliser (préférer dans cet ordre) :
- **AWS S3 anonyme** : `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_<TILE>/Copernicus_DSM_COG_10_<TILE>_DEM/Copernicus_DSM_COG_10_<TILE>_DEM.tif` (vérifier le URL exact, le path peut varier).
- Alternative : bucket OpenTopography ou ESA Copernicus Land monitoring (auth requise).
- Vérifier d'abord avec `curl -I` qu'un sample URL répond 200 avant le bulk.

Implémentation : `urllib.request` ou `requests` + checksum optionnel (taille fichier ≥ 1 MB).

### 3. Cibler le téléchargement vers Aqua

Le DEM est volumineux (~50-80 MB/tile × 24 = ~1.5 GB). Ne PAS télécharger en local. Workflow :
- Soit lancer le script directement sur Aqua via ssh : `ssh maitreje@aqua "cd dsw && module load Miniconda3/24.9.2-0 && source $(conda info --base)/etc/profile.d/conda.sh && conda activate fuxicfd && python services/data-ingestion/ingest_dem_copernicus.py --bbox 36,-10,42,-6 --output-dir /home/maitreje/dsw/data/raw/srtm_tiles/ --skip-existing"`
- Soit créer un PBS court (1-2h, 1 CPU, no GPU) `configs/hpc/ingest_dem_pt_es.pbs` et `qsub` (réseau Aqua interne plus stable).

**Output Aqua** : `/home/maitreje/dsw/data/raw/srtm_tiles/` (même répertoire que les 86 tiles existantes — pas un nouveau répertoire, sinon `_resolve_dem_path` ne les trouvera pas).

### 4. Étendre `_resolve_dem_path` si nécessaire

La fonction actuelle (cf. `services/module2b-surrogate/utils/inference_input.py:67-94`) gère déjà l'auto-detect par nommage `Copernicus_DSM_COG_10_N<NN>_00_<E|W><EEE>_00_DEM.tif` quand `dem` est un répertoire. Vérifier que le PBS `configs/hpc/infer_at_stations.pbs` passe bien `--dem $BASE/data/raw/srtm_tiles/` (répertoire) et pas un tif unique.

Si la PBS pointe sur un fichier `.tif` au lieu du répertoire (cf. ligne 20 `DEM:=$BASE/data/raw/srtm_perdigao_30m.tif`), créer une nouvelle PBS `infer_at_stations_v2.pbs` qui pointe sur le répertoire `srtm_tiles/` (à faire en Axe 3 — laisser un TODO ici).

### 5. Smoke verification

Sur Aqua après download :
```bash
ssh maitreje@aqua "ls /home/maitreje/dsw/data/raw/srtm_tiles/ | wc -l"
# attendu ≥ ~110 tiles (86 + 24)
ssh maitreje@aqua "ls /home/maitreje/dsw/data/raw/srtm_tiles/ | grep -E 'N(3[7-9]|4[0-2])_00_W(00[6-9]|010)' | wc -l"
# attendu ≥ ~15 nouvelles PT
```

Smoke Python sur 2-3 stations PT/sud ES via `_resolve_dem_path` :
```python
from services.module2b-surrogate.utils.inference_input import _resolve_dem_path
from pathlib import Path
# test lat=38.5, lon=-8.0 (Portugal central)
p = _resolve_dem_path(Path('/home/maitreje/dsw/data/raw/srtm_tiles/'), 38.5, -8.0)
print(p)  # must resolve to N38_00_W008
```

## Allowed edit zones

- `services/data-ingestion/ingest_dem_copernicus.py` (nouveau)
- `services/module2b-surrogate/utils/inference_input.py` (uniquement si extension du auto-discovery est nécessaire, e.g. nested dir support — ne pas casser le format actuel)
- `configs/hpc/ingest_dem_pt_es.pbs` (nouveau, si PBS préféré sur ssh inline)
- `tmp/dem_tiles_pt_es_sud/` (workspace local pour scripts)

## Forbidden actions

- NE PAS télécharger les tiles en local (volumineux).
- NE PAS supprimer ou renommer des tiles existantes.
- NE PAS commit / push sans approbation Boss.
- NE PAS modifier `infer_at_stations.py` (réservé à Axe 3).
- NE PAS modifier `extract_v2_input_at_coords.py`.
- Avant tout `rm`, `grep -r 'filename' --include='*.py'`.

## Exit criterion

1. Au moins 20 nouvelles tiles Copernicus DSM PT/sud ES téléchargées dans `/home/maitreje/dsw/data/raw/srtm_tiles/` sur Aqua.
2. `_resolve_dem_path` retourne avec succès le tile pour 3 coordonnées test : (38.5, -8.0) PT, (37.0, -5.0) ES Andalousie, (39.5, -8.5) PT Lisbonne.
3. Script `services/data-ingestion/ingest_dem_copernicus.py` versionné (mais non commit), CLI documentée, `--skip-existing` fonctionnel.
4. Output stdout du rapport ≤ 200 mots couvrant : (i) URL endpoint utilisé, (ii) liste tiles téléchargées (counts par zone), (iii) test de résolution sur 3 coords, (iv) caveats détectés (URLs mortes, etc.).

## Validation commands

```bash
# Sur Aqua
ssh maitreje@aqua "ls /home/maitreje/dsw/data/raw/srtm_tiles/ | wc -l"
ssh maitreje@aqua "ls /home/maitreje/dsw/data/raw/srtm_tiles/ | grep -E 'W(00[6-9]|010)' | sort | head"

# Local — git diff propre
cd /Users/guillaume/Documents/Recherche/downscalewind
git diff --check
git status --short
```

## Rapport attendu

≤ 200 mots, structure :
- Endpoint Copernicus DSM utilisé (URL exacte)
- Nombre de tiles téléchargées + bbox couverte (avant→après count)
- Test resolve_dem_path sur 3 coords (lat, lon) → résultats
- Caveats (URL 404, slow download, etc.)
- Modifications LOC delta (fichiers touchés)

# services/data-ingestion

Service d'ingestion des données brutes pour DownscaleWind.

## Responsabilité

Ce service télécharge et prépare toutes les données d'entrée du pipeline :
- **ERA5** : réanalyse ECMWF 25 km / 6h (training + validation historique)
- **IFS ECMWF** : prévisions Open-Meteo (inférence opérationnelle)
- **SRTM GL1** : topographie 30m (stub — V2)
- **CGLS-LC100** : couverture du sol 100m pour la rugosité z₀ (stub — V2)

## Schéma Zarr de sortie

```
era5_perdigao.zarr/
ifs_perdigao.zarr/
  pressure/
    u, v, z, t, q  [time, level, lat, lon]
  surface/
    u10, v10, t2m  [time, lat, lon]
  coords/
    time, level, lat, lon
```

Métadonnées CF-conventions, compression Blosc LZ4.

## Utilisation

```bash
# Installer les dépendances
pip install -r requirements.txt
pip install -e ../../shared/

# Créer ~/.cdsapirc avec votre clé API Copernicus
# Voir : https://cds.climate.copernicus.eu/how-to-api

# ERA5 — téléchargement 2016-2017 (durée ~2-4h selon la queue CDS)
python ingest_era5.py \
    --site perdigao \
    --start 2016-01 --end 2017-12 \
    --output ../../data/raw/era5_perdigao.zarr

# IFS Open-Meteo — depuis 2024 (sans authentification)
python ingest_ifs_openmeteo.py \
    --site perdigao \
    --start 2024-02-03 --end 2024-12-31 \
    --model ecmwf_ifs025 \
    --output ../../data/raw/ifs_perdigao.zarr

# IFS HRES Open-Meteo — depuis 2017 (pour domain shift analysis)
python ingest_ifs_openmeteo.py \
    --site perdigao \
    --start 2017-01-01 --end 2017-12-31 \
    --model ecmwf_ifs_analysis_long_window \
    --output ../../data/raw/ifs_hres_perdigao.zarr

# Test (dry-run sans téléchargement)
python ingest_era5.py --site perdigao --start 2017-05 --end 2017-05 \
    --output /tmp/test.zarr --dry-run
```

## Via Docker

```bash
# Depuis la racine du projet
docker-compose --profile ingestion build data-ingestion
docker-compose --profile ingestion run data-ingestion \
    python ingest_era5.py --site perdigao --start 2017-05 --end 2017-05 \
    --output /data/raw/era5_perdigao.zarr
```

## Checkpointing

Les téléchargements sont checkpointés mois par mois (ERA5) ou par blocs de 30 jours
(IFS). Un sentinel `.done` est créé dans `data/raw/.checkpoints/` après chaque
mois réussi avec le SHA256 du fichier téléchargé.

Pour reprendre après interruption, relancer la même commande — les mois déjà
traités sont automatiquement sautés.

Pour forcer le re-téléchargement d'un mois :
```bash
rm data/raw/.checkpoints/era5_2017_05.done
```

## Limites connues

- **Humidité spécifique (q) manquante dans IFS Open-Meteo** : reconstructible
  depuis RH via `shared.data_io.relative_humidity_to_specific_humidity()`.
  Voir `LIMITATIONS.md`.
- **Domain shift ERA5 → IFS** : ERA5 est une réanalyse (assimile des obs),
  IFS est une prévision. Le biais systématique est non caractérisé en V1.
  Utiliser IFS HRES depuis 2017 pour le quantifier sur la période Perdigão.


# DownscaleWind
Pipeline de recherche reproductible pour le downscaling de champs de vent de la
résolution synoptique (~25 km, ERA5) vers la résolution kilométrique (~1 km) en
terrain complexe, avec quantification d'incertitude.

## Objectifs

1. **Publication scientifique** (GMD ou Boundary-Layer Meteorology) — code reproductible,
   métriques comparables à la littérature Perdigão
2. **Dossier startup deeptech / BPI i-Lab** — verrou technologique démontré sur cas réel
3. **Application outdoor** (parapente, vélo de montagne) — profil vertical de vent
   à la demande avec enveloppe d'incertitude, latence < 1 seconde

## Architecture du pipeline

```
ERA5 25km (6h)
    │
    ├─► Module 1 — Downscaling temporel  (NeuralODE + GATv2, 6h → 1h)
    │       └── Sortie : champs horaires à résolution ERA5
    │
    └─► Module 2A — Profil vertical       (Monin-Obukhov + ERA5 pression)
            └── Conditions aux limites OpenFOAM
                    │
                    └─► Module 2A — CFD batch  (buoyantSimpleFoam, 240 runs)
                                └── Base de données CFD
                                        │
                                        └─► Module 2B — Surrogate GNN  (GATv2, bipartite)
                                                └── Champs 1km en <500ms
```

## Site de référence

**Perdigão, Portugal** — double crête parallèle, IOP mai–juin 2017.
Voir `configs/sites/perdigao.yaml` pour les métadonnées complètes.

## Structure du dépôt

```
downscalewind/
├── configs/          # Paramètres par site et par module
├── shared/           # Package Python partagé (logging, I/O)
├── services/
│   ├── data-ingestion/      # ERA5 (CDS), IFS (Open-Meteo), SRTM, land cover
│   ├── module1-temporal/    # NeuralODE + GNN temporal downscaling
│   ├── module2a-cfd/        # OpenFOAM buoyantSimpleFoam batch runner
│   ├── module2b-surrogate/  # GNN surrogate spatial
│   ├── module3-stochastic/  # Turbulence sous-horaire (V2)
│   └── validation/          # Métriques et comparaison baseline
├── data/
│   ├── raw/                 # ERA5, IFS, SRTM bruts (Zarr)
│   ├── processed/           # Profils verticaux, champs interpolés
│   └── cfd-database/        # Sorties OpenFOAM (Zarr)
└── notebooks/
    └── validation_report.ipynb
```

## Démarrage rapide

```bash
# 1. Environnement conda
conda env create -f environment.yml
conda activate downscalewind

# 2. Ingestion ERA5 (nécessite ~/.cdsapirc avec clé API)
cd services/data-ingestion
python ingest_era5.py --site perdigao --start 2016-01 --end 2017-12 \
                      --output ../../data/raw/era5_perdigao.zarr

# 3. Ingestion IFS via Open-Meteo (sans authentification)
python ingest_ifs_openmeteo.py --site perdigao --start 2024-02-03 --end 2024-12-31 \
                               --model ecmwf_ifs025 \
                               --output ../../data/raw/ifs_perdigao.zarr

# 4. Test bout-en-bout CFD (voir services/module2a-cfd/README.md)
```

## Reproductibilité

- Toutes les dépendances sont pinées dans `environment.yml`
- Les données sont téléchargées via scripts versionnés (jamais manuellement)
- Chaque fichier téléchargé est hashé (SHA256) et loggué
- Tous les seeds aléatoires sont fixés et loggués

## Documentation des limites

Voir `LIMITATIONS.md` pour les hypothèses et conditions de validité du pipeline.
Voir `DECISIONS.md` pour les choix architecturaux et les alternatives écartées.

## Références

- Fernando et al. (2019). The Perdigão: Peering into microscale details of mountain winds.
  *BAMS*. doi:10.1175/BAMS-D-17-0227.1
- Mann et al. (2017). Complex terrain experiments in the New European Wind Atlas.
  *PTRS-A*. doi:10.1098/rsta.2016.0101

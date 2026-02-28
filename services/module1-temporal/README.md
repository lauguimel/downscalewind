# services/module1-temporal

**Statut : TODO — V1**

## Responsabilité

Downscaling temporel : interpolation 6h → 1h (PoC) et 1h → 15min (V2) des champs
ERA5/IFS à résolution spatiale fixe.

## Architecture prévue

```
Contexte spatial : L1 (5×5 = 25 nœuds) ou L2 (7×7 = 49 nœuds) à résolution ERA5
                  ×  n_niveaux_pression × n_variables

GNN encodeur (GATv2Conv)
  nœuds   : (u, v, T, Z, q, lat, lon_sin, lon_cos, z_géop)
  arêtes  : (Δlat, Δlon, Δz, distance)
  → état latent z(t=0h), z(t=6h)

NeuralODE (torchdiffeq, Dormand-Prince adaptatif)
  dz/dt = f_θ(z(t), c(t))
  c(t)  = (cos/sin angle solaire, encodage temporel)
  → interpolation z(t+1h), ..., z(t+5h)

Décodeur MLP → (u, v, T, Z) par pas horaire
```

## Paires d'entraînement

- **Entrée** : ERA5 sous-échantillonné à 6h
- **Vérité terrain** : ERA5 horaire (disponible sur CDS, dataset `reanalysis-era5-pressure-levels` avec `time: ["00:00", ..., "23:00"]`)
- Auto-supervisé sur ERA5 → pas de données supplémentaires pour le PoC

## Propriété clé : agnosticisme en résolution

Les nœuds du graphe portent leurs coordonnées spatiales comme features.
Le modèle entraîné sur ERA5 0.25° peut être utilisé sur IFS 0.1° ou AROME 0.0125°
sans ré-entraînement.

## Fichiers à créer

- `train.py` : boucle d'entraînement + MLflow
- `model.py` : architecture GATv2 + NeuralODE
- `dataset.py` : chargement des paires ERA5 depuis le store Zarr
- `infer.py` : inférence sur un snapshot ERA5/IFS
- `requirements.txt`
- `Dockerfile`

## Questions ouvertes

- Taille de contexte optimal : L1 (25 nœuds) vs L2 (49 nœuds) ?
  → ablation study prévu
- Nombre de couches GATv2 pour capturer la structure synoptique ?
- Contexte temporel élargi (t-6h, t, t+6h, t+12h) vs contexte minimal (t, t+6h) ?

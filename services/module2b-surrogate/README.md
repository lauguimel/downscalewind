# services/module2b-surrogate

**Statut : TODO — V1 (après constitution de la base CFD)**

## Responsabilité

Surrogate GNN pour le downscaling spatial ERA5 25 km → 1 km en mode stationnaire.
Remplace les simulations CFD en inférence (<500 ms vs ~30 min).

## Architecture : GATv2, graphe bipartite ERA5→CFD

```
Contexte ERA5 : L1 (3×3=9 nœuds) ou L2 (5×5=25 nœuds) × niveaux pression

Nœuds CFD/sortie : ~2500 nœuds (50×50 km à 1 km)
  features : (élévation, pente, z₀, courbure, distance sol)

Graphe bipartite :
  arêtes ERA5→CFD : k-NN k=5, pondération gaussienne
  arêtes CFD↔CFD  : 8-voisins spatiaux

Encode-Process-Decode :
  Encodeur MLP sur features nœuds + arêtes
  N couches GATv2Conv (attention multi-tête)
    + conditionnement (dir_sin, dir_cos, |u|, Ri_b) par nœud
  Décodeur MLP → (u, v, w)

Loss = MSE_pondérée + λ_div · L_divergence
```

## Propriétés clés

- **sin/cos encoding** pour la direction du vent (pas de discontinuité 0°/360°)
- **Ri_b clamped** à [-2, +2] comme feature de conditionnement de stabilité
- **Ensemble** de 3–5 modèles (initialisations différentes) → enveloppe d'incertitude
- **Pruning bipartite** : k=5 voisins ERA5 par nœud CFD (ablation study k=3/5/8 prévu)

## Validation

1. LIDAR Doppler Perdigão (profils verticaux, IOP 2017)
2. Mâts Perdigão ~50 tours (u, v, |u| par hauteur)
3. Critère succès PoC : RMSE ≥ 30% de réduction vs ERA5 interpolé

## Latence inférence

Objectif : < 500 ms sur CPU pour un profil vertical sur 1 point géographique
- Base CFD lookup : O(1) → < 10 ms
- Inférence GNN TorchScript : ~50–200 ms (à benchmarker)
- Export : TorchScript ou ONNX

## Fichiers à créer

- `model.py` : architecture GATv2 + graphe bipartite
- `dataset.py` : chargement base CFD Zarr
- `train.py` : entraînement ensemble + MLflow
- `infer.py` : inférence + export TorchScript
- `build_graph.py` : construction du graphe depuis cellules OpenFOAM
- `requirements.txt`, `Dockerfile`

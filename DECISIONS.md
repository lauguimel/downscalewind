# Décisions architecturales — DownscaleWind

Ce fichier trace les choix techniques structurants, leur justification, et les
alternatives écartées. Il sert de mémoire du projet pour la publication et les
revues de code.

**Convention :** chaque décision a un statut — `STABLE`, `A_REVOIR`, `OUVERTE`.

---

## D1 — Format de données inter-modules : Zarr

**Statut :** STABLE

**Décision :** Zarr comme format d'échange entre tous les modules.

**Justification :**
- Cloud-native : accès possible depuis S3/GCS sans téléchargement complet
- Chunking flexible : optimisé pour les patterns d'accès (tranches temporelles,
  colonnes verticales, cartes horizontales) selon les besoins du module
- Accès parallèle en lecture sans verrou (contrairement à HDF5)
- Intégration native avec xarray (`.to_zarr()`, `xr.open_zarr()`)
- Compression Blosc LZ4 : bon compromis vitesse/ratio pour données météo

**Alternatives écartées :**
- HDF5 : verrous en lecture parallèle, moins cloud-friendly
- NetCDF4 (base HDF5) : mêmes limitations + overhead de compatibilité
- Parquet : orienté colonne, mal adapté aux tableaux N-D multi-variables

**Schéma de chunks :** `{"time": 120, "level": -1, "lat": -1, "lon": -1}`
(≈ 30 jours à 6h en temps, niveaux/lat/lon complets par chunk)

---

## D2 — Deux régimes d'ingestion séparés dès le début

**Statut :** STABLE

**Décision :** Script d'ingestion distinct pour ERA5/CDS (training) et IFS/Open-Meteo
(inférence opérationnelle). Schéma Zarr identique en sortie.

**Justification :**
- Le domain shift ERA5→IFS est une question scientifique ouverte importante
- Séparer les sources dès l'ingestion rend le shift visible et mesurable
- Open-Meteo IFS est gratuit, sans quota, accessible sans authentification
  → idéal pour le pipeline opérationnel (latence, disponibilité)
- CDS ERA5 est la source de référence pour la validation historique (Perdigão 2017)

**Domain shift :** non caractérisé en V1. IFS HRES Open-Meteo (depuis 2017-01-01)
permet de le quantifier sur la période Perdigão sans accès MARS ECMWF.

---

## D3 — Reconstruction u/v depuis speed+direction (Open-Meteo)

**Statut :** STABLE

**Décision :** Reconstruire les composantes u/v à partir de la vitesse scalaire et
la direction météorologique fournie par l'API Open-Meteo.

**Convention météorologique** (vent VENANT DE la direction indiquée) :
```python
u = -speed * np.sin(np.deg2rad(direction))  # composante Est (positive vers l'Est)
v = -speed * np.cos(np.deg2rad(direction))  # composante Nord (positive vers le Nord)
```

**Erreur de reconstruction :** < 0.01% (arithmétique exacte, pas d'approximation).

**Alternative écartée :** Accès au bucket S3 open-data ECMWF (format .om propriétaire,
nécessite bibliothèque spécifique, moins stable).

---

## D4 — Logging structuré JSON

**Statut :** STABLE

**Décision :** Tous les services utilisent un logger JSON partagé (`shared/logging_config.py`).

**Justification :**
- Machine-parsable : compatible avec Grafana Loki, ELK, Datadog
- Hash SHA256 des fichiers téléchargés loggué systématiquement → reproductibilité
- Champs structurés (module, étape, fichier, durée) → requêtes analytiques simples

**Alternative écartée :** logging texte standard — difficile à parser pour le monitoring.

---

## D5 — Téléchargement mensuel ERA5 avec checkpointing

**Statut :** STABLE

**Décision :** Télécharger ERA5 mois par mois, avec sentinel SHA256 par mois.

**Justification :**
- L'API CDS refuse les requêtes > 1000 pas de temps (limite technique)
- Un mois à 6h = 4 × 30 = 120 items → bien en dessous de la limite
- Le checkpointing permet de reprendre un téléchargement interrompu sans doublon

**Retry :** 3 essais avec backoff exponentiel (60s, 120s, 240s) via `tenacity`.

---

## D6 — Maillage CFD : snappyHexMesh (pas Gmsh)

**Statut :** STABLE

**Décision :** Pipeline de maillage `blockMesh` → `snappyHexMesh` → STL depuis SRTM.

**Justification :**
- Standard de fait pour la CFD atmosphérique en terrain complexe
- Tous les exemples ABL OpenFOAM utilisent snappyHexMesh
- Intégration native avec `atmBoundaryLayerInletVelocity`
- Entièrement scriptable : SRTM → `rasterio` → `numpy-stl` → STL → `snappyHexMeshDict`

**Critères de qualité maillage (checkMesh) :**
- Max non-orthogonality < 70° (idéal < 60°)
- Max skewness < 4 (idéal < 3)
- Rapport de taille entre cellules adjacentes < 20

**Alternative écartée :** Gmsh — plus flexible mais conversion polyMesh non native,
moins d'exemples ABL documentés. Pertinent si migration vers SU2 ou code_saturne.

---

## D7 — Solveur CFD : buoyantSimpleFoam (pas simpleFoam)

**Statut :** STABLE

**Décision :** `buoyantSimpleFoam` comme solveur principal pour le batch CFD.

| Solveur | Énergie | Flottabilité | Usage |
|---------|---------|--------------|-------|
| `simpleFoam` | Non | Non | Neutre isotherme seulement |
| `rhoSimpleFoam` | Oui | Partielle | Haute vitesse (Ma > 0.1) |
| **`buoyantSimpleFoam`** | **Oui** | **Primaire** | **ABL stratifiée ← choix** |

**Justification :** Les effets thermiques (courants de pente, brise de montagne,
stratification nocturne) sont des phénomènes clés à Perdigão. `buoyantSimpleFoam`
résout nativement l'équation d'énergie + terme de flottabilité.

**Fallback :** Si le budget CFD dépasse 40h/240 runs, utiliser `simpleFoam`
pour les runs neutres (Ri_b ≈ 0, ~60%) et `buoyantSimpleFoam` pour les
runs stables/instables (~40%). À décider après le benchmark du premier run.

---

## D8 — Conditions aux limites CFD : 3 faces inlet, domaine fixe

**Statut :** STABLE

**Décision :** Domaine orienté fixe (N-S/E-W). Direction du vent paramétrée via
`flowDir` dans `atmBoundaryLayerInletVelocity`. 3 faces latérales → inlet ;
back + top → pressure outlet ; bottom → wall.

**Justification :** Évite de générer un maillage par direction (×16 = 16 maillages).
Le vecteur `flowDir` couvre ±90° efficacement — les 3 faces inlet garantissent
qu'au moins 2 faces reçoivent le flux entrant pour toute direction.

**Nudging volumique :** non implémenté en V1. Mesure de dérive à faire sur le
premier batch : si |u_CFD(z > 3 km) − u_ERA5| / |u_ERA5| > 10%, envisager
`meanVelocityForce` dans la zone z > 3 km.

---

## D9 — Versioning des modèles : MLflow local

**Statut :** STABLE

**Décision :** MLflow local (`mlflow.pytorch.log_model()`) stocké dans `data/mlruns/`.

**Justification :**
- UI comparative sans infrastructure externe (`mlflow ui --port 5000`)
- Logging automatique des hyperparamètres, métriques par epoch, artefacts
- Migrable vers MLflow Tracking Server distant si déploiement startup

**Alternative écartée :** DVC (plus complexe, Git-coupled, pas d'UI native) ;
hash manuel (reproductible mais sans comparaison visuelle).

**Blocker :** À configurer avant le premier run d'entraînement GNN.

---

## D10 — Sérialisation GNN : TorchScript pour inférence < 500ms

**Statut :** STABLE

**Décision :** Export TorchScript (`torch.jit.script`) du surrogate GNN pour
l'inférence opérationnelle.

**Stratégie latence :**
- Base CFD pré-calculée, indexée par (direction, vitesse, Ri_b) → lookup O(1) < 10ms
- Inférence GNN TorchScript CPU : à benchmarker (objectif ~50–200ms)
- Total cible : < 500ms pour un profil vertical sur 1 point géographique

**Fallback :** ONNX export si TorchScript incompatible avec le déploiement edge.

---

## D11 — Pruning des arêtes bipartite ERA5→CFD

**Statut :** A_REVOIR après ablation study

**Décision provisoire :** k-NN avec k=5 voisins ERA5 par nœud CFD (distance 2D),
pondération `exp(−d²/σ²)` avec σ = résolution ERA5 / 2 ≈ 12.5 km.

**Justification :** Asymétrie forte (9–25 nœuds ERA5 vs ~2500 nœuds CFD). Sans
pruning, les nœuds ERA5 centraux dominent le passage de message et écrasent
le signal topographique. k=5 est un compromis standard dans la littérature GNN
pour les graphes multi-résolution.

**Ablation study prévue :** k=3 vs k=5 vs k=8 — mesurer l'impact sur RMSE
et le speed-up factor error.

---

## D12 — Module 1 : fenêtre de contexte spatiale L1 (5×5) ou L2 (7×7)

**Statut :** OUVERTE

**Question :** La fenêtre de 5×5 (25 nœuds, 1 anneau) est-elle suffisante pour
capturer les structures synoptiques nécessaires à l'interpolation temporelle 6h→1h ?
Ou faut-il 7×7 (49 nœuds, 2 anneaux) ?

**À tester :** Ablation study sur la taille de contexte. Commencer avec L1 (5×5)
pour le PoC, évaluer sur les métriques de validation temporelle.

---

## D13 — Module 2 : contexte ERA5 L1 (3×3) ou L2 (5×5)

**Statut :** OUVERTE

**Question :** Pour le downscaling spatial, 9 mailles ERA5 (3×3) suffisent-elles
à capturer les gradients de grande échelle, ou faut-il 25 mailles (5×5) pour
inclure les effets baroclines à grande échelle ?

**À tester :** Comparer RMSE et speed-up factor error avec L1 vs L2.

---

*Dernière mise à jour : initialisation du projet*

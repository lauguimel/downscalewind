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
- Intégration native avec les BCs ABL OpenFOAM (`inletOutlet`, wall functions)
- Entièrement scriptable : SRTM → `rasterio` → `numpy-stl` → STL → `snappyHexMeshDict`

**Critères de qualité maillage (checkMesh) :**
- Max non-orthogonality < 70° (idéal < 60°)
- Max skewness < 4 (idéal < 3)
- Rapport de taille entre cellules adjacentes < 20

**Alternative écartée :** Gmsh — plus flexible mais conversion polyMesh non native,
moins d'exemples ABL documentés. Pertinent si migration vers SU2 ou code_saturne.

---

## D7 — Solveur CFD : simpleFoam + k-ε (révisé)

**Statut :** STABLE (révisé 2026-03)

**Décision :** `simpleFoam` + k-ε modifié (Parente et al.) comme solveur principal.

| Solveur | Énergie | Flottabilité | Usage |
|---------|---------|--------------|-------|
| **`simpleFoam`** | **Non** | **Non** | **RANS neutre ← choix principal** |
| `buoyantBoussinesqSimpleFoam` | Oui | Boussinesq | Stratifié (à activer si besoin) |
| `buoyantSimpleFoam` | Oui | Primaire | Non recommandé (compressible inutile) |

**Justification (révision) :** Les études de référence à Perdigão (Letzgus et al.
WES 2023, Neunaber et al. WES 2022) utilisent toutes `simpleFoam` + k-ε pour le
RANS stationnaire. `buoyantBoussinesqSimpleFoam` sera activé pour les runs stratifiés
si nécessaire (approximation Boussinesq plus stable que compressible).

**Fallback stratifié :** `buoyantBoussinesqSimpleFoam` pour les runs avec Ri_b ≠ 0.

---

## D8 — Conditions aux limites CFD : Robin BC (inletOutlet), domaine fixe

**Statut :** STABLE (révisé)

**Décision :** Domaine orienté fixe (N-S/E-W). Toutes les 4 faces latérales
utilisent `inletOutlet` (Robin BC) : Dirichlet (profil prescrit) quand le flux
entre, Neumann (`zeroGradient`) quand il sort, basculé par face de cellule selon
le signe du flux local. Top : `slip`. Bottom : `noSlip` + wall functions.

**Justification :** Cohérent avec Venkatraman et al. (WES 2023), Neunaber et al.
(WES 2022), et Palma et al. (WES 2020) à Perdigão. Élimine la logique sin/cos
d'assignation inlet/outlet et fonctionne naturellement pour toute direction de
vent sans configuration manuelle.

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

## D14 — Modèle de turbulence : k-ε modifié (pas k-ω SST)

**Statut :** STABLE

**Décision :** Migrer de k-ω SST vers k-ε modifié (Parente et al. 2011) pour le
RANS stationnaire à Perdigão.

**Justification :**
- Seul modèle RANS validé quantitativement à Perdigão (Letzgus et al. WES 2023) :
  88M cellules, 12.5 m résolution, comparaison k-ε / k-ε modifié / k-ω
- k-ω SST perd son avantage avec wall functions (y+ >> 1), ce qui est le cas à
  notre résolution cible (≥ 40 m)
- Cohérence ABL native avec les BC `atmBoundaryLayer` d'OpenFOAM ESI v2006+
- Termes sources canopée (`atmPlantCanopyTurbSource`) conçus pour k-ε

**Données terrrain :** ESA WorldCover 2021 (10 m) + ETH Canopy Height 2020 (10 m)
remplacent CGLS-LC100 (100 m). z₀ = 0.1 × h_canopy (GWA4/DTU standard).

**Termes sources ajoutés :**
- `atmCoriolisUSource` (latitude 39.716°N)
- `atmPlantCanopyUSource` + `atmPlantCanopyTurbSource` (Cd=0.2, LAD depuis carte)

**Consensus littérature :** Le choix du modèle de turbulence est secondaire par
rapport à la résolution du maillage (≤ 40 m, Palma 2020) et les termes sources
canopée/Coriolis (Letzgus 2023).

**Alternative écartée :** k-ω SST — standard industriel mais aucun avantage démontré
à Perdigão avec wall functions. Convergence terrain raide légèrement meilleure mais
ne compense pas l'absence de validation.

**Références :**
- Letzgus et al. (WES 2023) — Source terms and inflow at Perdigão
- Neunaber et al. (WES 2022) — Wind turbine at Perdigão (E-Wind k-ε précurseur)
- Palma et al. (WES 2020) — Mesh resolution threshold 40 m
- Parente et al. (2011) — Modified k-ε for consistent ABL profiles

---

## D15 — Occupation du sol : WorldCover + ETH Canopy Height (pas CGLS-LC100)

**Statut :** STABLE

**Décision :** ESA WorldCover 2021 (10 m) + ETH Global Canopy Height 2020 (10 m)
comme données de référence pour z₀, displacement height d, et LAD.

**Justification :**
- Résolution 10× supérieure à CGLS-LC100 (10 m vs 100 m)
- WorldCover : 11 classes, Sentinel-1+2, validation globale >75% overall accuracy
- ETH Canopy Height : Sentinel-2 + GEDI LiDAR, résolution spatiale 10 m
- Conversion z₀ = 0.1 × h via table GWA4/DTU standard (même que Global Wind Atlas)
- LAD = LAI / h_canopy pour les termes sources canopée OpenFOAM

**Alternative écartée :** CGLS-LC100 — résolution trop grossière (100 m) pour un
maillage CFD cible de 40 m. Pas de hauteur de canopée.

---

*Dernière mise à jour : 2026-03 (migration k-ε + land cover)*

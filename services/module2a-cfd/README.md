# services/module2a-cfd

**Statut : TODO — V1**

## Responsabilité

1. **Reconstruction du profil vertical** ERA5/IFS → conditions aux limites OpenFOAM
2. **Batch runner CFD** : génération et exécution de 240 simulations `simpleFoam`
3. **Export** des champs CFD (u, v, w, p, k, ε, T) vers la base de données Zarr

## Solveur : simpleFoam + k-ε modifié

Résout les équations RANS incompressibles stationnaires — validé quantitativement
à Perdigão (Letzgus et al. WES 2023, 88M cellules, 12.5 m résolution).

Turbulence : k-ε modifié (Parente et al. 2011) avec `epsilonWallFunction`.
Source terms : Coriolis + canopée forestière via `fvOptions`.

## Conditions aux limites (Robin BC, domaine fixe)

| Face | U | k | ε | p_rgh | T |
|------|---|---|---|-------|---|
| West, East, South, North | `inletOutlet` | `inletOutlet` | `inletOutlet` | `fixedValue 0` | `inletOutlet` |
| Top | `slip` | `zeroGradient` | `zeroGradient` | `fixedValue 0` | `inletOutlet` |
| Terrain | `noSlip` | `kqRWallFunction` | `epsilonWallFunction` | `fixedFluxPressure` | (see T.j2) |

`inletOutlet` = Robin BC : Dirichlet (profil prescrit) quand le flux entre,
Neumann (`zeroGradient`) quand il sort. Basculé automatiquement par face de cellule.
Réf : Venkatraman et al. (WES 2023), Neunaber et al. (WES 2022).
Direction du vent paramétrée via `flowDir` → pas de rotation du maillage.

## Maillage : snappyHexMesh

Pipeline : SRTM 30m → rasterio (resample 1 km) → numpy-stl (STL) → snappyHexMesh

Critères qualité (checkMesh) :
- Max non-orthogonality < 70° (idéal < 60°)
- Max skewness < 4

## Plan de batch

```
16 directions × 5 vitesses × 3 stabilités = 240 runs de base
Temps estimé : 20–40 min/run × 240 = 80–160h sur 8 cœurs
→ BENCHMARKER sur 1 run avant de lancer le batch complet
```

## Métriques de cohérence physique (sans observations)

1. Résidu de divergence : max(|∇·u|·Δx/|u|) < 1e-3
2. Conformité loi log : R² > 0.95 sur u(z) en terrain plat amont
3. Facteur de speed-up vs théorie Jackson-Hunt
4. Intensité de turbulence TI ∈ [0.05, 0.25] près du sol
5. Résidus RANS : continuité < 1e-4, momentum < 1e-3

## Structure templates OpenFOAM

```
templates/openfoam/
├── system/
│   ├── blockMeshDict.j2
│   ├── snappyHexMeshDict.j2
│   ├── controlDict.j2
│   ├── fvSchemes.j2
│   └── fvSolution.j2
├── constant/
│   ├── turbulenceProperties
│   └── transportProperties
└── 0/
    ├── U.j2, T.j2, k.j2, epsilon.j2, p_rgh.j2, alphat.j2
```

## Fichiers à créer

- `prepare_inflow.py` : reconstruction profil vertical (3 couches + Helmholtz 1D)
- `generate_mesh.py` : SRTM → STL → blockMesh + snappyHexMesh
- `_archive/run_cfd_batch.py` : batch runner multiprocessing (archived — use run_sf_poc.py)
- `export_cfd.py` : fluidfoam → Zarr
- `check_coherence.py` : métriques de cohérence physique
- `requirements.txt`, `Dockerfile`

## Questions ouvertes

- Budget réel : benchmarker sur un run Perdigão avec snapshot 2017-05-01 00h
- Nudging volumique : mesurer dérive z > 3 km après le premier batch
- Nombre minimal de runs pour une base d'entraînement GNN suffisante ?

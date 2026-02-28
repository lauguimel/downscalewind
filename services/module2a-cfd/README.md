# services/module2a-cfd

**Statut : TODO — V1**

## Responsabilité

1. **Reconstruction du profil vertical** ERA5/IFS → conditions aux limites OpenFOAM
2. **Batch runner CFD** : génération et exécution de 240 simulations `buoyantSimpleFoam`
3. **Export** des champs CFD (u, v, w, p, k, ω, T) vers la base de données Zarr

## Solveur : buoyantSimpleFoam

Résout les équations de Navier-Stokes compressibles avec équation d'énergie
et terme de flottabilité — adapté aux écoulements ABL stratifiés.

Turbulence : k-ω SST avec `atmOmegaWallFunction` (standard ABL OpenFOAM).

## Conditions aux limites (domaine fixe)

| Face | Condition | Variables |
|------|-----------|-----------|
| Left, Front, Right | `atmBoundaryLayerInletVelocity` | U, k, ω |
| Back | `totalPressure` p=0 | p_rgh |
| Top | `pressureInletOutletVelocity` + p=0 | p_rgh, U |
| Bottom | `noSlip` + `atmNutWallFunction` | U=0, nut |

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
    ├── U.j2, T.j2, k.j2, omega.j2, p_rgh.j2, alphat.j2
```

## Fichiers à créer

- `prepare_inflow.py` : reconstruction profil vertical (3 couches + Helmholtz 1D)
- `generate_mesh.py` : SRTM → STL → blockMesh + snappyHexMesh
- `run_cfd_batch.py` : batch runner multiprocessing
- `export_cfd.py` : fluidfoam → Zarr
- `check_coherence.py` : métriques de cohérence physique
- `requirements.txt`, `Dockerfile`

## Questions ouvertes

- Budget réel : benchmarker sur un run Perdigão avec snapshot 2017-05-01 00h
- Nudging volumique : mesurer dérive z > 3 km après le premier batch
- Nombre minimal de runs pour une base d'entraînement GNN suffisante ?

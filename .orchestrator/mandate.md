# MANDATE — Ablation OFAT multi-collines : identifier le levier dominant de la conservation du momentum

Boss source of truth. Departments read only the relevant mission slice.

## 0. Contexte (résumé pour reprise rapide)

Le projet est stallé sur un constat : audit v2 teacher (500 cas, snapshot pré-recovery)
indique mediane `CFD_central / ERA5_u10 = 0.696` pour vent fort, et le diagnostic
sur `ct_d_fire_0170_case_ts014` montrait `crop/inflow @ 10m = 0.499` (config originale).

Le **recovery plan** a déjà identifié 4 leviers candidats et les a testés en **silos**
sur ct_d_fire_0170 et ct_d_fire_0056 :

| Stack | crop/inflow @10m | center/inflow @10m | Site |
|---|---:|---:|---|
| control (baseline) | 0.499 | 0.467 | ct_d_fire_0170 |
| + slip_top seul | 0.539 | 0.516 | ct_d_fire_0170 |
| + slip_top + pg_geo flip | 0.630 | 0.575 | ct_d_fire_0170 |
| z0_wall=0.005 seul (atmNutk) | 0.627 | — | ct_d_fire_0170 |
| **best-stack** (slip+p=0+pg_flip+z0=0.005) flat | **1.031** | 1.023 | flat analytique |
| best-stack ridge_cos2 | 1.081 | 1.590 | ridge analytique (crest p90 = 1.624) |
| best-stack + wc_capped_0.05 | 0.911 | 0.844 | ct_d_fire_0056 (vrai terrain) |

**Problème** : aucune ablation **toutes choses égales** sur le **même cas** avec
**plusieurs orientations de relief**. On ne sait pas quel facteur du best-stack
porte la majeure partie du gain. User hypothèse : `p_top = 0` + `pg_geo` sont
les dominants.

## 1. High-level objective

Construire un **cas test analytique multi-collines** (3-4 collines disposées en
croix/triangle, orientations variées) et y réaliser une **ablation OFAT** complète
du best-stack pour identifier **le levier dominant** restaurant la conservation
du momentum au sol au centre du domaine.

> Avec un cas test multi-collines fixé, partir du best-stack et retirer chaque
> facteur un par un (top U BC, top p BC, pg_geo, z0_wall) : quel facteur fait
> chuter `crop @ 10m` de >0.10 ? Quel facteur change la distribution de flux
> (acceleration crest, deceleration lee) le plus ?

**Done** = matrice ablation propre + distribution PDF flux + identification du
levier dominant + décision argumentée pour la regen 9k (best-stack à adopter
vs. simplifier vs. ajouter un facteur encore manquant).

## 2. Out of scope

- Régénération des 9000 cases (décision portée par cette mission, exécution
  hors scope).
- Modification du surrogate v2 (`train_v2.py`, `dataset_v2.py`).
- Ablation sur vrai terrain v2 (pure analytique pour cette mission).
- Tester `atmNutU` vs `atmNutk` (déjà fait, marginal +0.01).
- Calibrer le profil 2m (anomalie connue, hors scope).
- Lateral BC Robin cylindrique (déferré).

## 3. Constraints

- **Langue** : communication FR, code/commits EN, conventional commits.
- **HPC** : Aqua, PBS Pro, jamais mpirun sur login node, container Apptainer
  `openfoam_v2512.sif`, bind-mount case_dir → `/home/ofuser/run`, COPIER (jamais
  symlinker) polyMesh + section_*.
- **Confirmation** : confirmer avec utilisateur avant qsub, scp, ssh destructifs ;
  pas de commit auto.
- **Métrique imposée** : analyse en **distribution** (PDF + moyenne + médiane +
  p10/p90/max sur masques crest / lee / flat / global), pas un single scalar.
- **Cas test** : analytique 3-4 collines, terrainBlockMesher, mesh comparable
  à la campagne v2 (180×180×40 inner).
- **Builder réutilisable** : étendre `build_terrain_canary.py` avec `--terrain-kind
  multi_hill` (pas de script ad-hoc).

## 4. Architecture decisions (ADRs)

| Date       | Decision                                                          | Rationale                                              |
|------------|-------------------------------------------------------------------|--------------------------------------------------------|
| 2026-05-13 | Best-stack BC = slip_top + p=0 + pg_geo flip + z0_wall=0.005      | Phase B canary : flat=1.031, ridge crest=1.624         |
| 2026-05-16 | wc_capped_0.05 retenu pour la rugosité hétérogène                 | z0_treatment canary ct_d_fire_0056                     |
| 2026-05-18 | Ablation OFAT obligatoire avant toute regen 9k                    | Aucune ablation propre toutes choses égales fait sur le même site multi-orientations |
| 2026-05-18 | Cas test = analytique 3-4 collines (croix/triangle)               | Choix user : isoler le levier physique en terrain contrôlé |
| 2026-05-18 | Métrique = distribution PDF + crest/lee/flat masks                | Choix user : éviter qu'un seul scalaire masque les effets locaux |

## 5. Missions

### M6 — Audit critique + design cas test + matrice ablation finalisée

- **Status**: planned
- **Goal**: Department lit recovery plan + memories + builders existants, produit :
  (a) audit de l'état actuel (ce qui est cohérent, ce qui ne l'est pas) ;
  (b) spec géométrique précise du cas multi-hill (positions, hauteurs, largeurs,
  rotation, taille domaine) ; (c) matrice ablation OFAT finalisée (liste des
  variantes à lancer, par ordre de priorité) ; (d) liste des métriques de
  distribution à calculer + format CSV de sortie.
- **Allowed edit zones**: aucun (read-only audit + design)
- **Exit criterion**: rapport ≤300 mots Department avec spec, matrice ablation
  (~8-10 variantes), métriques. Pas de code.

### M7 — Étendre build_terrain_canary.py + smoke local

- **Status**: planned
- **Goal**: Codex (via Department) ajoute `--terrain-kind multi_hill` au builder
  existant + nouvelle métrique distribution dans `audit_terrain_canary.py` (ou
  équivalent). Smoke test local sur 1 variante (sans Apptainer si possible, sinon
  via Docker local) pour valider la géométrie et l'audit.
- **Allowed edit zones**:
  - `services/module2a-cfd/analysis/build_terrain_canary.py`
  - `services/module2a-cfd/analysis/audit_terrain_canary.py` (ou audit script
    relevant, à confirmer en M6)
  - `services/module2a-cfd/templates/openfoam/` (uniquement si nécessaire)
  - `test/scratch/`, `scratch/`, `tmp/` (smoke tests)
- **Exit criterion**: 1 variante générée localement, mesh + 0/ + system/ + STL
  multi-hill OK, dry-run audit script sur un grid.zarr factice.

### M8 — PBS array ablation + qsub Aqua + export

- **Status**: planned
- **Goal**: prep PBS array (1 case par variante d'ablation), upload base case sur
  Aqua, qsub, monitor, post-process (`writeCellCentres` + export grid.zarr), scp
  audit CSVs locaux.
- **Allowed edit zones**:
  - `configs/hpc/ablation_multi_hill.pbs` (nouveau)
  - `data/validation/ablation_multi_hill/` (CSVs locaux)
- **Exit criterion**: N variantes convergées (≥80% de la matrice), tous CSVs
  distribution présents localement.

### M9 — Analyse distribution + identification levier dominant + décision

- **Status**: planned
- **Goal**: Department produit :
  (a) tableau ablation (variante × hauteur × stat) ;
  (b) figure distribution PDF par variante + masques crest/lee/flat ;
  (c) classement des facteurs par poids relatif (Δ crop @10m, Δ crest, Δ lee) ;
  (d) décision argumentée : best-stack à adopter / simplifier / compléter pour
  la regen 9k.
- **Allowed edit zones**:
  - `data/validation/ablation_multi_hill/<study>_analysis.{csv,md,png}`
  - `docs/openfoam_wind_conservation_recovery_plan_2026-05-13.md` (nouvelle
    section "Phase C — ablation multi-hill")
- **Exit criterion**: tableau ablation + 1 figure PDF + paragraphe décision.

### M10 — Documentation + commit (Boss-only)

- **Status**: planned
- **Goal**: Boss filtre les memory candidates, commit conventionnel + push avec
  confirmation user.
- **Allowed edit zones**: Boss-only.

## 6. Mission graph

```
M6 ──► M7 ──► M8 ──► M9 ──► M10
        │              │
        └─ smoke local ─ ablation matrix sur Aqua
```

## 7. Decisions taken after M6 (2026-05-18)

| Question | Decision | Rationale |
|---|---|---|
| Geometry | 3 hills triangle asym. (H=200/250/300 m, L=600/800/1000 m, cos²) | Couvre windward/lee/cross en 1 run |
| Domain | 6×6×2.5 km, mesh v2 (180×180×40) | Pas de mesh confounder vs production |
| Inflow | Reuse ERA5 `ct_d_fire_0056_ts014` | Garde pg_geo calibré sur ERA5 réel |
| Directions | 270°W (V0-V8) + 0°N cross-check sur V0+V1 = 11 runs | Valider OFAT independent de direction |
| V4 (pg native sign) | Upfront | Cheap, sign check explicite |
| pg_geo calibration | ERA5 850/800/700 (méthode actuelle) | Calibration free-stream 1500m deferred à M9 conditionnel |

### Matrice ablation finale (11 variantes)

| id | top_U | top_p | pg_geo | z0_wall | extra | dir | rôle |
|---|---|---|---|---|---|---|---|
| V0 | inletOutlet | zeroGrad | OFF | 0.05 | wc native | 270°W | control |
| V1 | slip | fixed 0 | flip | 0.005 | wc_cap_0.05 | 270°W | best-stack |
| V2 | inletOutlet | zeroGrad | flip | 0.005 | wc_cap_0.05 | 270°W | -slip_top |
| V3 | slip | fixed 0 | OFF | 0.005 | wc_cap_0.05 | 270°W | -pg_geo |
| V4 | slip | fixed 0 | native | 0.005 | wc_cap_0.05 | 270°W | pg sign check |
| V5 | slip | fixed 0 | flip | 0.05 | wc_cap_0.05 | 270°W | -z0_wall low |
| V6 | slip | fixed 0 | flip | 0.005 | uniform 0.05 | 270°W | -wc heterogeneity |
| V7 | slip | zeroGrad | flip | 0.005 | wc_cap_0.05 | 270°W | top_p isolé |
| V8 | inletOutlet | zeroGrad | flip | 0.005 | wc_cap_0.05 | 270°W | -top entier |
| V0n | inletOutlet | zeroGrad | OFF | 0.05 | wc native | 0°N | control rotated |
| V1n | slip | fixed 0 | flip | 0.005 | wc_cap_0.05 | 0°N | best-stack rotated |

### Métriques de distribution (par variante × hauteur ∈ {2,10,20,50,100} m AGL)

- Bulk crop 4×4 km : `mean, median, p10, p50, p90, max` sur `speed` et `speed/U_inflow(h)`
- Masques per-hill : `crest_k = {terrain ≥ z_base+0.85·H_k} ∩ crop` ;
  `lee_k = {s_proj ∈ [0.25, 2.0]·L_k} ∩ {terrain ≤ z_base+0.3·H_k} ∩ crop` ;
  agrégat = max-over-hills (crest), min-over-hills (lee)
- `flat = crop ∩ {terrain ≤ z_base + 0.1·max(H_k)}` (vallées + plaines hors collines)
- Stats : `crest_max, crest_p90, lee_min, lee_p10, flat_mean, flat_p10`
- PDF : histogramme 40 bins de `speed/U_inflow` ∈ [0, 2.5] sur crop
- Output : `multi_hill_distribution.csv` (long format) + 1 PNG/variante (terrain+vent | ratio map @10m | PDF)

**Bug audit signalé** : 2 m AGL exclu des décisions OFAT (anomalie connue
`inflow_speed_at` normalization).

## 8. Pointers

- Recovery plan: `docs/openfoam_wind_conservation_recovery_plan_2026-05-13.md`
- Builder actuel: `services/module2a-cfd/analysis/build_terrain_canary.py`
  (supporte `flat`, `ridge_cos2`, `z0_treatment` mode)
- Audit actuel: `services/module2a-cfd/analysis/audit_v2_teacher_wind.py`,
  `audit_wall_z0.py`, `terrain_canary_metrics`
- z0 generator: `services/module2a-cfd/generate_z0_field.py`
- PBS template récent : `configs/hpc/z0_treatment_canary_ct_d_fire_0056_ts014.pbs`
- Sites CSV: `data/campaign/complex_terrain_v1/sites.csv`
- Memory : `.orchestrator/memory/{boss,department,engineer}.md`
- Auto-memory MEMORY.md: `~/.claude/projects/-Users-guillaume-Documents-Recherche-downscalewind/memory/`

## 9. Status historique (anciennes missions clôturées)

M1-M5 (z0_treatment canary sur site hétérogène) **done** 2026-05-16 → décision
`wc_capped_0.05` retenue. Archivé pour traçabilité, ne pas relancer.

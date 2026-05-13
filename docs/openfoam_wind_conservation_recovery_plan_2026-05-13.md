# DownscaleWind v2 - plan de recuperation conservation vent / relief

Date: 2026-05-13

## Diagnostic court

L'audit v2 indique que le surrogate n'est pas la cause principale du biais bas:
le teacher OpenFOAM freine deja le vent proche du sol.

Sur 500 cas v2 audites, a 10 m AGL:

- mediane `CFD central 2x2km / ERA5 u10 = 0.879`;
- cas venteux `ERA5 u10 >= 3 m/s`: mediane `CFD / ERA5 = 0.767`;
- cas `ERA5 u10 >= 5 m/s`: mediane `CFD / ERA5 = 0.696`;
- bord amont interne deja bas: mediane `upstream edge / inflow = 0.853`.

Sur le cas fort `ct_d_fire_0170_case_ts014`, les boundaryData laterales sont
correctes pres du sol, mais le centre tombe vers 2.5 m/s alors que ERA5 u10
vaut environ 5.0 m/s. Le probleme est donc une degradation interne du champ,
pas un inlet trop faible.

## Ce que le modele physique doit verifier

On ne doit pas demander a la moyenne du patch relief d'etre toujours superieure
a l'inflow: un relief produit aussi des zones de recirculation et de sillage.
En revanche, on doit imposer deux gates:

1. **Conservation bulk sur terrain plat**
   - a 10 m AGL, `mean_crop / inflow` dans `[0.95, 1.05]`;
   - `upstream_edge / inflow` dans `[0.95, 1.05]`;
   - pas de derive monotone amont -> centre -> aval sans forçage physique.

2. **Acceleration locale sur relief expose**
   - sur masques crete/versant au vent, `p90 / inflow > 1.10` pour les cas
     avec pente ou relief suffisant;
   - fraction du patch avec `U > inflow` non nulle et interpretable;
   - contraste crete/vallee et windward/lee coherent avec la direction du vent.

Le script `services/module2a-cfd/analysis/audit_v2_teacher_wind.py` calcule
maintenant aussi ces diagnostics relief:

```bash
python3 services/module2a-cfd/analysis/audit_v2_teacher_wind.py \
  --data-dir /path/to/v2/grid_zarrs \
  --output data/validation/v2_teacher_wind_relief_audit.csv \
  --summary-output data/validation/v2_teacher_wind_relief_audit_summary.csv \
  --heights 2,10,20,50,100 \
  --crop-km 2
```

Colonnes importantes: `ratio_crop_to_inflow`,
`ratio_upstream_edge_to_inflow`, `ratio_crest_p90_to_inflow`,
`ratio_windward_mean_to_inflow`, `fraction_crop_above_inflow`,
`terrain_relief_crop_m`, `terrain_slope_crop_p90_deg`.

## Solutions candidates, dans l'ordre

### 1. Forçage geostrophique du domaine

Le template actuel a Coriolis et les sources de turbulence ambiante, mais pas de
terme qui entretient la quantite de mouvement contre la friction de surface.
Avec plusieurs kilometres de fetch rugueux, le profil ABL peut donc s'effondrer
avant le patch central.

Le repo contient deja un prototype archive dans
`services/module2a-cfd/_archive/run_precursor.py`: pression geostrophique +
Coriolis. Pour le pipeline actuel, il faut ajouter seulement le pressure-gradient
si `atmCoriolisUSource` reste active:

```text
f = 2 Omega sin(latitude)
dp_dx = -f * V_g
dp_dy =  f * U_g
source_U = (dp_dx, dp_dy, 0)
```

Le template `services/module2a-cfd/templates/openfoam/constant/fvOptions.j2`
supporte maintenant ce bloc via `physics.pressure_gradient.enabled`, `dp_dx` et
`dp_dy`. Par defaut il est eteint, donc les anciens runs ne changent pas.

`U_g, V_g` doivent etre calibres pour que le cas plat conserve ERA5/inflow a
10 m. Ce n'est pas forcement `u10`: c'est le vent geostrophique qui produit le
profil pres du sol attendu apres friction.

Decision attendue:

- si le cas plat passe avec pressure-gradient, on peut regenerer un sous-ensemble
  v2 et comparer teacher vs stations;
- si le cas plat ne passe pas, il reste un probleme de wall functions, z0, mesh
  ou numerique.

### 2. `meanVelocityForce` comme clamp de diagnostic

OpenFOAM fournit `meanVelocityForce` et `patchMeanVelocityForce`. C'est utile
pour tester rapidement si un forçage de quantite de mouvement restaure le vent.
Ce n'est pas la premiere option publication, car le terme impose une moyenne de
vitesse et peut masquer une erreur physique.

Le template `fvOptions.j2` supporte aussi `physics.mean_velocity_force.enabled`
pour lancer ce canary sans modifier les fichiers OpenFOAM a la main.

Usage recommande:

- canary seulement;
- relaxation faible, par exemple 0.1 a 0.3;
- comparer contre le pressure-gradient;
- ne pas valider un teacher final uniquement avec ce clamp.

### 3. Coherence ABL inlet / wall functions

Les templates actuels utilisent:

- laterales `uniformMixed` + `mappedFile` pour U, k, epsilon;
- terrain `noSlip`;
- `kqRWallFunction`;
- `atmEpsilonWallFunction`;
- `atmNutkWallFunction`.

Les BC natives OpenFOAM disponibles incluent `atmBoundaryLayerInletVelocity`,
`atmBoundaryLayerInletK`, `atmBoundaryLayerInletEpsilon`, `atmNutUWallFunction`
et `atmNutkWallFunction`.

Test a faire:

- cas plat avec BC natives log-law pour isoler le probleme;
- cas plat avec BC production `mappedFile`;
- comparer `atmNutkWallFunction` vs `atmNutUWallFunction`.

Si les BC natives conservent le profil mais pas `mappedFile`, le bug est dans
notre initialisation/profil lateral. Si les deux echouent, le manque de
forçage/z0/fetch domine.

### 4. Rugosite z0 et land cover

Un `z0_eff = 0.05 m` uniforme peut etre trop rugueux pour stations OMM ouvertes,
mais il ne suffit probablement pas a expliquer seul un ratio 0.70 a 0.80 en cas
venteux. Il faut tout de meme auditer:

- `z0 = 0.005`, `0.01`, `0.03`, `0.05`, `0.10 m` sur terrain plat;
- WorldCover z0 mappe, avec lissage/capping;
- z0 station si la validation est a une station OMM/METAR exposee.

Gate: changer z0 doit modifier la couche de surface, mais ne doit pas etre la
seule maniere de compenser un manque de pressure-gradient.

### 5. Fallback surrogate: apprendre le speed-up, pas le vent absolu

Pour sauver rapidement le PoC FWI/API sans attendre une regeneration complete
OpenFOAM, on peut reutiliser les 9k simulations comme source de motifs de
relief, mais enlever leur biais bulk.

Principe:

1. projeter le vent CFD dans la base du vent inflow: composante parallele et
   composante transverse;
2. normaliser par le profil inflow ou par la moyenne centrale a chaque hauteur;
3. entrainer le surrogate a predire `speedup_parallel` et `speedup_cross`;
4. a l'inference, reconstruire avec ERA5/ERA5-Land/station-corrected wind.

Forme simple:

```text
e = U_inflow(z) / |U_inflow(z)|
e_perp = (-e_y, e_x)
alpha = dot(U_CFD, e) / mean_crop(dot(U_CFD, e))
beta  = dot(U_CFD, e_perp) / mean_crop(dot(U_CFD, e))

U_downscaled = U_ref(z) * (alpha * e + beta * e_perp)
```

Avec cette convention, le bulk du patch peut etre conserve par construction,
tout en gardant les accelerations relatives du relief. C'est moins pur qu'un
teacher CFD corrige, mais beaucoup plus robuste pour FWI et validation station.

## Matrice de tests minimale

### Phase A - canaries physiques

Cas:

- flat terrain, `z0 = 0.005, 0.01, 0.05`;
- ridge/hill analytique avec meme resolution verticale;
- 3 cas reels deja problematiques dont `ct_d_fire_0170_case_ts014`.

Options:

- template actuel;
- pressure-gradient geostrophique;
- pressure-gradient + z0 sensible;
- meanVelocityForce diagnostic.

Sorties:

- audit CSV aux hauteurs `2,10,20,50,100 m`;
- profils amont/centre/aval;
- cartes 10 m: speed, ratio to inflow, terrain, mask crete/vallee.

Builder reproductible:

```bash
python3 services/module2a-cfd/analysis/build_wind_conservation_canary.py \
  --base-case /path/to/prepared/openfoam/case \
  --output-dir /path/to/wind_canary \
  --overwrite

bash /path/to/wind_canary/run_canary_local_of.sh
bash /path/to/wind_canary/export_and_audit_canary.sh
```

Le builder cree quatre variantes a mesh/inlet identiques:

- `control`: cas original;
- `pg_geo`: pressure-gradient depuis le fit plan de geopotentiel ERA5;
- `pg_geo_flip`: meme amplitude, signe inverse, pour valider le signe OpenFOAM;
- `mean_force`: `meanVelocityForce` diagnostic.

Par defaut, le pressure-gradient utilise les niveaux ERA5 850/800/700 hPa si
disponibles dans `inflow.json`. Si `era5_grid` manque, le script bascule sur
une approximation par vent libre du profil a 1500 m.

Resultat canary reel `ct_d_fire_0170_case_ts014`, lance le 2026-05-13 sur Aqua:

- artifact: `/scratch/maitreje/dsw/wind_canary/ct_d_fire_0170_ts014`;
- job PBS: `21304211.aqua`;
- les quatre variantes convergent et exportent `grid.zarr`;
- le pressure-gradient corrige partiellement le damping, mais ne restaure pas
  encore le bulk central.

Moyenne sur `2,10,20,50,100 m AGL`, crop central 2 km:

| variante | crop / inflow | center / inflow | crop / ERA5 u10 |
|---|---:|---:|---:|
| `control` | 0.545 | 0.514 | 0.644 |
| `pg_geo` | 0.616 | 0.544 | 0.721 |
| `pg_geo_flip` | 0.647 | 0.600 | 0.759 |
| `mean_force` | 0.685 | 0.655 | 0.809 |

Lecture:

- `pg_geo_flip` est meilleur que `pg_geo`, donc le signe OpenFOAM/source doit
  etre traite comme suspect jusqu'au canary plat;
- meme le diagnostic `mean_force` reste sous l'inflow dans le crop central;
- a 2 m, le bord amont est deja proche de l'inflow (`control` 0.96,
  `mean_force` 1.12), mais le domaine perd fortement vers l'aval;
- conclusion: ne pas regenerer les 9k avec ce simple patch. Il faut maintenant
  calibrer sur terrain plat/ridge analytique ou passer au target speed-up
  conservatif pour le surrogate FWI.

Audit wall-z0 ajoute avec
`services/module2a-cfd/analysis/audit_wall_z0.py` sur les memes variantes:

- pas de mismatch `z0_eff` vs wall-z0 sur ce cas: `inflow_z0_eff = 0.05 m`,
  `nut/epsilon terrain z0 = uniform 0.05 m`;
- premiere cellule terrain mediane: `y = 5.63 m`, soit `y/z0 = 112`, donc on
  n'est pas dans un regime ou la rugosite depasse la premiere cellule;
- en revanche, la vitesse cellule murale reste bien sous le log-law attendu
  avec le `u_star` inflow:

| variante | wall U / log-law median | wall `u*_k` / inflow `u*` median |
|---|---:|---:|
| `control` | 0.557 | 0.624 |
| `pg_geo` | 0.610 | 0.656 |
| `pg_geo_flip` | 0.652 | 0.694 |
| `mean_force` | 0.700 | 0.756 |

Lecture wall-z0:

- le probleme n'est probablement pas une valeur `z0=0.05` mal lue par
  OpenFOAM;
- le mismatch plausible est plutot dynamique: `atmNutkWallFunction` depend du
  `k` local, et le `k` proche sol tombe a environ 60-75 % du `u*` inflow;
- prochaine canary utile: sweep `z0_wall = 0.005, 0.01, 0.03, 0.05` et
  comparaison `atmNutkWallFunction` vs `atmNutUWallFunction`, a inflow
  identique, pour separer rugosite trop forte et fermeture turbulente.

### Phase B - decision

Adopter le nouveau teacher si:

- flat 10 m `mean_crop/inflow >= 0.95`;
- cas relief: `crest_p90/inflow > 1.10` sur au moins les cas exposes;
- pas de sur-acceleration non physique massive: `crop_p90/inflow < 2.0` sauf cas
  extreme justifie;
- convergence numerique stable.

Sinon, ne pas regenerer les 9k. Passer au target speed-up pour le surrogate et
utiliser les CFD actuels uniquement comme motif relatif de relief.

### Phase C - production

1. Corriger template OpenFOAM ou definir le target speed-up.
2. Regenerer 100 a 300 cas QA seulement.
3. Validation station sur cas selectionnes.
4. Si gain net vs ERA5/ERA5-Land, lancer regeneration plus large ou fine-tune
   surrogate speed-up.

## Implication publication

Pour un papier Nature Communications, la narration doit eviter:

> "OpenFOAM predit le vent absolu vrai."

La narration robuste est:

> "Le pipeline apprend une reponse de terrain haute resolution, ancree sur les
> conditions meteorologiques synoptiques et corrigee par validation station."

Si le teacher CFD est corrige, on peut revendiquer un surrogate CFD multi-variable.
Si le teacher reste biaise en bulk mais fiable en motifs relatifs, on revendique
un downscaling hybride: ERA5/observations pour le bulk, CFD/surrogate pour la
structure sub-kilometrique du relief.

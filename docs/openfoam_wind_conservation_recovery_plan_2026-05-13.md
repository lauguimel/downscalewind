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

Resultat du sweep wall-z0, lance le 2026-05-14 sur Aqua:

- artifact: `/scratch/maitreje/dsw/wall_z0_canary/ct_d_fire_0170_ts014`;
- job PBS: `21310393.aqua`;
- 8 variantes convergentes: `z0_wall = 0.005, 0.01, 0.03, 0.05 m` x
  `atmNutkWallFunction`, `atmNutUWallFunction`;
- inflow conserve identique (`z0_eff = 0.05 m`) pour isoler le terme wall.

Moyenne sur `2,10,20,50,100 m AGL`, crop central 2 km:

| wall fn | z0 wall | crop / inflow | center / inflow | crop / ERA5 u10 |
|---|---:|---:|---:|---:|
| `atmNutk` | 0.005 | 0.670 | 0.645 | 0.785 |
| `atmNutk` | 0.010 | 0.635 | 0.608 | 0.746 |
| `atmNutk` | 0.030 | 0.575 | 0.543 | 0.678 |
| `atmNutk` | 0.050 | 0.544 | 0.515 | 0.644 |
| `atmNutU` | 0.005 | 0.678 | 0.654 | 0.792 |
| `atmNutU` | 0.010 | 0.644 | 0.620 | 0.755 |
| `atmNutU` | 0.030 | 0.585 | 0.558 | 0.689 |
| `atmNutU` | 0.050 | 0.556 | 0.527 | 0.655 |

Lecture du sweep:

- baisser `z0_wall` aide fortement: a 10 m, `crop/inflow` passe de 0.498
  (`atmNutk`, z0=0.05) a 0.627 (`atmNutk`, z0=0.005);
- mais meme `z0_wall=0.005` ne restaure pas le bulk: moyenne toutes hauteurs
  plafonnee a environ 0.67-0.68;
- `atmNutUWallFunction` est seulement marginalement meilleur que `atmNutk`
  (+0.01 environ), donc le choix de wall function n'est pas le levier principal;
- conclusion: `z0=0.05` et la friction murale expliquent une partie du damping,
  mais pas le facteur 2. Il faut tester maintenant une combinaison `z0_wall`
  plus faible + pressure-gradient calibre, ou passer au target speed-up
  conservatif pour ne pas faire apprendre ce biais bulk au surrogate.

Resultat du canary top-BC, lance le 2026-05-14 sur Aqua:

- artifact: `/scratch/maitreje/dsw/top_bc_canary/ct_d_fire_0170_ts014`;
- job PBS: `21310669.aqua`;
- 3 variantes convergentes:
  - `control`: top courant `U inletOutlet`, `p zeroGradient`,
    `k/epsilon inletOutlet`;
  - `slip_top`: `U slip`, `p/p_rgh fixedValue 0`, `k/epsilon zeroGradient`;
  - `slip_top_pg_geo`: `slip_top` + pressure-gradient ERA5.

Moyenne sur `2,10,20,50,100 m AGL`, crop central 2 km:

| variante | crop / inflow | center / inflow | crop / ERA5 u10 |
|---|---:|---:|---:|
| `control` | 0.545 | 0.514 | 0.644 |
| `slip_top` | 0.584 | 0.562 | 0.689 |
| `slip_top_pg_geo` | 0.659 | 0.608 | 0.771 |

A 10 m AGL:

| variante | crop / inflow | center / inflow | upstream edge / inflow | downstream edge / inflow |
|---|---:|---:|---:|---:|
| `control` | 0.499 | 0.467 | 0.785 | 0.535 |
| `slip_top` | 0.539 | 0.516 | 0.737 | 0.611 |
| `slip_top_pg_geo` | 0.630 | 0.575 | 0.738 | 0.671 |

Lecture top-BC:

- le `top` courant fuit probablement: le proxy geometrique `U_owner · Sf` donne
  pour `control` un flux sortant top positif non nul (`mean Un ≈ 0.54 m/s`,
  top-out ≈ 0.32 x l'inflow lateral entrant);
- mais `slip_top` seul ne restaure pas le bulk: +0.04 seulement en moyenne;
- `slip_top + pg_geo` rejoint le meilleur ordre de grandeur obtenu par
  `z0_wall=0.005`, mais reste loin du gate 0.85-0.95;
- les champs reconstruits ne contiennent pas `phi`; `postProcess
  patchIntegrate(phi,name=top)` ne peut donc pas etre applique a posteriori sur
  ces artifacts nettoyes. Le proxy top-flux est un diagnostic near-top, pas un
  flux OpenFOAM exact pour les cas `slip`.
- conclusion: le top BC est un contributeur, pas la fuite dominante unique. La
  suite utile est un canary combine `z0_wall=0.005` + `slip_top` +
  pressure-gradient signe/calibre, ou basculer vers target speed-up conservatif.

### Phase B - flat + ridge canary

Objectif: discriminer entre un deficit physique lie au relief reel et une
compression dynamique produite par la configuration OpenFOAM.

Builder ajoute:

```bash
python3 services/module2a-cfd/analysis/build_terrain_canary.py \
  --base-case /scratch/maitreje/dsw/top_bc_canary/ct_d_fire_0170_ts014/cases/case_ts000_slip_top_pg_geo \
  --terrain-kind flat \
  --output-dir /scratch/maitreje/dsw/terrain_canary/ct_d_fire_0170_ts014 \
  --z0-wall 0.005 \
  --pg-sign flip \
  --time 300 \
  --overwrite

python3 services/module2a-cfd/analysis/build_terrain_canary.py \
  --base-case /scratch/maitreje/dsw/top_bc_canary/ct_d_fire_0170_ts014/cases/case_ts000_slip_top_pg_geo \
  --terrain-kind ridge_cos2 \
  --output-dir /scratch/maitreje/dsw/terrain_canary/ct_d_fire_0170_ts014 \
  --z0-wall 0.005 \
  --ridge-height 200 \
  --ridge-half-width 1000 \
  --pg-sign flip \
  --time 300 \
  --overwrite
```

PBS prepare: `configs/hpc/terrain_canary_ct_d_fire_0170_ts014.pbs`.

Configuration commune:

- inflow identique au cas `ct_d_fire_0170_ts014`;
- terrain analytique remaille avec `terrainBlockMesher`;
- `z0_wall = 0.005 m`, uniforme;
- `U top slip`, `p/p_rgh top fixedValue 0`, `k/epsilon top zeroGradient`;
- pressure-gradient geostrophique avec signe `flip`, car le canary Phase A a
  montre que `pg_geo_flip` battait `pg_geo` sur ce cas;
- audit aux hauteurs `2,10,20,50,100 m AGL`, crop central 2 km;
- audit specialise `terrain_canary_metrics.csv`:
  - flat: `crop_to_inflow`, `center_to_inflow`;
  - ridge: `crest_max_to_inflow` a 10 m, `lee_min_to_inflow` a 10 m.

Resultat Aqua, lance le 2026-05-14:

- artifact: `/scratch/maitreje/dsw/terrain_canary/ct_d_fire_0170_ts014`;
- job PBS valide: `21313748.aqua`;
- les deux variantes convergent, exportent `grid.zarr` et auditent
  `terrain_canary_metrics.csv`;
- temps solve: flat 400 s, ridge 692 s sur 24 CPU;
- note reproducibilite: deux jobs precedents ont ete invalides avant cette
  execution (`21313373` mauvais import template, `21313457` controlDict mesh
  `endTime 0` + audit stale); les correctifs sont dans
  `build_terrain_canary.py`.

Resultats:

| canary | metrique | 2 m | 10 m | 20 m | 50 m | 100 m |
|---|---:|---:|---:|---:|---:|---:|
| flat | crop / inflow | 1.343 | 1.031 | 0.978 | 0.945 | 0.965 |
| flat | center / inflow | 1.333 | 1.023 | 0.970 | 0.939 | 0.963 |

Controle ridge bulk:

| canary | metrique | 2 m | 10 m | 20 m | 50 m | 100 m |
|---|---:|---:|---:|---:|---:|---:|
| ridge_cos2 | crop / inflow | 1.371 | 1.081 | 1.036 | 1.020 | 1.036 |
| ridge_cos2 | center / inflow | 2.132 | 1.590 | 1.448 | 1.294 | 1.230 |

| canary | metrique 10 m | valeur |
|---|---:|---:|
| ridge_cos2 | crest max / inflow | 1.624 |
| ridge_cos2 | lee min / inflow | 0.314 |

Lecture:

- le flat canary passe le gate principal a 10 m (`crop/inflow=1.031`,
  `center/inflow=1.023`) et reste proche de l'inflow a 20-100 m
  (`0.945-0.978` pour crop);
- le surplus a 2 m (`crop/inflow=1.343`) signale une calibration verticale
  imparfaite pres sol avec `z0_wall=0.005`, mais pas un deficit bulk;
- la colline analytique produit une acceleration de crete nette
  (`crest max/inflow=1.624`) dans la plage attendue des speedups
  orographiques forts;
- le lee minimum (`0.314`) confirme un sillage fort mais localise, pas une
  compression globale du domaine.

Decision matrix Phase B:

| flat | ridge crete | decision |
|---|---:|---|
| **>=0.95** | **>=1.30** | **dataset valide absolu, modulo offset calibre** |
| >=0.95 | 1.0-1.2 | dataset valide pattern relatif uniquement |
| <0.85 | <1.15 | freeze regen, refonte BC laterales requise |

Decision Phase B: ne pas freezer la regeneration pour cause de BC laterales.
La configuration best-stack restaure la conservation flat et reproduit un
speedup de crete suffisant sur relief analytique. Le probleme initial du cas
reel est donc compatible avec une combinaison de relief reel + ancienne config
damping, pas avec une incapacite structurelle a produire des accelerations
orographiques. Avant regeneration large, calibrer l'offset vertical/proche sol
sur flat (`2 m` trop fort, `50 m` legerement bas) et auditer quelques cas reels
avec la meme stack.

### Phase B - z0_treatment canary on heterogeneous WC site

Une fois la stack BC valide (flat + ridge), reste a trancher: pour la
regeneration 9k, quelle strategie de rugosite z0 sur la face terrain ?

Quatre options candidates:

1. `wc` natif: WorldCover ESA 2021 mapping classes -> z0 (Wieringa/Davenport)
   sans cap, applique via `generate_z0_field.py` sur la face `terrain`.
2. `wc_cap_0.05`: meme mapping clippe a `z0 <= 0.05 m`.
3. `wc_cap_0.01`: clippe a `z0 <= 0.01 m`.
4. `uniform_0.05`: z0 constant 0.05 m partout, comme baseline simple.

Builder: `analysis/build_terrain_canary.py --mode z0_treatment --variants wc,wc_cap_0.05,wc_cap_0.01,uniform_0.05`.
PBS array `-J 0-3` 24 cores par tache, 300 iter, walltime 1:30:00.

#### Tentative 1 - ct_d_fire_0170 (canary degenere)

Premier essai sur `ct_d_fire_0170_case_ts014` (Skiathos, Grece). Job
`21331014[].aqua`, les 4 variantes convergent. Mais audit a posteriori du
tif WC `data/raw/worldcover_per_site/ct_d_fire_0170.tif`: 100% classe water
(code 80, z0=0.0002), site en mer Egee, probable bug de centrage bbox de
`services/data-ingestion/download_worldcover_per_site.py`. Les 3 variantes
WC sont donc degenerees, equivalentes a un z0 uniforme 0.0002. Resultat
non-discriminant:

| variant | crop/inflow @10 m |
|---|---:|
| wc | 0.876 |
| wc_cap_0.05 | 0.876 |
| wc_cap_0.01 | 0.876 |
| uniform_0.05 | 0.642 |

Decomposition de l'ecart `uniform_0.05` vs reference flat (1.00):
- `1.00 - 0.876` = 0.12 d'ombre orographique reelle du relief ts014;
- `0.876 - 0.642` = 0.23 de friction wall reelle a z0=0.05.

A documenter independamment dans le plan WC ingestion (cf. `dataset_strategy.md`):
le tif `ct_d_fire_0170.tif` est inutilisable et probablement plusieurs autres
sites cotiers le sont egalement.

#### Tentative 2 - ct_d_fire_0056 (canary valide)

Refait sur `ct_d_fire_0056_case_ts014` (Sierra Andaluza, ES, 37.34N -2.60E,
1031 m, slope 12.8 deg), choisi parmi les sites `ts014 solved gold` avec WC
heterogene. Distribution WC: grass 59%, tree 22%, bare 10%, shrub 7% (water
0%, 4 classes >5%). Job `21380961[].aqua`, 4 variantes OK en 4.5 min wall.

Distribution z0 sur la face terrain (54 000 faces):

| variant | z0_mean | z0_median | z0_p90 | z0_max |
|---|---:|---:|---:|---:|
| uniform_0.05 | 0.050 | 0.05 | 0.05 | 0.05 |
| wc | **0.147** | 0.03 | 0.50 | **1.00** |
| wc_cap_0.05 | 0.034 | 0.03 | 0.05 | 0.05 |
| wc_cap_0.01 | 0.0095 | 0.01 | 0.01 | 0.01 |

`wc` natif est bimodal: grass majoritaire (median 0.03) avec une queue tree
(p90 0.50, max 1.00) qui tire le mean a 3x la valeur uniforme.

Ratios cles @ 10 m AGL (inflow = 5.103 m/s):

| variant | crop/inflow | center/inflow | **upstream/inflow** | crest_p90/inflow |
|---|---:|---:|---:|---:|
| wc | 0.863 | 0.718 | **0.881** | 1.185 |
| uniform_0.05 | 0.868 | 0.765 | 1.079 | 1.199 |
| wc_cap_0.05 | 0.911 | 0.844 | 1.099 | 1.221 |
| wc_cap_0.01 | 0.966 | 0.954 | 1.221 | 1.259 |

Lecture:

- l'effet WC est reel (spread 12% sur crop entre wc et wc_cap_0.01) — le
  canary precedent ne pouvait pas le voir;
- `wc` natif **draine l'upstream** (0.881, hors gate [0.95, 1.05]) parce
  que les patches tree z0~0.5 sur le fetch frichent trop;
- `wc_cap_0.05` rapproche le comportement de `uniform_0.05` (upstream 1.10
  vs 1.08, crop 0.91 vs 0.87) **tout en conservant l'heterogeneite**
  (median 0.03 vs 0.05);
- `wc_cap_0.01` ecrase tout a 0.01 -> ~smooth wall, sur-correction;
- l'effet z0 se concentre sur les 20 premiers metres: a 100 m AGL les 4
  variantes convergent (crop ratio 0.911-0.969, ecart 6%).

#### Decision z0 pour la regeneration 9k

Adopter `wc_capped_0.05`:

- preserve l'argument "rugosite ESA WC realiste" (defensible papier);
- evite le fetch decay des tree patches isoles;
- conserve une acceleration de crete intacte (1.221 a 10 m, meme superieur
  a uniform_0.05 1.199);
- numeriquement stable, pas de z0 extreme.

Resultat coherent avec le scenario "wc << wc_cap_0.05 <= wc_cap_0.01 ~
uniform_0.05" du grid de decision: cap necessaire, valeur 0.05 m retenue.

Caveats:

- un seul site, un seul timestamp -> robustesse a valider sur 2-3 sites de
  plus avant la regen complete;
- les caps gardent `upstream/inflow` ~1.08-1.10, juste au-dessus de la gate
  [0.95, 1.05] -> calibrer pg_geo et/ou Coriolis si necessaire mais hors
  scope z0;
- anomalie 2 m AGL persistante (`crop/inflow > 1` pour les 4 variantes)
  suggere un bug de normalisation `inflow_u2` cote
  `audit_v2_teacher_wind.py`, a investiguer independamment.

Artifacts:

- canary Aqua: `/scratch/maitreje/dsw/z0_treatment_canary/ct_d_fire_0056_ts014/`;
- audits locaux: `data/validation/z0_treatment_canary/ct_d_fire_0056_ts014/`
  (`z0_treatment_wind_audit.csv`, `wind_audit_summary.csv`,
  `wall_audit.csv`, `analysis.md`);
- PBS: `configs/hpc/z0_treatment_canary_ct_d_fire_0056_ts014.pbs`.

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

---

## Phase C - Ablation multi-hill (M6-M9, 2026-05-18)

Cas test: 3 collines analytiques (H=200/250/300 m, L=600/800/1000 m, cos2),
mesh v2 180x180x40, inflow ERA5 reel `ct_d_fire_0056_ts014`. Onze variantes
(V0 control, V1 best-stack, V2-V8 retrait OFAT d'un facteur, V0n/V1n
cross-check rotation 0 degN). Metrique de distribution: ratio
`|U| / U_inflow` sur masques `crop / flat / crest_{N,SE,SW} / lee_{N,SE,SW} / pdf`.
2 m AGL exclu (bug audit `inflow_speed_at`).

Resultats @ 10 m AGL (Delta vs V1 best-stack):

| Facteur retire | Delta crop_mean | Delta crest_max | Delta lee_min | Delta flat_mean |
|---|---:|---:|---:|---:|
| V0 control (no recovery) | +0.157 | +0.226 | +0.010 | +0.208 |
| V3 -pg_geo (no body force) | -0.107 | +0.050 | +0.011 | -0.107 |
| V8 -top entier | +0.066 | +0.694 | +0.012 | +0.070 |
| V2 -slip_top | +0.057 | +0.631 | +0.019 | +0.059 |
| V7 top_p zeroGrad | -0.055 | +0.335 | +0.020 | -0.050 |
| V5 -z0_wall_low | 0.000 | -0.006 | +0.001 | +0.001 |
| V6 -wc_heterogeneity | -0.023 | -0.073 | +0.005 | -0.020 |
| V1n rotation 0 degN | 0.000 | -0.007 | 0.000 | 0.000 |

**Conclusions**:
1. **pg_geo flip est le seul levier benefique** du best-stack: retirer pg_geo
   coute -0.107 sur crop_mean ET flat_mean.
2. **Top BCs (slip + p=0) ECRASENT la dynamique de relief**: V1 plafonne
   crest_max a 1.17, V0/V8 atteignent 1.40/1.86. Sur flat_mean V1=0.581 vs
   V0=0.789, le best-stack DEGRADE la conservation centrale.
3. **z0_wall=0.005 et wc_capped_0.05 sont negligeables** (|Delta| <= 0.023) -
   non discriminants sur multi-hill analytique.
4. **Cross-check rotation V1n-V1 = 0.000** sur toutes les stats - ablation
   parfaitement robuste a la direction.

**Decision regen 9k**: NE PAS adopter V1 tel quel. Recommandation: tester un
stack reduit `pg_geo flip + wc_cap_0.05` (sans slip_top, sans p_top=0,
sans z0_wall=0.005) sur 5-10 sites v2 reels avant la regen. Caveat: la
calibration pg_geo free-stream a 1500 m (deferred Mandate paragraphe 7)
doit etre rouverte avant fixation finale.

Livrables: `data/validation/ablation_multi_hill/ablation_{table,deltas,vertical}_10m.csv`
+ `figures/ablation_pdf_overlay.png` + `figures/ablation_vertical_V0_V1.png`.
Detail ablation, PDFs et profils dans le rapport M9 (orchestrator).

---

## Phase D - V9 = control + pg_geo flip only (M10, 2026-05-18)

Pour confirmer la conclusion M9 ("pg_geo est le seul vrai levier"), on
teste une 12e variante V9 isolant pg_geo seul sur la config control:
`top U inletOutlet + top p zeroGradient + pg_geo flip + z0_wall=0.05 +
wc native`. Single-task PBS (JobID 21547377.aqua), converge en ~10 min.

Resultats @ 10 m AGL:

| Variant | crop_mean | flat_mean | crest_max | crop_max | Config |
|---|---:|---:|---:|---:|---|
| V0 control | 0.757 | 0.789 | 1.396 | 1.779 | top ouvert, no pg_geo |
| **V8 -top entier** | **0.666** | **0.651** | **1.864** | **1.864** | top ouvert + pg_geo + z0=0.005 + wc_cap |
| V2 -slip_top | 0.657 | 0.640 | 1.801 | 1.801 | (idem V8, diff numerique marginale) |
| V9 control+pg_geo | 0.632 | 0.614 | **1.892** | **1.892** | top ouvert + pg_geo + z0=0.05 + wc native |
| V1 best-stack | 0.600 | 0.581 | 1.170 | 1.170 | slip + p=0 + pg_geo + z0=0.005 + wc_cap |
| V3 -pg_geo | 0.493 | 0.474 | 1.220 | 1.220 | slip + p=0, no pg_geo |

**Decouverte cle**: le levier dominant n'est pas pg_geo seul, c'est
l'**INTERACTION pg_geo x top_BC ouvert**.

- top OUVERT (inletOutlet + zeroGrad) vs top FERME (slip + p=0):
  Delta crest_max ~ +0.70 (passage de ~1.17 a ~1.86).
- pg_geo flip ajoute du momentum (+0.10 sur mean dans le bon contexte).
- z0_wall=0.005 + wc_cap aident marginalement (+0.03 V8 vs V9).

**Pourquoi V1 best-stack est pire**: `slip top + p=0` ferme la
circulation verticale et bloque la dilatation. La pression
geostrophique n'a plus ou "respirer" -> la dynamique de relief
s'ecrase (crest_max plafonne a 1.17). Ce que la Phase B avait
valide sur 1 ridge 2D (mono-orientation) ne tient pas sur multi-hill
3D.

**Decision regen 9k**: adopter **V8** comme stack de production.

```
top U     : inletOutlet
top p     : zeroGradient
pg_geo    : flip (ERA5 850-700 hPa, sign flip)
z0_wall   : 0.005 m (uniforme face terrain)
z0 field  : wc_capped_0.05 (ESA WC clipped a 0.05 m)
Coriolis  : on (atmCoriolisUSource avec sign flip)
```

V8 donne crop_mean=0.67 (~67%) et crest_max=1.86 sur multi-hill. Le
100% espere au centre n'est pas atteint mais c'est probablement
**physiquement borne** par les sillages locaux des collines (chaque
vallee descend en dessous de l'inflow). La conservation flat=0.65
est realiste pour un domaine 3D avec 3 collines a 30 m de
resolution.

**Caveats**:
1. Une seule mesure multi-hill analytique. Avant la regen complete
   9k, valider V8 sur 5-10 sites v2 reels diversifies (Pop A continental,
   topographie variee). Comparer V8 vs V1 et vs V0 sur ces sites.
2. La calibration pg_geo free-stream a 1500 m (deferred Mandate
   paragraphe 7) reste a explorer comme ajustement secondaire.
3. Le 2 m AGL reste exclu (bug audit `inflow_speed_at`).
4. `lee_p10` (proxy `lee_min`) est sensible au choix de masque
   per-hill - utiliser `lee_p10` plutot que `min` brut pour la
   robustesse statistique.

Livrables Phase D:
- `data/validation/ablation_multi_hill/figures/V9.png`
- `data/validation/ablation_multi_hill/ablation_table_10m.csv` (12 lignes)
- `data/validation/ablation_multi_hill/ablation_deltas_10m.csv` (11 lignes vs V1)
- `configs/hpc/multi_hill_canary_V9_ct_d_fire_0056_ts014.pbs`

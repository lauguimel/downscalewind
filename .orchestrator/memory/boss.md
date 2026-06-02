# Boss memory — DownscaleWind orchestrator

## Site-selection lesson (2026-05-15)

**Before** picking a canary site, always filter sites.csv by
`case_status='solved'` AND `tier in {gold, silver}` on the relevant
`case_ts<NN>`. The freeze manifest
`data/campaign/complex_terrain_v1/manifests/dataset_v2_status.csv` is
authoritative. Approximately 2857 / 12109 cases are `rejected` (mostly
`diverged`) — Pop B (steep terrain) hit RANS convergence limits,
typically slope > 25° + elev > 2500 m.

Why: a Department's WC-heterogeneity audit alone will happily pick
beautiful alpine sites that never converged in v1.

How to apply: in any future canary selection, filter solved-and-gold
FIRST, then rank by WC heterogeneity / z0 contrast on the survivors.

## Case_status vocabulary

The `case_status` column uses `solved` / `diverged` / `early_converged`
/ `mesh_failed` — NOT `ok` / `fail`. The OF solver status JSON written
by `run_multisite_campaign.solve_case` uses `ok` / `solve_failed`.
Don't conflate them.

## Canary base case naming

z0_treatment canary base cases under `/scratch/maitreje/dsw/z0_treatment_canary/<SITE>_ts014/`.
PBS template lives under `configs/hpc/z0_treatment_canary_<SITE>_ts014.pbs`.
PBS submits a `-J 0-3` array (4 variants : wc, wc_cap_0.05, wc_cap_0.01, uniform_0.05).

## 2 m AGL anomaly is an audit bug, not physics (2026-05-18)

The `crop/inflow > 1.0` at 2 m AGL observed on flat canary (1.343 with
best-stack, all four z0_treatment variants on 0056) is an
`inflow_speed_at` normalization issue in
`audit_v2_teacher_wind.py` (and any audit script reusing the same
helper). It is NOT a physical over-acceleration near the wall.

How to apply: exclude 2 m AGL from OFAT decisions in M9 until the audit
script is fixed. Decisions should rest on 10/20/50/100 m AGL stats.

## Stack regen 9k : V8 retenu, pas V1 (2026-05-18, Phase D M10)

Phase D a confirmé : **V8 est le stack optimal**, pas V1.
- V8 = `inletOutlet top U + zeroGrad top p + pg_geo flip + z0_wall=0.005 + wc_capped_0.05`
- V8 @ 10m : crop_mean=0.666, flat_mean=0.651, crest_max=1.864
- V1 @ 10m : crop_mean=0.600, flat_mean=0.581, crest_max=1.170

V9 (control + pg_geo seul, sans z0/wc tuning) : crop_mean=0.632,
crest_max=1.892. V8 légèrement supérieur sur mean (+0.034), V9
marginalement supérieur sur crest_max (+0.028). V8 retenu.

**Levier réel = INTERACTION pg_geo × top_BC ouvert**. Top fermé (slip
+ p=0) bloque la dilatation verticale → la pression géostrophique
n'a plus où respirer → dynamique de relief écrasée. Ce que la Phase B
ridge 2D mono-orientation avait validé ne tient pas sur 3D
multi-orientation.

Avant la regen 9k : valider V8 sur 5-10 sites v2 réels diversifiés
(Pop A continental, topographies variées). C'est la **Phase E** à
ouvrir comme nouvelle mission orchestrator.

## Best-stack écrase la dynamique sur multi-hill (2026-05-18, M9)

L'ablation OFAT propre toutes-choses-égales sur multi-hill (3 collines
analytiques, mesh v2, inflow ERA5 ct_d_fire_0056) RETOURNE la décision
Phase B :
- V0 control : crop_mean=0.757, flat_mean=0.789, crest_max=1.40
- V1 best-stack : crop_mean=0.600, flat_mean=0.581, crest_max=1.17
- V8 -top entier : crest_max=1.86 (le plus haut)

**Top BCs (slip_top + p_top=0) écrasent la dynamique de relief**.
Phase B avait validé best-stack sur 1 ridge 2D mono-orientation —
en 3D multi-orientation, slip_top ferme la circulation top et bride
les accélérations sur crêtes.

**Seul pg_geo flip est un vrai gain** : Δ crop_mean = -0.107 quand
retiré. Confirmé robust par cross-check rotation 0°N
(V1↔V1n |Δ|≤0.001).

**z0_wall=0.005 et wc_capped_0.05 sont non-discriminants** (|Δ|≤0.023)
sur OFAT propre — les décisions M3/M5 silos cross-site ne tiennent
pas.

**Pour la regen 9k : NE PAS adopter V1 tel quel**. Stack candidat
réduit `pg_geo flip + wc_cap (or uniform)` SANS slip_top / sans
p_top=0 / sans z0_wall=0.005 — à valider Phase D sur sites v2 réels
avant fixation finale.

## Diagnostic stall trap: trust the JSON, not qstat (2026-05-18)

When monitoring a PBS array, the absence of qstat output (`Unknown Job
Id`) only means the array left the queue, NOT that tasks failed. Tasks
write their own `solve_status/*.json` AFTER their last successful step.
If a non-critical late step (e.g. matplotlib in audit) crashes,
`solve_status` stays at the last-written value ("running") even though
the solver/export both completed and `grid.zarr` is on disk.

**Verification protocol BEFORE declaring a task failed**:
1. `ls .../cases/<V>/grid.zarr` — does the data exist?
2. `tail log.simpleFoam` — does it end with `End / Finalising parallel
   run`? (Then it's a post-solve issue, not a solver fail.)
3. `cat solve_status/*.json` — what was the last status?

Only after all three say "failed" is the variant actually broken.
The M8 mis-diagnostic (declared 7/11 failed when 11/11 succeeded)
wasted ~1 h of orchestration time. Don't repeat.

## libgeostrophicPressureGradient .so warning is cosmetic (2026-05-18)

In simpleFoam runs with `codedSource` fvOptions (e.g. our
geostrophicPressureGradient), the warning
`Could not load .../dynamicCode/.../lib<name>_<hash>.so` printed
AFTER `End / Finalising parallel run` is an OF runtime cleanup
artifact. The solver converged. Do NOT treat it as a failure
signal.

## WC tif coastal-bug — global audit pending (2026-05-18)

`download_worldcover_per_site.py` is suspected to have a bbox-centering
bug for Mediterranean coastal sites: `ct_d_fire_0170.tif` (Skiathos)
came back 100% water. Other Mediterranean coastal sites in the 9k
campaign may share the bug. The fix and the global audit are
**out of scope** for the multi-hill ablation mission, but must be
opened as a separate task before any 9k regeneration.

How to apply: when selecting a base-case inflow for a canary, always
re-verify the WC class distribution of the site (look at the tif
or the `freeze_dataset_v2_status` row) BEFORE using it. Continental
sites (>50 km from coast) are safe by construction.

## build_terrain_canary.py LOC over ceiling (2026-05-18)

Post-M7 additions push `build_terrain_canary.py` to **1020 LOC**
(hard ceiling 700, soft 500). Department M7 flagged this for a future
SPLIT mission — split `multi_hill` mode into a sibling
`build_multi_hill.py` (or a `canary/` sub-package). NOT a blocker for
M8/M9 because the new code is structured in dedicated private
functions, but any *substantive* future change to this file MUST start
with the SPLIT, not a 5-LOC patch.

## Prior canary table is silo'd cross-site (2026-05-18)

The MANDATE §0 summary table mixes rows from ct_d_fire_0170_ts014 (WC
bug → effectively z0=2e-4), ct_d_fire_0056_ts014 (real WC), and pure
analytic flat/ridge derived from 0170 inflow. The increments
("slip_top → +0.04") cannot be added or compared as if from the same
ablation. This is exactly why M6→M9 ablation is being done on a
single multi-hill mesh with one inflow.

## Sources OBS Phase G — 2026-05-21 RED + pivot NOAA ISD

**Bilan smoke audit 2026-05-21** :

- **M_G2 SYNOP Météo France RED** : URL `donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/postesSynop.csv` retourne une page HTML "Données non disponibles". URL invalidée par le provider. Code OK mais source bloquée.
- **M_G4 OGIMET RED** : parser HTML ne trouve plus la table dans les pages OGIMET (format serveur changé entre l'écriture de la spec et l'exécution). lxml installé OK mais structure HTML attendue absente.
- **Open-Meteo Historical Weather API = ERA5 reanalysis sur grille**, pas obs in-situ. **NE PAS** l'utiliser comme proxy OBS pour benchmark surrogate v2 → circulaire (le surrogate downscale déjà ERA5).
- **Open-Meteo Forecast API = IFS** : non-circulaire mais pas obs in-situ non plus. Utilisable comme baseline intermédiaire si besoin.

**Pivot 2026-05-21 = NOAA ISD** (Integrated Surface Database) :
- `ftp://ftp.ncdc.noaa.gov/pub/data/noaa/` ou Climate Data Online API
- ~12k stations EU, SYNOP-derived, vraies obs in-situ
- Hourly, multi-decade, vent 10 m AGL + T + RH
- Format ISH (Integrated Surface History) à parser

How to apply : avant d'investir dans un nouveau pipeline d'ingestion "obs", vérifier que la source est bien in-situ (pas reanalysis). Le critère discriminant : si la source fournit des données à toute coordonnée arbitraire = c'est du modèle/grille, pas des stations.

## Erreur Boss rm hâtif (2026-05-21) — lesson learned

J'ai supprimé `_obs_unified.py` et `utils/obs_zarr_writer.py` en pensant qu'ils étaient des orphelins du 1er kill M_G3, sans avoir vérifié les imports. **Ils étaient en réalité utilisés par les 4 scripts ingest_* livrés par M_G1/G1.5/G4**. Heureusement récupérables depuis les engineer logs (Codex apply_patch contient le code complet).

How to apply : avant tout `rm` de fichiers non-tracked, exécuter
`grep -r "filename_without_ext" --include='*.py'` pour vérifier
qu'aucun import ne les référence. CLAUDE.md warning "investigate before
deleting" était précisément pour ce cas.

## Diagnostic M_H'1d′ direction 2026-06-01 — faits mesurés (corriger erreurs antérieures)

**Erreurs Boss de la session à NE PAS reproduire** :
- J'ai d'abord affirmé "cache 8755/49%" : FABRIQUÉ (commandes annulées en cascade jamais
  exécutées). Le vrai cache `phase_H_prime_M_H1_jja_cache` = **60971 dirs présents / 63022
  pairings parquet = 97%**. Les 2051 manques = dates bordure mai/sept (hors JJA). Toutes les
  214 stations couvertes (val-43 : 12061 pairings cachés).
- Leçon : ne JAMAIS écrire un chiffre sans l'avoir lu dans un tool_result réel. Une commande
  bash qui échoue (ex `conda: command not found` en SSH non-interactif → exit 127) **annule
  en cascade tout le batch parallèle** (Write/Edit/Agent inclus) → rien n'est persisté, c'est
  un faux sentiment de progrès. Charger conda en SSH : `module load Miniconda3/24.9.2-0;
  eval "$(conda shell.bash hook)"; conda activate fuxicfd`.

**Parquet `noaa_jja2023_v2.parquet` (63022 rows, 214 stations) = sortie RAW surrogate v2**
(inférence M_G7). Colonnes : `speed_pred,u_pred,v_pred,w_pred` (RAW, PAS corrigé ANN),
`speed_obs,u_obs,v_obs,wind_dir`? non — `u_obs,v_obs,speed_obs`, `speed_era5_baseline,
u10_era5_baseline,v10_era5_baseline`, `era5_time_delta_minutes` (médiane 0 = on-time).
→ Le chiffre CORRIGÉ M_H'1a n'est PAS dans le parquet → eval forward GPU nécessaire.

**Direction (surrogate RAW vs OBS, val-43, mesuré)** :
- surrogate mae_dir **43.3°** med 27.4° biais **+16.4°** (rotation horaire systématique)
- ERA5 mae_dir 34.4° med 19.7° → **le surrogate raw DÉGRADE la direction vs ERA5**
- 22/43 stations val médiane dir >30° ; 15% des pairings >90°
- **Nuance user (importante)** : "différent d'ERA5 ≠ faux" — la déflexion relief-aware du
  surrogate est attendue/désirable ; SEULE la comparaison vs OBS tranche. Ici vs OBS le raw
  est pire qu'ERA5 → déflexion mauvais sens/trop forte. À confirmer sur le CORRIGÉ.
- **Structurel** : ANN M_H'1a entraîné loss VITESSE seule (pas de terme angulaire) → aucune
  incitation à corriger la direction, et peut la dégrader en bougeant u,v pour la norme.
  DEVINE 2024 avait DEUX réseaux séparés (ANN_speed + ANN_direction).

**Vitesse (surrogate RAW vs OBS, val-43)** : mae 1.60 biais -1.06 ; classe high>6m/s mae 2.96
biais **-2.88** (compression slope-0.59 connue). NB : 12488 pairings val ⊃ 3015 val training
→ pas exactement apples-to-apples avec le 1.812 mandate.

**Décision user 2026-06-01** : faire l'eval GPU corrigé (réparer eval_devine_loso.py : filtre
cache au lieu du garde-fou tout-ou-rien). Mesurer dir corrigée vs OBS + déflexion corrigé−ERA5
stratifiée terrain pour distinguer relief-aware utile vs bruit.

## M_H'1d″ eval corrigé GREEN (vitesse) 2026-06-02 — job 22162041 exit 0, walltime 1h09

VITESSE CONFIRMÉE DÉCISIVEMENT (43 stations held-out, channels=4, reproduit exactement
le training M_H'1a) :
- val mae_corrected **1.223** / raw 1.812 / era5 1.315 → **−32.5%**, biais +0.008
- train mae_corrected 1.236 / raw 1.796 → **−31.2%** (gap train↔val quasi nul = pas d'overfit)
- Le corrigé BAT ERA5 (1.223 < 1.315) sur stations jamais vues.
- Par classe vent (val) : faible<3 corr 1.07 / raw 0.88 / era5 0.86 (correction sur-corrige
  le calme) ; moyen 3-7 corr 1.08 / raw 1.76 / era5 1.25 ; fort>7 corr **1.75** / raw 3.58 /
  era5 2.25 (divise l'erreur par 2, bat era5). → gain là où ça compte (vent fort = fire weather).
- Cache inchangé 60971. Couverture 98.2% (kept 60971/62062, dropped mai/sept hors JJA).

DIRECTION = MODULE EVAL BUGGÉ, NE PAS UTILISER ces chiffres :
- eval donne raw 67° / corr 71° / **era5 74°** ; or recalcul direct parquet (même convention
  meteo from-dir) = raw 43° / **era5 34°**. ERA5 (aucun modèle, juste u10/v10) ne peut valoir
  34 ET 74 → bug convention/appariement dans le volet direction de eval_devine_loso.py.
  Corrélation déflexion-terrain (~0, pearson 0.01/−0.08) en hérite → non fiable.
- FIABLE (recalcul direct val-43 vs OBS) : surrogate RAW 43° DÉGRADE vs ERA5 34°. 22/43 >45°.
- **La direction du modèle CORRIGÉ reste non mesurée proprement** (le seul outil qui l'avait
  est buggé). Question "ANN speed-only aide/empire la direction ?" → OUVERTE.

Verdict : SPEED-only validé production-scale honnête (paper NatComms fire : −32% MAE summer,
bat ERA5, surtout vent fort). Direction = (a) soit fixer le bug direction de eval_devine_loso.py
+ re-run pour trancher, (b) soit déférer (fire/FWI = vitesse-dominé) et passer M_H'1c Perdigão IOP.

## M_H'1e DIRECTION verdict 2026-06-02 — bug fixé, ANN speed-only AMÉLIORE la direction

Job 22194874 (H100, walltime 39min). Exit_status=1 mais COSMÉTIQUE (le bloc de vérif PBS
cherche encore l'ancien `loso_summary.json`/colonnes obsolètes ; les vrais livrables
`loso_summary_dir.json` + `pairing_dir.parquet` sont écrits OK à 09:52). À nettoyer : adapter
le PY de vérif du PBS au nouveau schema dir.

SANITY OK : recompute parquet-only ALL val raw 41.2° / era5 32.1° (≈ recalcul Boss 43/34)
→ bug du +33° fantôme (2e obs-store join) bien corrigé.

**Direction val-43 (agrégé sur pairings, speed≥1, arbitre OBS)** :
| | corrigé | raw | era5 |
|---|---|---|---|
| MAE dir | **36.1°** | 41.2° | 32.1° |
| médiane | 22.6° | 26.0° | 18.7° |
| biais | +11.8° | +20.1° | +5.2° |

**L'ANN speed-only (loss SANS terme directionnel) AMÉLIORE la direction** : 36.1° vs raw 41.2°
(−5°), biais +20→+12°. Confirme la nuance user : en bougeant u,v pour la vitesse, la correction
redresse partiellement la déflexion relief-aware excessive du surrogate brut (PAS du bruit).
Ne rattrape pas tout à fait ERA5 (32°) mais l'écart est petit.

Par classe vent (val, MAE dir corr/raw/era5) :
- calme<1 : 90/88/85 (non informatif, exclu headline)
- faible 1-3 : 55/63/50
- moyen 3-7 : 29/34/25
- **fort>7 : 15.9/17.2/13.4** ← régime fire/éolien : direction corrigée excellente, ~égale ERA5

Distribution corrigée : 35% pairings <15°, 60% <30°. Train cohérent (33.3/38.4/29.2, gap
train↔val faible). `pairing_dir.parquet` persisté → toute ré-analyse dir future = CPU-only.

**VERDICT GLOBAL M_H'1a** : speed-only VALIDÉ. Vitesse −32.5% bat ERA5 ; direction non dégradée
(améliorée). Pas besoin d'ANN_direction pour fire/FWI (vent fort dir≈ERA5). ANN_direction =
optionnel futur pour track éolien si on veut fermer les ~4° résiduels en vent faible.
**Next : M_H'1c Perdigão IOP** (propagation spatiale 41 stations dans 6×6 km).

## M_H'1c Perdigão IOP verdict 2026-06-02 — propagation LISSE ✅, mais OOD au centre ⚠️

Job 22195330 (A100, exit 0, 5min, 328 pairings sous-éch. 8ts/station, override store
era5_europe_spring2017_v2.zarr Δt=6h). H100 22195325 tué (redondant). JJA cache intact 60971.

**PROPAGATION SPATIALE (test principal) = PARFAITE** :
- pct_smooth **100%** (0 pic), spike_ratio médian 1.01, p90 1.04
- delta(ring_k)/delta(centre) : ring1 1.01, ring2 0.99, ring3 0.94, ring5 0.90 → décroissance douce
- → la correction NE overfite PAS le pixel calibré ; elle s'applique de façon spatialement
  cohérente sur tout le voisinage. **Verrou propagation aux pixels non-station LEVÉ.**

**JUSTESSE AU CENTRE = DÉGRADE sur Perdigão (inverse du gain NOAA)** :
- MAE centre corr **2.27** vs raw **1.37** ; biais corr +1.92 vs raw +0.58 → sur-correction.
- Cause = DISTRIBUTION SHIFT, pas propagation : ANN entraîné sur JJA-2023 NOAA plaine FR/ES/PT
  (régime vent fort sous-estimé) appliqué à Perdigão mai-juin 2017, double crête microrelief,
  vent faible (raw déjà bon biais +0.58). L'ANN pousse trop haut un régime qui n'en a pas besoin.
- Les 2 résultats sont COHÉRENTS : propagation = propriété structurelle (lisse partout) ;
  justesse = dépend du régime d'entraînement. Perdigão IOP = OOD pour cet ANN JJA-plaine.

**Implications** :
- Architecture DEVINE-style VALIDÉE (propagation physique cohérente confirmée empiriquement).
- M_H'1a déployable POUR son domaine (été, plaine/colline EU, NOAA-like). PAS pour terrain
  alpin/microrelief hiver/printemps sans ré-entraînement incluant ces régimes.
- **Renforce le besoin de M_H'1b 4-saisons + features physiques** : un seul modèle JJA-plaine
  ne couvre pas Perdigão. M_H'1b (winter+MAM+JJA+SON + gradient T/RH) doit réduire ce shift.
- ⚠️ Pour le papier : NE PAS présenter Perdigão comme validation de justesse (c'est OOD) ;
  le présenter comme validation de PROPAGATION (100% lisse) + honnête caveat OOD régime.

**Décision next** : M_H'1b multi-season (le shift Perdigão justifie l'enrichissement saisonnier).

## M_H'1a JJA TERMINÉ GREEN 2026-06-01 — Phase H' DEVINE-style validée à production-scale

Job 22135584.aqua, walltime 11h15, exit 0, 8 epochs complets sur 12061 train + 3015 val pairings JJA 2023 (~214 stations FR/ES/PT held-out random val 20%).

**Verdict empirique massif** :

| Métrique | Raw surrogate v2 | M_H'1a corrigé | Gain |
|---|---|---|---|
| **val_mae** | 1.812 m/s | **1.222 m/s** | **−32.5%** |
| **val_bias** | −1.453 m/s | **+0.008 m/s** | bias quasi-nul |
| **best epoch** | — | 5 | converge fast |

Config V2 winner (arbitrage Dept A vs Dept B) : lr=5e-4, grad_clip=1.0, τ=0.6/0.4 DEVINE default, batch_size=4, 8 epochs. Stable, monotone, pas d'overshoot epoch 1.

**Pourquoi JJA gain 3× smoke v3 winter** : surrogate v2 raw a été entraîné sur cas CFD k-ε neutre. En été = vent thermique + canicule = régime physique le plus loin de l'entraînement. Bias raw -1.45 m/s en JJA vs -0.80 en winter = plus de marge à corriger. DEVINE-style comble.

**Implications stratégiques** :
- **Paper NatComms fire weather** : preuve concrète -32% MAE summer wind sur 214 stations EU held-out, bias=0 → exploitable FWI direct
- **Startup downscaling** : surrogate v2 raw était NO-GO (REGRESS -8% vs ERA5). Surrogate + correction DEVINE-style bat ERA5 raw clairement

**Caveats restants** :
- Validation Perdigão IOP (propagation spatiale 41 stations dans 6×6 km) PAS encore faite
- **CORRECTION 2026-06-01** : le split val EST watertight station-disjoint (`watertight_station_split` dans dataset_v2_obs_centered.py:385, appelé inconditionnellement train_v2_devine_style.py:285). val = ~44 stations FR/ES/PT JAMAIS vues à l'entraînement (20% des stations, perdigao exclu). Donc **val_mae 1.222 EST déjà le chiffre LOSO honnête**. L'ancienne note "random 20% pairings" était FAUSSE. La mission M_H'1d "re-train honnête" était un no-op → Department BLOCKED à raison (économise 9h H100).
- 3 saisons restantes (winter/MAM/SON) à faire (loader actuellement mono-era5_store, refactor nécessaire)
- 21 stations Perdigão exclues à raison du flag mandate §5, mais inclues comme test propagation explicite séparée

**Plan suivant** :
1. Validation Perdigão IOP sur best.pt M_H'1a → vérifier que la correction généralise aux pixels non-station
2. LOSO honest spatial sur M_H'1a (re-split 80/20 stations différentes, eval)
3. M_H'1b multi-season (refactor loader + train sur 264k pairings 4 saisons)
4. Si Perdigão GREEN + LOSO honest GREEN → M_H'3 audit final + GO déploiement

## M_H'0 smoke v3 GREEN 2026-05-29 — pipeline DEVINE-style valid empiriquement

Smoke v3 (200 stations × winter2223 × 25k train + 6k val × WC tiles ESA enabled × 15 epochs target, killed walltime à 6 epochs) :

| Epoch | train_mae | val_mae | val_mae_raw | Δmae |
|---|---|---|---|---|
| 0 | 1.498 | 1.405 | 1.602 | **-0.196** |
| 2 | 1.421 | 1.407 | 1.602 | -0.195 |
| 5 | 1.399 | 1.417 | 1.602 | -0.185 |

**Verdict net** (vs smoke v1 ambigu) :
- train + val descendent ensemble (pas d'overfitting)
- **Δmae STABLE -0.18/-0.20 sur 6 epochs** (vs v1 oscillait -0.16/+0.02)
- 0 WC tile missing warnings (vs v1 = 2354)
- Convergence rapide : epoch 0-2 trouve l'amélioration, epochs 2-5 la stabilisent

**Caveat structural** : val_bias passe -0.796 (raw) → +0.5 (corrigé) = **over-correction** par τ asymétrique 0.6/0.4. Le modèle saute le bon point. Fix : tuner τ à 0.55/0.45 ou ajouter loss term `λ × |bias|`.

**Compute estimate pour M_H'1 full** :
- Smoke v3 : 25k train pairings × 33 min/epoch = ~80 ms/pairing
- Full 264k × 30 epochs × batch_size=4 = ~88h ❌ trop long
- Avec batch_size=16 + bf16 + epochs=7 (convergence rapide observée) = ~10h ✅
- Ou batch_size=8 + epochs=10 = ~30h

**Plan M_H'1 full** :
1. Tuner τ à 0.55/0.45 sur smoke court (~2h pour valider bias closer à 0)
2. Full training sur 264k pairings, batch_size=16 (si A100 80GB) ou 8 (si 40GB), bf16, 7-10 epochs
3. Validation : LOSO spatial split + Perdigão IOP 41 stations propagation test

## Phase H' pivot DEVINE-style validé 2026-05-27 — abandon Strategy A E.2

**Décision Boss + user 2026-05-27** : après verdict M_H1 YELLOW (toggle_delta plat 3.4e-6 à epoch 29, canal OBS jamais activé) et lecture du paper DEVINE 2024 (Le Toumelin et al. NPG), **abandon de l'option E.2** (canal OBS in-model) et pivot vers **option B/D revisited inspirée DEVINE** :

```
ERA5_input + topo_features → ANN(MLP small) → ERA5_corrigé → surrogate v2 FROZEN → champ pred
                                                                                       ↓
                                                                         loss au pixel central station seul
                                                                         (τ asymétrique 0.6/0.4)
```

**Architecture DEVINE-style** :
1. Pas de canal OBS in-model. OBS = target seul.
2. ANN simple (MLP 2-3 layers, ~quelques milliers de params) qui corrige l'**input ERA5** (vector 408)
3. Skip connection résiduelle (ANN apprend Δ vs absolute)
4. Surrogate v2 frozen consume corrected ERA5, produit champ pred
5. Patch centré sur station (pixel central 90,90 = station), pas random
6. Loss au pixel central avec τ asymétrique (penalize underestimation comme DEVINE : surrogate v2 slope 0.59 → underestime vent fort)
7. Backprop end-to-end à travers surrogate v2 frozen — l'ANN voit comment le surrogate transforme et apprend à anticiper la compression

**Pourquoi ça marchera (vs E.2 qui a échoué)** :
- DEVINE 2024 a fait exactement ça avec succès : MAE 1.42 m/s, -7.8% vs DEVINE seul, -15.7% vs raw AROME
- Pas de problème d'init faible (skip connection résiduelle absorbe l'init)
- Pas de problème de pixel random (station toujours au centre du patch)
- Pas de signal dilué (loss concentrée sur 1 pixel)
- Pas de loss qui n'incite pas à utiliser OBS (loss EST l'OBS)
- DEVINE est exactement notre setup : CNN surrogate downscaler frozen + small ANN correction in front

**Verrou résiduel** : propagation au reste du domaine présumée par architecture (surrogate v2 frozen propage cohérence physique). **Validation explicite** via **Perdigão IOP 41 stations dans 6×6 km** = golden test set propagation spatiale (seul dataset qui le permet).

**Plan secours si Phase H' plafonne** : recasting vers une architecture pixel-wise complète, ou retour aux approches statistiques (XGBoost stratifié). Mais DEVINE est notre meilleur pari empirique + théorique.

## M_H1 preflight GREEN + full submitted 2026-05-27 — training E.1 en cours

**Preflight 22027837** : 1 epoch sur 32 train + 16 val cases, 18.6s wall sur H100. Exit 0.
- val_mse = **0.097** (mieux que baseline 0.121, le surrogate s'améliore en partant des nouveaux poids OBS init petits + 1 epoch)
- toggle_delta = 2.4e-7 (plat, attendu à 1 epoch)
- toggle_mean_abs_pred_diff = 2.4e-5

**M_H1 full PBS 22029594.aqua RUNNING** depuis 09:29 le 2026-05-27.
- 30 epochs sur 6567 train + 1300 val cases full
- Monitor toggle test à epoch 5/10/15/20/25/30
- Auto-warning si Δ < 1e-4 à epoch 10 (escalate Boss avant gaspiller 6h+)
- ETA 6-12h

## M_H+ Axe 2 ERA5 chained jobs GREEN 2026-05-27 confirmé

Tous les 3 jobs ERA5 ont en réalité **réussi** (le "Time Use 24-46s" dans qstat -x = CPU time, PAS wall time). Wall réel :
- 21982386 (MAM 2023) : 1h51, DONE
- 21982395 (JJA 2023) : 3h11, DONE
- 21982396 (SON 2023) : 2h52, DONE

Stores : `/home/maitreje/dsw/data/raw/era5_europe_hourly_{mam,jja,son}2023.zarr/` 1.2 GB chacun, {coords, pressure, surface{u10,v10,t2m,d2m}} tous présents.

**M_H+ entièrement GREEN** (axe 1 DEM, axe 2 ERA5, axe 3 multiproc).

**Prochaine étape** : après M_H1 full GREEN convergé, lancer re-inference v2 production sur 4 saisons × 362 stations via `qsub -v SEASON=<...> configs/hpc/infer_at_stations_v2.pbs`. Cible parquet étendu ≥200k pairings.

How to apply : leçon "qstat -x Time Use = CPU time only, pas wall" → toujours lire wall séparément via PBS o file ou `qstat -f`. Le diagnostic prématuré sur 24-46s a failli mal interpréter un succès.

## M_H0_smoke YELLOW 2026-05-26 — pipeline OK, canal OBS pas encore activé (smoke trop court)

PBS 21980675.aqua terminé en ~5 min, val_mse=0.12139 identique au baseline 0.121.

**Toggle test révèle problème prévisible** :
- mse_drop0=0.12138843 vs mse_drop1=0.12138876 → Δ = 3.2e-7 négligeable
- mean_abs(pred_drop0 - pred_drop1) = 2.1e-5 (~0%)
- → **Le modèle n'utilise pas encore le canal OBS** après 1 epoch

**Diagnostic** : init `obs_init_std=1e-3` + 1 epoch × 8 cases (~400 updates) → obs_mlp n'a quasi pas bougé. Modèle s'est essentiellement maintenu à baseline (val_mse 0.121 inchangé). C'est **attendu et acceptable** pour un smoke aussi court.

**Validé par smoke** : forward path, init mapping (238 weights loaded, 4 new skipped), eval pipeline, MLflow logging, PBS chain training+eval. Infrastructure GREEN.

**NON validé par smoke** : utilité effective du canal OBS (besoin training plus long).

**Décision Boss (avec accord user) : proceed M_H1 full directement** sur 30 epochs × 6567 cases (~200k updates). Ajout d'un **monitor toggle-test périodique** (epoch 5, 10, 15, 20, 25, 30) comme proxy d'activation du canal — permet abort early si plat à epoch 10 au lieu de gaspiller 6h H100.

**Plan M_H1** :
- Init from `surrogate_v2_e2_stage1_smoke/best.pt` (essentiellement baseline)
- 30 epochs, full splits, walltime PBS 12h
- Monitor toggle test → bascule revise architecture (Strategy B/C/D) si plat à epoch 10

## M_H+ Department PARTIAL_GREEN 2026-05-26 — extension dataset OBS livrée, ERA5 chained PBS en cours

Mission lancée en parallèle de M_H0_smoke, terminée en ~35 min en background.

**Axe 1 (DEM tiles)** GREEN — 100 nouvelles tiles Copernicus DSM 30m via AWS S3 anonyme (`ingest_dem_copernicus.py`, 199 LOC). Couverture étendue : PT, sud ES, nord ES, UK sud, Atlantique côtier. Total 86 → 186 tiles. Resolve smoke validé sur Lisbon/Granada/Coimbra/Faro/Malaga/A Coruña.

**Axe 2 (multi-season ERA5)** PARTIAL — 3 jobs PBS chained `afterok` soumis pour MAM/JJA/SON 2023 (21982386 R + 21982395+96 H). ETA 24-48h. **2 fixes critiques** :
- `PRESSURE_LEVELS dataset_v2` réaligné `[1000,925,850,700,600,500,400,300,250,200]` (l'ancien engineer.md disait `[..., 200, 150]` SANS 600 = faux, vérifié 2026-05-26 sur grid.zarr réel)
- CDS 2026 cost-limit `403 "Your request is too large"` → mitigation `--max-days-per-req 7 --no-z`

**Axe 3 (multiprocessing)** GREEN — `ProcessPoolExecutor` dans `infer_at_stations.py` (+153 LOC) + PBS v2 (16 ncpus, n_prep_workers=8). Smoke 5× speedup (24/24 built, 34s wall). GPU 0% observé sur smoke court — attendu ≥60% en prod 300k pairings.

**Recommended next move** :
1. Attendre completion des 3 ERA5 chained jobs (24-48h). Le `afterok` chain les enchaine automatiquement.
2. Lancer re-inference v2 production : `qsub -v SEASON=mam2023 configs/hpc/infer_at_stations_v2.pbs` × 4 saisons après stores prêts.
3. Re-run audit sur parquet étendu `noaa_seasons_2022_2023.parquet`. Cible ≥200k pairings (winter actuel 57k + ~150k attendus depuis nouvelles tiles+saisons).

Memory candidates persistés : engineer.md §CDS 2026 cost-limit + §Copernicus DSM S3 endpoint + correction §PRESSURE_LEVELS. boss.md cette entrée.

## Phase H ouverte 2026-05-26 — option E.2 verrouillée (fine-tune surrogate v2 avec canal OBS)

**Décision user 2026-05-26** : Phase H = fine-tune **du surrogate v2 lui-même** avec un canal d'entrée OBS, plutôt qu'un 2e étage de correction post-process. Option E.2 = 2 étapes :
- **Stage 1 (E.1)** : fine-tune surrogate v2 sur dataset CFD existant (9252 cases) avec canal OBS synthétique = valeur CFD au pixel random + dropout 50%. Le modèle apprend à ancrer.
- **Stage 2** : fine-tune subséquent sur dataset OBS réel (~273k pairings après extension M_H+), loss au pixel station = OBS réelle. Le modèle apprend à corriger en plus d'ancrer.

**Décisions structurantes verrouillées** :
1. Granularité = surface (10 m AGL) seul. Pas de correction 3D, pas de classification a priori des régimes (thermique/catabatique).
2. Skip M_H0 research/comparaison A/B/D vs E (user fixe E.2 directement). Single Department Codex Engineer pour chaque mission.
3. Extension dataset (M_H+) = prérequis hard parallèle.

**Pourquoi E.2 plutôt que B/D post-process** : cohérence physique garantie par le receptive field du surrogate v2 lui-même (CNN/transformer 3D propage l'ancrage), pas de risque d'incohérence pixel-wise (option A) ni de paramètres dupliqués (option B). À l'inférence sans OBS : canal vide → surrogate fonctionne comme avant (zéro perte).

**Verrou conceptuel à surveiller** : les 9252 sites CFD training ne sont PAS co-localisés avec stations OBS. Stage 1 utilise OBS **synthétique** depuis ground truth CFD → le modèle apprend à recopier la valeur fournie au pixel. Stage 2 force la correction (loss OBS réelle) sur dataset étendu. Si dataset OBS reste insuffisant pour LOSO honest, risque que le stage 2 overfit climatologie au lieu de physique.

**Plan de secours si E.2 plafonne** : bascule vers option B (U-Net post-process sur champ entier) ou D (hybride U-Net pixel-wise training + full-conv inference) comme mission de secours en M_H3 (`RESCOPE Phase H'`).

**Cible empirique M_H2** : MAE global < 1.0 m/s + MAE wind_class=high < 1.5 m/s + bias wind_class=high > -0.5 m/s.

**Missions lancées 2026-05-26 en parallèle (background)** :
- **M_H0_smoke** : Department spawné, Codex/Claude Engineer pour smoke 5-10 cases CFD. ETA 2-3 jours.
- **M_H+** : Department spawné, extension dataset (DEM PT+sud ES + multi-season ERA5 + GPU multiprocessing). ETA 7-10 jours.

## VERDICT FINAL Phase G — M_G9 GO Phase H + Conditional Phase I (2026-05-26)

Production run M_G7 sur Aqua H100 (jobs 21901136 killed walltime → 21950018 + parquet checkpointing) a livré **57,010 pairings exploitables** (95 stations NOAA FR/ES dans bbox tiles DEM, fenêtre winter 2022-23 4 mois). C'est **8350× plus** que le M17 baseline (N=7 ICOS).

**Métriques globales** :
- MAE surrogate v2 = 1.476 m/s
- MAE ERA5 baseline raw = 1.361 m/s
- ΔMAE = -0.115 m/s (surrogate REGRESS -8.4%)
- Affine fit `speed_pred = 0.593·speed_obs + 1.475` (R²=0.606)

**M16/M17 verdict CONFIRMÉ statistiquement** : pattern affine (slope ~0.55, intercept ~+1.5) reproduit à grande échelle. Compression dynamique typique RANS k-ε + wall function.

**Décisions M_G9** (trinité) :

1. **Surrogate v2 raw = NO-GO déploiement**. MAE 1.476 m/s + REGRESS vs ERA5 brut → ne pas exposer aux applications fire/wind/paragliding sans correction.

2. **Phase H DNN bias correction = GO**. Dataset suffisant (57k pairings × 6 strata), pattern affine reproductible exploitable. Cible : DNN correction `speed_corr = f(speed_pred, terrain_class, height_bucket, wind_class, season, era5_freshness)` apprenant à fermer le 8% gap.

3. **Phase I alpine = CONDITIONAL**. summit (N=51) MAE 2.754, bias -2.259, R² 0.08 → RANS 6km insuffisant. Re-simuler ~50 cas alpine domaine 10×10×5 km **si Phase H DNN ne suffit pas pour cette classe**.

**Caveats prod run** :
- 2 jobs walltime kill 6h (CPU-bound `materialise_grid_zarr`, GPU 0-5% utilization). Optimisations future : multiprocessing.Pool sur les prep grid.zarr.
- Coverage limitée à tiles DEM dispo (FR + nord ES, ~26% de la population NOAA). PT zéro stations.
- Single window winter2223 (4 mois). Saisons MAM/JJA/SON à valider via Phase H.

**Code livré** :
- 10 commits Phase G (c577ecf → 01ae5fb)
- Pipeline complet : OBS unifié + surrogate input extract + inference batched + audit stratifié
- Memory engineer.md/department.md enrichies (12 patterns Zarr/PBS/audit)

## Phase G — M_G7 done, inférence surrogate aux stations validée (2026-05-23)

`services/module2b-surrogate/infer_at_stations.py` (514 LOC) +
`utils/inference_batch.py` (121 LOC) +
`configs/hpc/infer_at_stations.pbs` (49 LOC) livrent le pipeline batched
d'inférence aux pairings OBS.

**Smoke confirmé sur Perdigão rne01 mai 2017 IOP** : 10/10 pairings,
`speed_pred` 1.93-2.29 m/s vs `speed_obs` 1.11-2.27 m/s → magnitudes
physiques cohérentes, **légère surestimation typique V0** confirmant le
verdict de la session précédente (biais affine `U_cfd = 0.54·U_obs +
1.88` sur ICOS, M16). À ce stade le surrogate v2 reproduit fidèlement
ce qu'aurait donné un OpenFOAM v2 sur ces coords.

**best.pt local** : `/Users/guillaume/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`
(116 MB downloadé Aqua→local).

**Caveats à propager à M_G8** :
- ERA5 d2m requis (legacy store sans d2m crash) → utiliser
  `era5_europe_spring2017_v2.zarr` pour smoke. Production large = re-
  ingest ERA5 hourly avec d2m.
- ERA5 Δt=6h → predictions constantes par blocs 6h ; le parquet
  contient `era5_time_delta_minutes` pour tracer.
- M_G8 doit stratifier audit par `abs(era5_time_delta_minutes)` et
  reporter ERA5-on-time vs ERA5-interpolated séparément.

How to apply : M_G8 = audit OBS vs surrogate stratifié. M_G7 production
peut être lancé via PBS Aqua H100 dès que le full NOAA ISD zarr est
disponible. Penser à régénérer ERA5 hourly Europe avec d2m comme
prérequis production-scale.

## Phase G — M_G6 done, extract input surrogate à coords arbitraires (2026-05-22)

`services/module2b-surrogate/extract_v2_input_at_coords.py` (338 LOC) +
`utils/inference_input.py` (341 LOC) livrent le pipeline pure
`(lat, lon, timestamp) → grid.zarr/input` au format consommé par
`WindV2DatasetViT`. Smoke 3/3 OK sur Perdigão, shapes validées
identiques au reference Aqua `ct_d_fire_0056_case_ts014`.

**Caveats à archiver** :
- z0_eff calculé en geometric mean omnidirectionnel sur patch 3 km
  radius (≠ upstream-only de prepare_inflow original). Choix justifié
  pour inférence sans direction connue, mais à valider en M_G8 vs OBS.
- ERA5 store `era5_europe_spring2017_v2.zarr` est Δt=6h ; pour pairings
  hourly stricts M_G7 doit interpoler en amont OU utiliser un store
  hourly (e.g. `era5_europe.zarr` si Δt=1h, à vérifier).
- Pas de normalisation appliquée à l'écriture du grid.zarr ; runtime
  `WindV2DatasetViT` applique `DEFAULT_NORM ∪ overrides
  (dataset_v2_norm.yaml)` à la lecture.

How to apply : M_G7 inférence doit (1) vérifier la cadence ERA5 du
store utilisé, (2) batched-load les grid.zarr produits par M_G6,
(3) appliquer `WindV2DatasetViT` puis le best.pt, (4) extraire U/V/W
au voxel central (90, 90, k(z_obs=elev+10m)) en interpolant via
`coords/z[180,180,40]`.

## Phase G ouverte 2026-05-21 — extension dataset OBS + inférence surrogate aux stations

**Stratégie validée par user** : pas de re-simulation OF. Inférence
surrogate v2 (best identifié = `~/dsw/data/models/surrogate_v2_vit_base_resid_s4_geo_agl100_k24_surface/best.pt`,
val_mse=0.121, 24 AGL FWI 0-100 m, ERA5 surface in input) aux coordonnées
des stations OBS comme oracle CFD.

**Sources OBS ciblées** : Perdigão IOP 2017 (déjà ingéré) + SYNOP MF
~62 + AEMET ES ~250 + IPMA PT ~120 (via OGIMET archive) + ICOS 7
existants = **~480 stations** (sub-1000 du brief mais >10× seuil
LOSO honest).

**Caveats à archiver** :
- IPMA n'a pas d'archive open historique 2018-2023 → OGIMET (synops
  décodés gratuits) est le fallback obligatoire pour PT.
- AEMET API rate-limit 60 req/min → ingestion en 5 batchs régionaux
  séquentiels (Norte/Centro/Sur/Baleares/Canarias).
- ViT-Large (val_loss 0.4966) tenu en réserve si val_mse plus bas du
  base_agl100_k24 ne se généralise pas en M_G8.

**Pairing strategy** : pas d'index dans cases v2, M_G6 extrait input
surrogate à la volée depuis SRTM+WC+ERA5_europe.zarr. Stratification
4×3×4 = 48 cellules × 30 timestamps × 480 stations ≈ 690k pairings.

How to apply : avant tout audit final, vérifier que la stratification
n'a pas un trou critique sur une classe sous-représentée (alpine
summits, coastal sites). Penser à inclure les sites ICOS Méditerranéen
(FR-Pue, ES-LJu) dans la validation focused car ils sont les pivots
des résultats FWI antérieurs.

## VERDICT FINAL session 2026-05-18/20 : V0 statu quo, ne pas regen 9k (M17)

Après 6 retournements (Phase B → C → D → E → M14 → step-back → M16 → M17),
convergence vers **V0 = dataset v2 actuel = état de l'art**.

**Conclusions empilées** :
- Le "30% déficit" initial (`crop/inflow = 0.696`) est un **artefact
  d'audit** : ne vaut que pour n=20 cas vent fort ≥5 m/s (4% du
  sample500). Médiane sur tout le sample = 0.88. 41% des cas ont
  CFD ≥ ERA5.
- **V0 actuel bat ERA5** sur tall-tower 2020 : MAE médian −18% sur
  15 pairings. **CFD < OBS sur 8/15** (= ERA5 idem 10/15) ; **CFD > OBS
  sur 4/15**. Pas systématique.
- Tous les patches BC (V1, V8, V9, V10) sont des **degrés de liberté
  empiriques** qui flippent décision selon le test set (2D → 3D → réel).
- pg_geo "flip" vs "native" : la convention native est physiquement
  correcte (`source.x = -f·V_g`) mais la calibration sur geopotential
  850-700 hPa = mauvaise altitude (rotation Ekman 30-60° entre 750 hPa
  et surface). Sign optimal site-dépendant → patch fragile, à éviter.
- Sur OBS : biais affine `U_cfd = 0.54·U_obs + 1.88` (R²=0.43) hors
  alpine summits. **Mais ERA5 a la même compression** (a=0.47). C'est
  physique RANS k-ε + mesh grossier + wall functions, pas BC tuning.
- M17 POC XGBoost sur 7 sites : **ne bat pas CFD raw** en LOSO. Avec
  7 sites le ML apprend climatologie (top features = lat/lon/elev,
  CFD importance 0.04). Affine fix est pire que CFD raw partout.
- Dataset OBS actuel insuffisant pour ML correction utile.

**Stack final** = V0 = `inletOutlet top + zeroGradient + pg OFF +
z0=0.05 + wc native + Coriolis ON + Parente ambient + atmNutk +
simpleFoam k-ε 300 iter` = essentiellement **Venkatraman Perdigão WES
2023 adapté aux 9k sites v2**.

**Roadmap post-session** (voir REPORT.md §9) :
- **Phase G** (2-3 sem) : extension dataset OBS à ~1000+ stations
  (Perdigão IOP 2017 + SYNOP FR/ES/PT + ICOS multi)
- **Phase H** (1 sem) : DNN bias correction stratifié (terrain × height
  × wind class × season), inspiré du module 3 precip stratified QM
- **Phase I** (~10× compute, optionnel) : domaine 10×10×5 km pour
  ~50 cas alpine summit

**Leçons orchestration** :
1. 1 cas analytique ≠ vérité universelle (Phase B ridge 2D → Phase C
   multi-hill 3D inversait la décision)
2. Sign d'un patch empirique est suspect (pg_geo flip vs native)
3. Audits CSV proxies fragiles (crop/inflow médian sur 500 cas non
   stratifiés cachait que 96% du sample est vent faible où CFD ≥ ERA5)
4. **Bench OBS direct AVANT d'investir dans stack BC** — on a passé
   3 jours sur BC tuning avant de mesurer que V0 actuel bat ERA5
5. ML correction sur N=7 sites = climatologie pas physique. Seuil
   empirique : ~30-50 sites min pour LOSO honest
6. **Adversarial dual-spawn (A+B convergents) = solide** — les deux
   Departments step-back ont indépendamment conclu V0 statu quo

**Le commit `43f5e90` ("V1 retained for 9k regen") est mis en doute
par cette session**. À traiter via commit fix qui pointe vers le
verdict V0 final.

## Phase E retourne V10 → V1 retenu pour regen 9k (2026-05-18, M13)

Phase E sur 5 sites v2 réels Pop A FR continental (ct_c_morpho_0000,
ct_d_fire_0017, ct_e_mountain_0023, ct_f_wind_onshore_0001,
ct_g_paragliding_0006) : **V1 (best-stack flip) bat V10 (top open
native) sur 4/4 sites complets** en ratio physique
`crop_mean / ERA5_U10_nominal`.

Mean ratio across sites : V0=1.35, V10=1.89, **V1=2.31**.

Le proxy `edge_W` (vent au bord W amont CFD) n'est PAS un bon proxy
d'inflow car le forçage pg_geo change la dynamique d'inflow (V1
edge_W=5.21 vs V10 edge_W=1.14 sur ct_d_fire_0017). La métrique
physique correcte est `crop_mean / ERA5_U10[1,1]` (extrait de
`input/era5_surface/u10` du grid.zarr).

**Pour la regen 9k : V1 final**. Stack :
- top U : slip
- top p : fixedValue 0
- pg_geo : flip
- z0_wall : 0.005 m
- z0 field : wc_capped_0.05
- Coriolis : on

**Leçon orchestration** : le multi-hill analytique a induit en
erreur (recommandait V10) parce que la topographie analytique
manque les structures de recirculation du vrai terrain. Toujours
valider sur sites v2 réels (au moins 5, diversifiés Pop A) avant
adoption finale d'un stack.

**Bug PBS phaseE_5sites.pbs** : oubliait `writeCellCentres` (donc
`0/Cx` manquait pour l'export). Fix : `phaseE_export_only.pbs`
séquentiel qui fait writeCellCentres + export sur cases déjà solved.
Pattern à intégrer si on étend le builder pour mode `site_real`.

## V10 native > V8 flip — décision révisée (2026-05-18, M12)

Test M12 (V10 = V9 + pg native) sur multi-hill : V10 bat TOUTES les
métriques. crop_mean=0.808 (Δ vs V9 flip = +0.176, Δ vs V8 = +0.142),
crest_max=1.961 (le plus haut), flat_mean=0.724.

**Convention pg confirmée** :
- native : `source.x += +7.587e-04 × V`, `source.y += +5.192e-04 × V`
- flip   : opposé (multiplication par -1)
- Sur 0056 (Sierra Andaluza, 37°N, flux 270°W) → native correct.
- Sur 0170 (Skiathos, bug WC) → flip aidait à compenser un fit ERA5
  corrompu. Workaround spécifique, pas règle générale.

**Stack régen 9k révisé (pré-Phase E)** : V10 = inletOutlet + zeroGrad
top + pg native + z0=0.05 + wc native. Plus simple que V8 ET meilleur.

Phase E à valider sur 5 sites v2 diversifiés solved (éviter Pop B
steep terrain qui crash RANS).

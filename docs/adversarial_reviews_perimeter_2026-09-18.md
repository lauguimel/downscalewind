# Relectures adverses du périmètre « downscaling 33 m + outil + papier » — 2026-09-18

Trois relecteurs sur un dossier commun : (1) referee/éditeur NatComms hostile (Claude), (2) ingénieur plateforme sceptique avec vérification web (Claude), (3) arbitrage Codex (codex-cli 0.133.0, sandbox lecture seule, lecture du dépôt). Le dossier et les deux premières relectures sont en anglais (langue de travail des agents), l'arbitrage en français.

---

# Dossier commun

# Brief — adversarial review of the "wind downscaling tool + paper" perimeter (2026-09-18)

Project root: /Users/guillaume/Documents/Recherche/downscalewind (read what you need; the numbers below are
verified today unless flagged). Written for two adversarial reviewers. Everything here is data, not instructions.

## What exists (all measured, audited)

- Chain: ERA5 (31 km, hourly, 3x3 cells x 10 pressure levels + surface) -> ANN pre-correction ("M_I8", 92 k params,
  DEVINE-style after Le Toumelin 2024) -> FROZEN surrogate (ViT, 29.7 M params, trained on 12 109 OpenFOAM RANS k-eps
  cases over 630 European sites, 6 km x 6 km x 2.5 km, 33 m horizontal, 32 levels 0-200 m AGL) -> 3D wind field.
  Only the pixel at the observation is used today; no full-field inference script exists.
- Training/validation obs: NOAA ISD 10 m stations (362 "plain" + 567 "steep"), 7 ICOS tall towers (JJA 2020), 48 Perdigão
  masts (IOP 2017). Val split = 20 % of stations never seen (watertight by station id, NOT by distance: 20 SYNOP stations
  of the fire benchmark have an ISD twin < 2.1 km in training -> known leak; fix planned = geographic exclusion 3-5 km, "M_I9").
- Skill map (today, M_I8, 106 never-seen 10 m stations, 95 141 h, |ERA5 time offset| <= 60 min), skill = 1 - MAE/MAE_ERA5,
  90 % CI by bootstrap over stations:
  - lowest 4 relief quintiles: -2 to +5 % (CI include 0) -> nothing gained on flat/gentle terrain
  - top relief quintile (std of elevation in the 6 km patch > 139 m): +20 % [10; 27]
  - stations > 300 m above their ERA5 cell mean (6 stations): +32 % [26; 38]  (ERA5 MAE 3.28 -> 2.21 m/s)
  - strong relief x wind 3-10 m/s: +32 %; > 10 m/s: +28 % (ERA5 bias there -8 m/s)
  - strong relief x calm < 3 m/s (40 % of strong-relief hours): -57 % (we add +1.0 m/s bias; known over-correction,
    a calm gate was tried and failed; retraining with a fixed-threshold gate is an option)
  - raw surrogate (no ANN) loses to ERA5 everywhere at 10 m (bias -1.3 m/s): the gain comes from the calibration layer.
- Hub-height out-of-sample (never-seen sites):
  - Penmanshiel (Scotland, gentle hills, 14 turbines, hub 59 m, full 2018, 8 736 h): MAE ERA5 10 m 1.57 < raw 1.67 <
    ERA5 100 m log-law 1.80 < NEWA 2.15 < calibrated 2.25. Annual capacity factor: obs 0.308, raw 0.324, ERA5 10 m 0.304,
    ERA5 100 m 0.426, NEWA 0.446, calibrated 0.455.
  - KIT tower (Rhine plain, JJA 2020): 60 m calibrated 0.97 ~ ERA5 0.99; 100 m ERA5 native 100 m 1.08 < calibrated 1.29
    < NEWA 1.55; 200 m calibrated 1.50 < NEWA 1.68 (ERA5 has nothing at 200 m).
  - Three more ICOS towers (OXK 1 020 m Fichtelgebirge, wind only at 163 m; TOH; KRE) running this week.
- Benchmark vs FuXi-CFD (Lin et al. 2026, NatComms, CC BY-NC weights): on the common subset (n = 500) ours 1.002 vs
  FuXi 1.292 vs ERA5 1.207 MAE; we beat FuXi at all 4 of their EU tower x height pairs. FuXi-CFD is direct prior art
  (ERA5 -> CFD-surrogate -> obs).
- Prior art for the mechanism: Le Toumelin et al. 2024 (DEVINE): obs-based correction upstream of a frozen downscaler.
  Our novelty claim can only be the object (3D RANS 33 m, multi-variable ERA5 forcing, 630 EU sites, 0-200 m profile)
  and the application/tool, not the mechanism.
- Fire-weather side (separate paper plan): ISI/FFMC resolved at 33 m; existing ICOS-based validation is BROKEN
  (midnight forcing, heights never matched); campaign v2 dataset is sound. Not the subject here, but shares the chain.
- Existing plans and an earlier external critique: docs/plan_paper_wind_energy_2026-09-16.md (v1.1),
  docs/plan_paper_fire_weather_2026-09-16.md, docs/codex_review_paper_plans_2026-09-16.md. That critique already
  killed "hourly European atlas" (storage 53 GB per variable for 100 km^2-year; 10-20 k tiles x 8 760 h).

## The perimeter under review ("first perimeter", Europe first)

1. Data extension to retrain the calibration on strong relief AND height (currently 6 val stations > 300 m above cell,
   one tower > 100 m in strong relief). Verified-open candidates: Hill of Towie SCADA (21 nacelles ~82 m, hilly Scotland,
   2016-2024, CC BY 4.0, Zenodo 14870023); GeoSphere Austria (~260 10 m stations, CC BY 4.0, API); MeteoSwiss OGD
   (~160 stations, attribution); Météo-France (~2 000 stations, Etalab); SLF IMIS (~200 ridge stations 2 000-3 000 m,
   SLF terms, sensor height unverified). Second wave (US, out-of-continent test only): NEON tower profiles (~10 mountain
   towers, 4-6 levels, CC BY 4.0), WFIP2 (19 sodars, DOE account). Alaiz 118 m ridge mast exists (CC BY) but sits behind a
   DTU database access the PI refuses to request.
   Constraints: ERA5 via CDS = 7-day requests, ~1 block/h (ARCO-ERA5 on Google Cloud unverified as replacement);
   cache 120 KB per station-hour (1 000 stations x 1 year = 1 TB -> subsample); retrain = ~35 h H100 + queue;
   geographic split (no val station within 5 km of a training one) mandatory.
2. Retrain M_I9 with (a) geographic split, (b) the new data, (c) a fix for the calm over-correction.
3. Tool: the PI wants a public, non-commercial service where anyone gets a 33 m wind estimate at a point — "source
   agnostic" (feed it ERA5 OR a forecast; it adapts the fields to terrain). Idea on the table: host the model on Google
   Vertex AI (Earth Engine can call Vertex endpoints via ee.Model.fromVertexAi) so that USERS pay their own inference,
   and the listing gives visibility. Open questions: who actually pays on Vertex (endpoint owner vs caller), whether
   a Vertex/Earth Engine listing gives any visibility vs Hugging Face / Zenodo / a plain web app, whether ERA5 pressure
   levels are reachable from Earth Engine at all, and the domain shift when feeding forecasts (IFS/GFS) to a model
   trained on ERA5 only.
4. Paper target: Nature Communications or similar high-visibility venue, framed as method + validated tool.
   Fallbacks: Wind Energy Science, GMD, Environmental Modelling & Software.

## Resources
One researcher (not a professional developer) + agentic coding; HPC (A100/H100, PBS queue days-long for 40 h jobs);
no new field campaigns; no budget line for cloud (small personal spend possible); timeline: wants to submit within months.

---

# Two adversarial reviews already produced today (Claude subagents) — to be contested or confirmed

## Review 1 — hostile NatComms referee/editor

Verdict: NatComms as is = desk reject probable; after the planned data extension = reject after review or major
revision at best. Realistic fallback: Geoscientific Model Development after M_I9 + ablations. Wind Energy Science only
if the hub-height result changes sign.

Five attacks:
A. Gain exists only in the top relief quintile and is negative 40 % of the time there (calm < 3 m/s: -57 %, +1.0 m/s
   bias). Referee will ask for hour-weighted net skill per wind class. Data extension does not answer; only a calm fix does.
B. The 0-200 m profile is refuted by own out-of-sample data (Penmanshiel 59 m calibrated last of five; KIT 100 m ERA5
   native beats calibrated). Need >= 3 independent 60-130 m sites in relief where calibrated beats ERA5 100 m and NEWA,
   plus a flat control not degraded.
C. Headline +32 % rests on 6 stations and a leaky split (station-id split, 20 SYNOP with ISD twin < 2.1 km). Need
   geographic split 5 km, >= 300 never-seen strong-relief stations, >= 30 stations > 300 m above cell, >= 3 countries.
D. The CFD engine (12 109 RANS, 29.7 M params) brings nothing measured at 10 m: raw surrogate loses to ERA5 everywhere;
   all gain comes from a 92 k-param ANN. Missing ablation: same ANN on ERA5 + terrain descriptors WITHOUT surrogate
   (Wind-Topo / TerraWind style). Without it the surrogate's contribution is a hypothesis.
E. FuXi-CFD comparison not like-for-like (calibrated vs uncalibrated). Expected answer: FuXi-CFD + same ANN layer.
Secondary: neutral RANS; 34 CFD cases > 8 m/s while the claimed gain is in strong wind; no full-field inference exists;
"source agnostic" never tested.

Novelty: defensible = multi-level reanalysis -> 3D RANS surrogate 33 m -> obs calibration at 630-site scale (an
assembly, not a mechanism). Not new: upstream calibration (DEVINE), terrain-informed CFD surrogate (FuXi-CFD, neural
operators 2026, Kim et al.), direct station learning in complex terrain (Wind-Topo 2022, TerraWind 2024), public weights.
The "tool" angle is a prerequisite competitors already meet, not a novelty; Vertex/Earth Engine hosting adds nothing
scientifically and invites "neither reproducible nor free".

Minimum evidence to flip to plausible: M_I9 geographic split; >= 300 strong-relief never-seen stations, 3 countries,
>= 30 > 300 m above cell, positive hour-weighted skill in EVERY wind class incl. calm; >= 3 hub-height relief sites
beating ERA5 100 m and NEWA; ablations (a) ANN-only no surrogate, (b) surrogate-only, (c) calm gate on/off,
(d) FuXi-CFD + same ANN; baselines everywhere; full-field inference + weights + CLI + DOI. 4-6 months.
If (a) matches the full chain, the NatComms paper dies and becomes a GMD paper on the CFD database.

Compromise ranked: 1 cut Vertex/EE/source-agnostic/forecast, publish weights+CLI+DOI (+HF Space); 2 add obs-only
ablation; 3 keep 10 m strong relief as central claim with geographic split and >= 300 stations; 4 add calm fix;
5 hub height only if Hill of Towie + OXK flip the sign, else limitation and drop "0-200 m" from title; 6 FuXi-CFD
calibrated-vs-calibrated or labelled unfair; 7 cut fire weather and NEON/WFIP2.

## Review 2 — sceptical ML-platform engineer (facts verified with URLs)

Code read: run_hub_height_inference.py builds a grid.zarr per (station, hour), runs the ViT twice, discards everything
but pixel (90,90) at level k_obs. The full field already exists in memory; "full-field script" = denormalise + write.

Vertex AI: NO — hosting there does not make users pay nor give visibility. Online prediction endpoints are billed per
node-hour while deployed, no scale-to-zero (e2-standard-4 ~ 112 $/month idle; A100 ~ 2 600 $/month idle). Earth Engine
callers need the "Vertex AI User" role on the project hosting the model -> the endpoint lives in the PI's project and
the PI's card is charged. "Users pay" only exists if they redeploy the weights in their own project (i.e. publish the
weights). No public listing of third-party Vertex endpoints. Hugging Face shows downloads/likes; free CPU Space 2 vCPU /
16 GB, sleeps after 48 h idle; ZeroGPU 3.5 min/day free. Useful fact: ERA5 pressure levels ARE in Earth Engine
(ECMWF/ERA5/HOURLY_PRESSURE_LEVELS, 37 levels, u/v/T/q, hourly).

Architecture proposed: weights on HF Hub + Zenodo DOI; one Python function predict_point(lat, lon, t, source) reusing
build_one; DEM/WorldCover tiled on the fly; ERA5 from ARCO-ERA5 (gs://gcp-public-data-arco-era5, hourly, 37 levels,
u/v/T/q + surface, anonymous access; one isolated point costs ~1 GB read per hour -> regional cache needed); forecast
source = ECMWF open data IFS 0.25°, CC BY 4.0, 14 levels covering our 10, 3-hourly, rolling 2-3 days; demo = Gradio
Space on free CPU; 5-15 s per point on 2 vCPU (computed, not measured); Cloud Run GPU L4 0.67 $/h scale-to-zero if
needed. Monthly cost 0-9 $. Breaks first: ERA5 read latency + 3-month ERA5 lag, then the sleeping Space.

Data plan: Hill of Towie = nacelle anemometer (wake, manufacturer correction) -> validation only, not training.
GeoSphere: best relief/effort, but tawes-v1-10min covers only last 3 months; need the historical climate dataset (name
unverified). IMIS: sensors 6.8 m above bare ground on exposed ridges, height varies with snowpack, winter icing, sites
out of the RANS distribution (pop B steep excluded from dataset v2) -> extreme validation only. Météo-France 2 000
stations x 120 KB/station-hour = cache trap; subsample to relief quintiles 4-5. ERA5: do not use CDS; extract Europe x
10 levels x 8 vars from ARCO on a VM next to the bucket into a regional zarr of a few GB. Leak: filter new networks
against ISD/SYNOP at 5 km (national networks share sites with ISD). Fastest to M_I9: (1) geographic split on existing
data -> M_I9a in one week; (2) GeoSphere; (3) Hill of Towie validation; (4) rest. NEON/WFIP2 after the paper.

Effort (supervised agentic-coding days): full-field 1-2; tool MVP 6-10; data wave 1 8-12; M_I9 3-5 + 35 h H100 + queue.

Riskiest assumption: that "source agnostic" and "useful everywhere" hold. All gain comes from the ANN calibrated on
ERA5; a correction does not transfer across backbones; gain is nil on the least rugged half of Europe; at Penmanshiel
calibrated (2.25) is worse than ERA5 10 m (1.57). A public tool shipped as is returns, for most points and any IFS input,
a number worse than ERA5 without saying so. A guard (return ERA5 outside the competence domain) is mandatory.

---

# Review 3 — arbitrage Codex

**1. Arbitrage**

- A confirmé : le gain est concentré en relief fort, et le calme casse la chaîne : `skill_map_M_I8` donne +19,9 % en Q5 relief, mais -56,6 % en relief haut et vent observé 0-3 m/s.
- B confirmé pour le claim hauteur de moyeu : Penmanshiel 59 m donne `ours_corr` MAE 2,25 contre ERA5 10 m 1,57, ERA5 100 m log-law 1,80, NEWA 2,15 (`hub_height_benchmark/.../REPORT.md`).
- C confirmé, mais seuil à calibrer : +32,4 % repose sur 6 stations `dz_cell >300 m`; le split est par `station_id` (`watertight_station_split`) et non géographique. Pour NatComms, c’est insuffisant.
- D confirmé à 10 m : le brut perd partout contre ERA5 dans `REPORT_skill_map.md`; sans ablation ANN-seule, les 12 109 RANS et 29,7 M paramètres ne portent pas la preuve.
- E affaibli : la comparaison FuXi est exploitable comme benchmark commun, mais pas comme claim principal, car c’est “nous calibré” contre FuXi non calibré; en plus, le rapport FuXi global inclut Perdigão où FuXi bat nous : MAE 1,19 contre 2,09.
- Hypothèse la plus risquée confirmée : “source agnostic / utile partout” est fausse aujourd’hui. Le gain est nul sur relief faible, négatif au calme, et Penmanshiel montre un outil public qui peut dégrader ERA5.

**2. Ce que les deux revues ont manqué**

- Le papier ne valide pas un champ 3D, seulement des pixels centraux. `run_hub_height_inference.py` appelle le surrogate sur le patch puis `_denorm_uv_at_center`; aucune métrique ne teste le champ complet, les gradients, ni la direction.
- Le relief utilisé dans `skill_map.py` peut être biaisé près des côtes ou tuiles manquantes : `_read_box(..., nodata=0.0)` met la mer/le manque à 0 m, ce qui peut gonfler `relief_std_6km`. À corriger avant toute carte de domaine de compétence.
- Le vent directionnel est absent du périmètre de preuve. Les rapports scorent surtout la vitesse et le facteur de charge; un front-end “on donne des champs, il les adapte au relief” doit au minimum rapporter erreur de direction, biais u/v et rotation en relief.
- Les intervalles de confiance actuels bootstrapent par station, pas par réseau/pays. Avec Météo-France, GeoSphere, MeteoSwiss, le risque principal devient le biais réseau/instrument, pas seulement le nombre d’heures.
- Non vérifié ici : la littérature WAsP/NEWA/WindNinja attend une preuve de domaine d’application et d’abstention; un modèle qui ne répond que sur crêtes ventées peut être publiable, mais pas vendu comme downscaler général.

**3. Vertex AI**

Non : Vertex AI ne donne pas un modèle “les utilisateurs paient leur inférence” pour un endpoint du PI; Earth Engine exige l’accès IAM au projet qui héberge l’endpoint et signale Vertex AI/Cloud Storage/Earth Engine comme composants facturables. Google documente `ee.Model.fromVertexAi` comme un connecteur vers un endpoint déployé, pas comme une place de marché publique. Les options à durée limitée type Flex-start existent pour réduire le coût de jobs courts, mais pas comme visibilité ni modèle économique public stable. Sources : Earth Engine Vertex AI docs, IAM/coûts `ee.Model.fromVertexAi` ; Google Flex-start billing.  

**4. Compromis périmètre**

1. Figer le protocole M_I9 : métriques, split géographique 5 km, classes relief/vent, exclusion des jumelles ISD/SYNOP. Objectif : publication. Effort : 3 j. Stop : si les données nécessaires ne peuvent pas être redistribuées ou auditées.
2. Rejouer M_I8 avec audit de fuite et split géographique sur l’existant. Objectif : publication. Effort : 6 j. Stop : si le gain Q5 relief tombe sous +10 % ou si l’IC 90 % inclut 0.
3. Ingérer seulement GeoSphere + MeteoSwiss + sous-échantillon Météo-France relief Q4-Q5. Objectif : publication. Effort : 15 j. Stop : si après exclusion 5 km il reste <150 stations relief fort ou <20 stations `dz_cell >300 m` sur au moins 3 pays.
4. Réentraîner M_I9 avec correction du calme et garde d’abstention. Objectif : les deux. Effort : 8 j + 35 h H100 + queue. Stop : si relief haut 0-3 m/s reste négatif ou si relief faible est dégradé de >5 % contre ERA5.
5. Faire les ablations obligatoires : ANN-seule sans surrogate, brut, ANN+surrogate, calme on/off, FuXi clairement étiqueté. Objectif : publication. Effort : 6 j. Stop : si ANN-seule égale la chaîne complète à ±2 points de skill.
6. Hub-height seulement en validation, pas en titre avant résultat : Penmanshiel, Hill of Towie validation, OXK/TOH/KRE si disponibles. Objectif : publication. Effort : 12 j. Stop : si moins de 2 sites relief indépendants battent ERA5 100 m ou NEWA.
7. Outil minimal : `predict_point(lat, lon, time, source="ERA5")`, poids Zenodo/Hugging Face, CLI, garde “return ERA5 / no claim” hors domaine. Objectif : outil. Effort : 8 j. Stop : si latence CPU >30 s/point ou si la garde couvre <10 % des requêtes relief utiles.
8. Rédaction + dépôt reproductible des figures. Objectif : publication. Effort : 20 j. Stop : si les figures centrales ne sont pas régénérables hors Aqua avec chemins documentés.

Venue réaliste : GMD ou Environmental Modelling & Software; Wind Energy Science seulement si le bloc hub-height passe. NatComms n’est réaliste que si les items 2, 4, 5 et 6 passent tous nettement. Titre défendable : “A guarded observation-calibrated CFD surrogate for point wind downscaling in complex European terrain.”

**5. Ce que le PI ne peut pas avoir en 6 mois**

Il ne peut pas avoir en même temps NatComms, outil source-agnostic prévision/réanalyse, service public Vertex, validation hauteur de moyeu solide, atlas Europe et correction du calme. Chacun est un chantier à risque; ensemble, ils diluent la preuve.

Il ne peut pas revendiquer “utile partout” : les chiffres actuels disent “utile surtout en relief fort et vent établi”.

Il ne peut pas remplacer Alaiz sans conséquence : refuser le mât de crête réduit fortement la crédibilité hauteur de moyeu en terrain complexe.
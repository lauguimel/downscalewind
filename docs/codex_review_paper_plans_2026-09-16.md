# Relectures critiques Codex des deux plans de papier — 2026-09-16

Brief commun : rôle relecteur Nature + ingénieur solo ; lecture seule du dépôt ; sortie en français.
Modèle : codex-cli 0.133.0, sandbox read-only. Verdicts intégrés dans les plans v1.1 le même jour.

## Plan 1 — vent à hauteur de moyeu

**FATAL FLAWS**

1. Surpromesse “premier atlas horaire calibré Europe”. Prior art lourd : GWA v3, NEWA, atlite, renewables.ninja, Staffell & Pfenninger 2016, validation européenne ERA5/NEWA/GWA. Le plan ne peut pas vendre “Europe 10-200 m” avec une seule année 2022-2023, domaine incomplet et validation limitée. Fix : revendiquer “méthode CFD-surrogate calibrée observation pour terrains complexes, démontrée sur massifs ciblés”.

2. Inférence régionale inexistante. Le plan le reconnaît, mais c’est un blocage d’ingénierie : `infer_at_stations.py` matérialise un `grid.zarr` par couple station-temps puis extrait seulement la colonne centrale ([infer_at_stations.py](/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/infer_at_stations.py:9), [ligne 393](/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/infer_at_stations.py:393)). `build_one` ne fait qu’un patch 6 km ([extract_v2_input_at_coords.py](/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/extract_v2_input_at_coords.py:59)). Fix : P0 doit être “moteur régional minimal + test I/O”, pas analyse de bord d’abord.

3. Budget stockage faux. Le plan annonce 8 Go pour 100×100 km, 8760 h, 3 hauteurs, 100 m ([plan](/Users/guillaume/Documents/Recherche/downscalewind/docs/plan_paper_wind_energy_2026-09-16.md:69)). Calcul : 1000×1000 pixels × 8760 × 3 × float16 = 52.6 Go pour une seule variable ; vitesse+direction = 105 Go, hors chunks/métadonnées. À l’échelle 10-20k tuiles, l’horaire complet devient PB si on garde les champs. Fix : ne stocker que climatologies et séries turbine/validation ; pas de zarr horaire dense sauf région pilote.

4. Validation hors échantillon pas encore acquise. Le plan dit que les tours ICOS et Perdigão sont dans M_I8 ([plan](/Users/guillaume/Documents/Recherche/downscalewind/docs/plan_paper_wind_energy_2026-09-16.md:46)); le YAML confirme `exclude_substrings: []` ([yaml](/Users/guillaume/Documents/Recherche/downscalewind/configs/training/devine_style_M_I8_multiheight_v3.yaml:112)). Fix : publier seulement après Alaiz/Penmanshiel/Rödeser Berg réellement ingérés et pré-enregistrés ; sinon papier méthode, pas atlas.

5. “Terrain complexe” non aligné avec les outils existants. Le plan veut écart-type d’altitude 6 km ([plan](/Users/guillaume/Documents/Recherche/downscalewind/docs/plan_paper_wind_energy_2026-09-16.md:91)), mais `strata.py::class_topo` classe par altitude seule ([strata.py](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/utils/strata.py:28)). Fix : implémenter rugosité/topographic position/slope cohérents avec le masque d’entraînement, sinon les strates ne supportent pas la thèse.

**UNVERIFIED ASSUMPTIONS**

- Débit ≥50 tuile-heures/s : vérifiable en <1 h, mais aujourd’hui le PBS station cible 60% GPU sans preuve ([infer_at_stations_v2.pbs](/Users/guillaume/Documents/Recherche/downscalewind/configs/hpc/infer_at_stations_v2.pbs:8)).
- Existence locale des checkpoints M_I8/v3 : vérifiable en <1 h sur Aqua ; non présent dans ce checkout.
- ERA5 hors domaine actuel : vérifiable en 1 j ; les stores existants ne couvrent pas Écosse/Scandinavie/Carpates ([plan](/Users/guillaume/Documents/Recherche/downscalewind/docs/plan_paper_wind_energy_2026-09-16.md:140)).
- Alaiz/Rödeser Berg données exploitables et licences : vérifiable en 1 j. Alaiz semble public via CENER/CKAN ; Rödeser Berg/Fraunhofer moins clair.
- SCADA Penmanshiel : vérifiable en <1 h, Zenodo CC-BY-4.0. Kelmarsh/La Haute Borne : vérifiable en 1 j.
- OSM pour inventaire turbines : risque de fond, ODbL/share-alike peut contaminer une base dérivée publiée.
- Nature/Joule acceptent pipeline CFD privé : risque de fond. Nature demande code/données nécessaires à vérifier les conclusions ; restrictions doivent être déclarées.

**PRIOR ART SOUS-PONDÉRÉ**

- “Validation of European-scale simulated wind speed and wind generation time series”, Applied Energy 2022 : ERA5, NEWA, EIWR, GWA scaling, 32 mâts, production pays.
- “Validation of European wind generation time series simulation: Importance of wakes, micro-scale adjustments and stochastic simulations”, Applied Energy 2025.
- NEWA, y compris Alaiz ALEX17 et Kassel/Rödeser Berg, pas seulement comme données mais comme concurrent scientifique.
- KNW Atlas, NORA3, CERRA : utiles comme benchmarks régionaux.
- FuXi-CFD : déjà dans le dépôt via `score_fuxi_eu_towers.py`, mais absent du cadrage papier.
- WAsP/Wind Atlas Methodology : il faut expliquer pourquoi le surrogate CFD apporte plus qu’un WAsP/GWA scaling climatologique.

**SCOPE**

À couper : atlas Europe horaire dense, OSM multi-pays, TWh par pays, ENTSO-E comme validation, comparaison exhaustive GWA. Garder : Navarre-Alaiz + Penmanshiel + une région montagneuse.

MVP en 10 lignes :
1. Navarre-Aragon comme région pilote.
2. Moteur régional sur 100×100 km.
3. 2023 seulement, ou 1000 heures stratifiées.
4. Alaiz hors échantillon.
5. Penmanshiel SCADA comme production mensuelle.
6. ERA5 direct baseline atlite-like.
7. GWA/NEWA climatologie comme benchmark moyen.
8. M_I8/M_I9 clairement séparés train/test.
9. Stocker climatologies + séries points, pas champs horaires complets.
10. Papier “méthode et démonstration”, pas atlas Europe.

**ORDER**

Changer l’ordre : P0.1 moteur régional + benchmark I/O ; P0.2 validation data access/licences ; P0.3 test Alaiz/Penmanshiel minimal ; P0.4 bord de patch. La fenêtre de bord ne sert à rien si l’I/O ou les données cassent. P1 puissance seulement après validation vitesse. P3 Europe seulement après acceptation interne des figures MVP.

**EFFORT**

- P0 5 j : trop court si moteur régional + zarr + PBS + métriques. Compter 8-12 j.
- P1 10-15 j : plausible pour power curves, trop bas avec inventaire multi-pays et licences. 20 j.
- P2 15-20 j : optimiste ; ingestion SCADA/mâts + QC + alignement hauteur = 25-35 j.
- P3 10 j + calcul : faux si stockage/I/O non résolu. 20 j d’ingénierie avant calcul.
- GPU P3 500-1000 h : calcul arithmétique ok à 50/s, mais I/O et matérialisation risquent de dominer.
- Stockage 50-200 Go : faux pour champs horaires ; ok seulement pour sorties agrégées.

Sources web consultées : GWA méthode/data DTU, NEWA/Alaiz, Penmanshiel Zenodo, OPSD, OSM licence, Nature data/code policy, atlite/renewables.ninja.

Verdict :
Publiable à Nature Energy/Joule : non, pas avec ce scope.
Publiable Nature Communications : possible seulement comme méthode validée hors échantillon.
Condition 1 : réduire l’ambition Europe horaire.
Condition 2 : validation Alaiz/Penmanshiel positive et pré-enregistrée.
Condition 3 : disponibilité code/données compatible revue, sans pipeline opaque indispensable.
## Plan 2 — météo du feu

**FATAL FLAWS**

1. **La revendication centrale repose sur une chaîne non exécutée.**  
   Évidence : le seul test complet est négatif : DownscaleWind `RMSE=6.11` contre `5.31` pour ERA5 + pluie corrigée ([rapport](/Users/guillaume/Documents/Recherche/downscalewind/docs/fwi_station_uvt_inference_check_2026-05-12.md)). Le script station existant ne charge que le surrogate brut, pas l’ANN M_I8 ([run_station_surrogate_inference.py:43](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/run_station_surrogate_inference.py:43), [181](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/run_station_surrogate_inference.py:181)). Le motif M_I8 existe ailleurs seulement pour Perdigão ([run_masts_M_K2.py:122](/Users/guillaume/Documents/Recherche/downscalewind/data/validation/crest_deficit/run_masts_M_K2.py:122), [149](/Users/guillaume/Documents/Recherche/downscalewind/data/validation/crest_deficit/run_masts_M_K2.py:149)).  
   Fix : P0 doit être un vrai kill test, avant tout plan papier. Pas de phrase Nature tant que M_I8 station `val` ne bat pas ERA5+pluie corrigée.

2. **Validation hors échantillon insuffisante / fuite spatiale probable.**  
   Évidence : M_I8 a `exclude_substrings: []` ([yaml:113-118](/Users/guillaume/Documents/Recherche/downscalewind/configs/training/devine_style_M_I8_multiheight_v3.yaml:113)). Le split est étanche par `station_id`, pas par proximité géographique ni jumelle SYNOP/ISD ([dataset_v2_obs_centered.py:589](/Users/guillaume/Documents/Recherche/downscalewind/services/module2b-surrogate/src/dataset_v2_obs_centered.py:589)).  
   Fix : figer le banc d’abord, exclure par coordonnées/WMO/ISD à <3-5 km, puis réentraîner. Reporter train et val séparément, mais ne publier que val.

3. **Le banc initial est enrichi en extrêmes observés, pas représentatif.**  
   Évidence : stations sélectionnées par score dépendant de `p95_fwi_obs`, jours >21/>38, vent max ([build_fwi_station_audit.py:247](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/build_fwi_station_audit.py:247)), puis les timestamps sont les plus forts FWI observés ([312](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/build_fwi_station_audit.py:312)).  
   Fix : deux bancs séparés : saison complète figée pour skill climatologique, sous-échantillon extrême pour stress test.

4. **“FWI à 33 m” est scientifiquement fragile.**  
   Évidence : le code horaire garde BUI uniforme depuis le driver, donc seule la partie rapide varie spatialement ([shared/fwi.py:522](/Users/guillaume/Documents/Recherche/downscalewind/shared/fwi.py:522), [528](/Users/guillaume/Documents/Recherche/downscalewind/shared/fwi.py:528)). T/RH non calibrées, pluie tuile uniforme.  
   Fix : appeler cela “FWI/ISI avec vent résolu à 33 m”, faire porter les cartes surtout sur ISI/FFMC, et réserver FWI aux stations avec ablations.

5. **La démonstration spatiale P3 dépend d’un moteur absent.**  
   Évidence : le plan éolien dit explicitement que l’inférence plein domaine/tuilée n’existe pas. Dans le dépôt, pas de `infer_regional.py`; les scripts station extraient le pixel central ([run_station_surrogate_inference.py:145](/Users/guillaume/Documents/Recherche/downscalewind/services/validation/run_station_surrogate_inference.py:145)).  
   Fix : couper P3 du MVP, ou ajouter un smoke régional comme porte P0 bis.

6. **Comparaison CEMS/EFFIS et spin-up FWI pas encore verrouillées.**  
   Évidence : `build_fwi_baseline_validation.py` recalcule la fenêtre depuis le premier jour sélectionné, pas forcément Jan-May spin-up complet ([rapport: “full-history model FWI still requires Jan-May…”](/Users/guillaume/Documents/Recherche/downscalewind/data/validation/fwi_station_audit_2022_n20_surface/fwi_baseline_report.md)). CEMS ingestion est codée pour Iberia 2017, pas Méditerranée 2022-2026 ([ingest_effis_fwi.py:8](/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_effis_fwi.py:8), [38](/Users/guillaume/Documents/Recherche/downscalewind/services/data-ingestion/ingest_effis_fwi.py:38)).  
   Fix : protocole FWI unique : initialisation, local noon, seuils, version CEMS, résolution.

**UNVERIFIED ASSUMPTIONS**

- Débit GPU `<150 h` hors réentraînement : **vérifiable en <1 h** par benchmark 1k cas.
- M_I8 améliore les stations SYNOP val : **vérifiable en 1 j**, après script calibré.
- 100-150 stations Ogimet avec rr24 fiable à 12 UTC : **vérifiable en 1 j**.
- Licences Ogimet / redistribution SYNOP : **risque de fond**. Ogimet donne accès formulaire, pas licence claire.
- AEMET open data réutilisable : **vérifiable en <1 h** ; AEMET annonce réutilisation commerciale/non commerciale.
- EFFIS burnt perimeters 2025-2026 disponibles en shapefile exploitable : **vérifiable en <1 h** ; EFFIS indique WMS/shapefile, demandes pour historiques.
- Open-Meteo IFS historique stable et redistribuable : **vérifiable en <1 h** ; ECMWF open data est CC-BY 4.0, Open-Meteo non-commercial selon sa doc.
- Nature-family accepte pipeline CFD privé : **risque de fond**. Nature demande le code central disponible aux reviewers et encourage dépôt DOI.
- Chiffres d’actualité 2026 : **vérifiable en <1 h**, à sourcer officiellement.
- Feux 2025-2026 en relief avec périmètres de qualité : **vérifiable en 1 j**.

**PRIOR ART À TRAITER**

- CEMS/GWIS/GEFF : Vitolo et al., 2020, *ERA5-based global meteorological wildfire danger maps* ; dataset `cems-fire-historical-v1`, ERA5, 0.25°, quotidien.
- Di Giuseppe et al. sur predictability/skill FWI ECMWF, dont NHESS 2020 sur EPS.
- Van Wagner 1977/1987, `cffdrs` R/Python pour FWI horaire et journalier.
- WindNinja : Forthofer et al. 2014 IJWF ; Wagenbrenner et al. 2016 ACP, downscaling de vent en terrain complexe pour feu.
- Le Toumelin et al. 2024, *A two-fold deep-learning strategy to correct and downscale winds over mountains* ; c’est l’antériorité directe M_I8/DEVINE.
- Miralles et al., *Downscaling of Historical Wind Fields over Switzerland Using GANs*, Weather and Forecasting/AIES 2023.
- Nature Comm 2026, *Reconstructing fine-scale 3D wind fields with terrain-informed machine learning* : à citer si réellement proche ; je n’ai pas vérifié l’indépendance méthodologique au-delà du résumé.
- Produits FWI Europe downscalés/biais corrigés CMIP6, FirEUrisk/Hetzer et al. : important pour positionner “haute résolution” vs “projection climatique”.

**SCOPE**

À couper : IFS prévision, M_I9b T/RH, feux 2025-2026 multiples, Landes contrôle, grand banc Méditerranée Ogimet, FWI horaire sauf annexe, revendication Nature Communications.

Manque pour tier Nature : validation indépendante en relief réel, incertitudes/bootstrap, benchmark WindNinja ou NWP haute résolution, code/données déposés, protocole seuils nationaux, accès aux données externes.

MVP en 10 lignes :
1. France + Espagne/Pyrénées si licence claire.  
2. Étés 2022-2023, saison complète.  
3. Banc figé avant scoring.  
4. M_I9 excluant toutes jumelles du banc.  
5. ERA5+pluie corrigée vs ERA5+CEMS vs vent calibré+pluie corrigée.  
6. Métriques FWI, FFMC, ISI, extrêmes FWI ≥30/50.  
7. Stratification relief simple mais justifiée.  
8. Perdigão hors entraînement comme démonstration relief, sinon annexe.  
9. Pas de prévision opérationnelle.  
10. Article cible : GMD/NHESS/ESSD-style, puis upscale.

**ORDER**

Changer l’ordre :  
P0a licences/données/seuils/spin-up ; P0b fuite spatiale et exclusion ; P0c inférence M_I8 station ; gate dur.  
P1 réentraînement M_I9 seulement si P0c positif.  
P2 scoring station complet.  
P3 spatial seulement si moteur régional existe et validé.  
P4 rédaction après gel des figures.

**EFFORT**

- P0 3-5 j, GPU <2 h : trop optimiste. Le codage M_I8 station + split jumelles + matérialisation + scoring = 5-8 j.
- P1 10 j + 2×40 h : trop bas si Ogimet multi-pays et retrain propre. Prévoir 3-4 semaines.
- M_I9b “un job 40 h” : sous-estime dataset/loss/QC T/RH ; plutôt chantier séparé.
- P2 10-15 j : bas. CEMS/CDS, IMERG/GEE, seuils nationaux, IFS q/RH : 3 semaines minimum.
- P3 10 j : irréaliste sans `infer_regional.py`; 3-5 semaines avec QA.
- P4 15-20 j : plausible seulement si résultats déjà propres.

**Verdict**

Pas publiable à ce niveau maintenant.  
Publiable en revue spécialisée si P0 montre un gain hors échantillon clair.  
Nature-family seulement avec validation indépendante en relief et code central auditable.  
Le papier doit revendiquer le vent/ISI résolu, pas “FWI 33 m” plein.  
Condition minimale : M_I9 sans fuite, banc figé, protocole CEMS/spin-up/licences irréprochable.

Sources externes consultées : CEMS historical FWI dataset, EFFIS data services, Nature code policy, WindNinja USFS, Le Toumelin et al. 2024, Vitolo et al. 2020.
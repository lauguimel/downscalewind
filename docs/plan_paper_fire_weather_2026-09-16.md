# Plan de développement — Papier 2 : vent et propagation initiale résolus à 33 m pour la météo du feu

Date : 2026-09-16. Version 1.1, révisée après la relecture Codex du même jour
(`docs/codex_review_paper_plans_2026-09-16.md`). Statut : à valider par Guillaume.
Vocabulaire : FWI = Fire Weather Index, indice canadien de danger d'incendie calculé à midi
local depuis température, humidité relative, vent à 10 m et pluie des 24 h ; FFMC/ISI = ses
composantes rapides (humidité du combustible fin, propagation initiale) ; BUI = composante
lente (pluie des semaines passées) ; CEMS/EFFIS = produit européen opérationnel de FWI sur ERA5
à 25 km ; « M_I8 » = correction du vent apprise sur stations, en amont du surrogate v3 gelé ;
« test d'arrêt » = expérience dont un résultat négatif arrête le projet.

## 0. Ce que la relecture a changé (v1.0 → v1.1)

- Titre et revendication : on ne dit plus « FWI à 33 m ». Seule la composante rapide (FFMC,
  ISI) varie spatialement : la pluie est uniforme à la tuile et BUI vient du pilote. Les cartes
  portent sur ISI/FFMC ; le FWI complet n'est scoré qu'aux stations, avec ablations.
- Le banc de mai est un banc d'extrêmes : stations choisies par un score qui pèse p95 du FWI
  observé et nombre de jours > 38 (`build_fwi_station_audit.py:247-253`), puis les 20 jours de
  FWI observé le plus fort par station (`:312`). Vérifié. Il faut deux bancs : saison complète
  (compétence climatologique) et sous-échantillon extrême (test de résistance).
- Fuite d'entraînement traitée par exclusion géographique (< 3-5 km, jumelles ISD/OMM), pas
  seulement par identifiant.
- Périmètre réduit : pas de bras prévision IFS, pas de bras T/RH, pas de feux multiples ni de
  contrôle Landes dans la version soumise ; ces éléments passent en extension.
- Protocole FWI unique à écrire avant tout score : initialisation et amorçage (janvier-mai),
  heure locale de midi, seuils, version et résolution CEMS.
- Efforts revus à la hausse ; ordre de P0 changé (licences et fuite avant l'inférence).
- Antériorités ajoutées : Vitolo et al. 2020 (CEMS/GEFF), Di Giuseppe et al. (compétence FWI
  ECMWF), WindNinja (Forthofer 2014, Wagenbrenner 2016), Miralles 2023 (GAN Suisse),
  produits FWI CMIP6 corrigés (FirEUrisk).

## 1. Revendication et chiffre qui voyage

1. Aux stations méditerranéennes, un indice calculé depuis ERA5 avec le vent calibré à 33 m et
   la pluie corrigée est plus proche de l'indice observé que le même indice avec le vent ERA5,
   et que le produit CEMS ; le gain porte sur ISI et sur la détection des jours FWI ≥ 30 et ≥ 50.
2. Sur relief, l'ISI résolu à 33 m révèle un contraste crête-vallée que le pixel à 25 km ne
   voit pas, mesuré à Perdigão sur des mâts jamais vus à l'entraînement.

Ce que le papier ne revendique pas : prédiction des feux, température et humidité calibrées,
prévision (tant que le test IFS n'est pas fait), « FWI à 33 m » plein.

Public : version soumise d'abord en revue spécialisée (NHESS, GMD, Agricultural and Forest
Meteorology) sauf si P0 et P2 donnent un gain hors échantillon large et propre, auquel cas
Communications Earth & Environment ou Nature Communications, avec validation indépendante en
relief, intervalles par rééchantillonnage, comparaison WindNinja ou PNT haute résolution, et
code central déposé. Actualité 2026 (434 000 ha en Europe, Gironde-Landes) à sourcer
officiellement ; le mégafeu français est en terrain plat, donc contrôle et non vitrine.

## 2. État de départ (inventaire du 2026-09-16, vérifié par Codex)

Acquis :
- Banc de 20 stations SYNOP françaises, été 2022, 2 354 station-jours, produits de référence
  scorés (`build_fwi_station_audit.py`, `build_fwi_baseline_validation.py`, résultats
  `data/validation/fwi_station_audit_2022_n20*`). Résultat solide : pluie ≈ 53 % / météo ≈ 47 %
  de l'erreur de l'indice ERA5-Land. Mais banc d'extrêmes (voir §0).
- Correction de pluie : `StratifiedQMCorrector`, modèle `qm_stratified.npz` ; gain mesuré
  4.22 → 3.14 avec météo ERA5. Pas de CLI autonome.
- Indice horaire (Van Wagner 1977) : `shared/fwi.py::hffmc`, `compute_hfwi_series` ; les 12
  tests de `shared/tests/test_fwi_hourly.py` passent (vérifié 2026-09-16). Non commité.
- Vent calibré : surrogate v3 + M_I8 ; motif d'appel `run_masts_M_K2.py`.
- Ingestion prête, jamais exécutée : `ingest_effis_fwi.py` (codé pour Ibérie 2017, bbox et
  années à élargir), `ingest_ifs_openmeteo.py` (q manquant). Observations : SYNOP Météo-France
  (dans le script d'audit), `ingest_ogimet.py` (SYNOP décodés, pluie 24 h à 12 UTC ; licence
  de redistribution non claire), `obs_unified_aemet_es.zarr`, périmètres via GEE.

Faux ou manquant :
- Seul test complet (12 mai) négatif : vent surrogate v2 non calibré, 6.11 contre 5.31 pour
  ERA5 + pluie corrigée, sur 400 cas extrêmes. Le vent calibré n'a jamais été rebranché.
- Aucun script station n'applique M_I8 (`run_station_surrogate_inference.py:43` charge le
  surrogate seul).
- Fuite : les 20 stations ont une jumelle ISD à < 2.1 km dans le jeu d'entraînement M_I8
  (`exclude_substrings: []`) ; le split est par identifiant, pas par proximité.
- T et humidité ≈ ERA5, non calibrées. Queue de vent fort mince dans la CFD (34 cas > 8 m/s).
- Amorçage FWI : la fenêtre est recalculée depuis le premier jour sélectionné ; l'amorçage
  janvier-mai complet n'est pas garanti (rapport de mai).

## 3. Phases

### P0 — Test d'arrêt (5-8 j, GPU < 2 h)

Ordre : ce qui invalide le test avant le test.

1. Protocole FWI écrit et figé (`docs/fwi_protocol.md`) : amorçage depuis janvier avec ERA5,
   heure locale de midi (UTC+1/+2 selon pays), seuils 30 et 50, version CEMS
   (`cems-fire-historical-v1`, ERA5 0.25°), variables et hauteurs (vent 10 m, T/RH 2 m).
2. Fuite : sur Aqua, recalculer le split M_I8 (`watertight_station_split`, seed 42) et
   marquer chaque station SYNOP par sa jumelle ISD (< 3 km). Étendre aux ~40 stations SYNOP de
   la boîte France pour grossir le sous-ensemble `val`. Ne publier que `val`.
3. Deux bancs : saison complète juin-septembre 2022 (tous les jours, toutes les stations
   retenues par couverture ≥ 75 %) et sous-échantillon extrême (banc de mai, étiqueté).
4. Inférence calibrée : `services/validation/infer_fwi_stations_calibrated.py`, calqué sur
   `run_masts_M_K2.py` : pour chaque station-jour à 12 UTC, `build_one`, `speed_raw`,
   `speed_corr` à 10 m, T et q à 2 m → RH ; colonnes attendues par
   `build_fwi_baseline_validation.py --downscaled-uvt`. PBS dérivé de `mk2_crest_deficit.pbs`.
   Attention au volume : saison complète ≈ 40 stations × 120 jours = 4 800 cas (≈ 1 h A100).
5. Score de trois chaînes : (a) ERA5 T/RH + vent calibré + pluie corrigée ; (b) ERA5 T/RH +
   vent ERA5 + pluie corrigée ; (c) CEMS. Métriques : RMSE/MAE/biais sur FWI, FFMC, ISI ;
   détection FWI ≥ 30 et ≥ 50 ; intervalles par rééchantillonnage des stations.

Porte P0 (dure) : sur les stations `val` du banc saison complète, (a) < (b) sur ISI et FWI, et
meilleure détection des jours extrêmes. Sinon le papier 2 s'arrête ; le vent calibré reste une
perspective du papier 1.

### P1 — Banc final figé et réentraînement sans fuite (3-4 semaines, 1 job 40 h)

1. Banc final : France (SYNOP Météo-France, ~40) + Espagne si la licence des observations
   permet la publication (AEMET open data : réutilisation annoncée, à confirmer ; Ogimet :
   accès sans licence claire, à éviter pour la version publiée). Étés 2022 et 2023. Sélection
   par critères objectifs écrits avant tout score (couverture, présence de pluie 24 h,
   classes de relief représentées).
2. M_I9 : réentraînement M_I8 en excluant les jumelles ISD du banc (< 3-5 km) et les mâts
   Perdigão (`exclude_substrings` rempli, `init_from` M_I8). Sert aussi au papier 1.
3. Contrôle : perte de performance vent M_I9 vs M_I8 sur le val commun.

### P2 — Références et score final (3 semaines)

1. CEMS : étendre `ingest_effis_fwi.py` (bbox Méditerranée, 2022-2023) ; indice au pixel des
   stations.
2. Pluie : CLI `services/module3-precip/apply_qm_correction.py` autour de
   `StratifiedQMCorrector` ; IMERG aux stations via le téléchargement GEE existant ; variante
   « pluie observée » (rr24 SYNOP) pour isoler le terme vent.
3. Score final du banc avec M_I9 : chaînes (a), (b), (c), plus « vent calibré + pluie observée ».
   Stratification par relief (masque du papier 1 : écart-type d'altitude, pente, position
   topographique — `class_topo` par altitude seule ne suffit pas) et par classe de vent.
   Ablation un facteur à la fois (pluie / vent / T-RH), comme l'audit de mai.

Porte P2 : gain hors échantillon par pays et par classe de relief, avec intervalles.

### P3 — Démonstration relief (2-3 semaines, dépend de `infer_regional.py` du papier 1)

1. Perdigão hors échantillon (M_I9) : contraste crête-vallée de l'ISI à 12 UTC, réutilise
   `data/validation/crest_deficit/` (classification par position topographique, intervalles).
2. Un seul feu réel sur relief (Aude, août 2025, Ribaute) : ERA5 horaire du jour, ISI horaire
   à 33 m sur les tuiles couvrant le périmètre, comparaison au pixel CEMS, superposition du
   périmètre EFFIS (format et disponibilité à vérifier). Chiffre : fraction de la surface
   brûlée au-dessus du seuil selon chaque produit.
   Si le moteur régional n'existe pas encore : compter une semaine pour un moteur minimal.

### P4 — Rédaction (15-20 j de Guillaume)

Commiter `shared/fwi.py` (tests verts). Figures : (1) carte des stations, gain de détection des
jours extrêmes ; (2) Perdigão hors échantillon ; (3) Aude, ISI à 33 m vs pixel, périmètre ;
(4) ablation pluie / vent / T-RH. Méthodes : chaîne, calibration (Le Toumelin 2024 en
antériorité), protocole pré-enregistré, limites (§4), déclaration de disponibilité.

### Extensions (hors version soumise)

Bras prévision IFS (adapter `ingest_ifs_openmeteo.py`, q depuis RH, 10 niveaux canoniques ;
pluie prévue ; licence Open-Meteo non commerciale à vérifier) ; bras T/RH (M_I9b, jeu de données
et perte multi-variable : chantier séparé) ; feux ibériques 2026 ; contrôle Landes ; indice
horaire en carte.

## 4. Risques et limites à écrire

- P0 peut être négatif : c'est sa fonction.
- Seul l'ISI est résolu spatialement ; la pluie est uniforme à la tuile ; T/RH ≈ ERA5.
- Stations SYNOP = aéroports, terrain doux ; le gain de vent y est plus faible qu'en relief.
  Inclure les stations d'altitude disponibles (Millau, Embrun, Le Puy) et stratifier.
- Queue de vent fort mince dans la CFD ; la carte un jour de tempête de feu est une
  extrapolation à signaler.
- Référence honnête pour un hindcast = CEMS sur ERA5 ; le produit prévision n'est comparable
  que dans un bras IFS, hors périmètre.
- Licences des observations (Ogimet, AEMET) et des périmètres EFFIS : à vérifier avant usage.
- Nature exige le code central disponible aux relecteurs ; pipeline CFD privé acceptable si
  `shared/fwi.py`, l'inférence, la correction et les sorties sont déposés avec DOI.

## 5. Budget indicatif (révisé)

P0 5-8 j, GPU < 2 h ; P1 15-20 j + 40 h H100 ; P2 15 j ; P3 10-15 j (+ 5 j si moteur régional
absent) ; P4 15-20 j de Guillaume. Durée : 3-4 mois jusqu'au brouillon, arrêt possible dès P0.

## 6. Décisions à prendre

1. Lancer P0 maintenant, indépendamment du papier 1.
2. Accepter la revendication réduite (ISI/FFMC résolus, FWI aux stations).
3. Espagne dans le banc : oui seulement si la licence AEMET permet la publication.
4. Politique de code : publier `shared/fwi.py`, l'inférence et la correction avec DOI.

# Plan de développement — Papier 1 : vent calibré à hauteur de moyeu en terrain complexe

Date : 2026-09-16. Version 1.1, révisée après la relecture Codex du même jour
(`docs/codex_review_paper_plans_2026-09-16.md`). Statut : à valider par Guillaume.
Vocabulaire : « surrogate » = réseau qui remplace la CFD ; « M_I8 » = réseau de correction
appris sur stations, placé en amont du surrogate v3 gelé ; « facteur de charge » = production
moyenne d'une éolienne rapportée à sa puissance nominale ; « hors échantillon » = mesuré sur des
sites que le modèle n'a jamais vus à l'entraînement.

## 0. Ce que la relecture a changé (v1.0 → v1.1)

- Revendication réduite : plus « premier atlas horaire Europe », mais « méthode CFD-surrogate
  calibrée sur observations, validée hors échantillon à hauteur de moyeu, démontrée sur massifs
  ciblés ». L'atlas Europe devient une extension, pas la colonne vertébrale.
- Ordre de P0 inversé : moteur régional et accès aux données de validation d'abord, bord de
  patch ensuite.
- Budget stockage corrigé : 100 × 100 km à 100 m, 8 760 h, 3 hauteurs, float16 = 53 Go par
  variable, pas 8 Go. On ne stocke pas de champs horaires denses hors région pilote.
- Efforts revus à la hausse (×1.5 à ×2 selon la phase).
- Ajout des antériorités énergie (Applied Energy 2022 et 2025, NEWA, KNW, NORA3, CERRA, WAsP).
- Risque licence OSM (ODbL, partage à l'identique) signalé pour l'inventaire d'éoliennes.

## 1. Revendication et chiffre qui voyage

Revendication : un surrogate CFD 3D à 33 m, calibré sur observations à travers le modèle gelé,
donne à hauteur de moyeu (80-150 m) en terrain complexe un vent plus juste qu'ERA5 direct et
que les atlas climatologiques (GWA, NEWA), et ce gain se mesure sur des mâts et des parcs jamais
vus à l'entraînement.

Chiffre cible : biais du facteur de charge mensuel à Penmanshiel et biais de vitesse au moyeu à
Alaiz, pour trois produits (ERA5 direct, atlas climatologique, nous). Puis, en extension :
écart de ressource ERA5 direct / nous sur un ou deux massifs pilotes. Le signe net par massif
n'est pas connu (ERA5 sous-estime les crêtes en vent fort, surestime les vallées).

Public : Nature Communications (comme méthode validée hors échantillon), sinon Wind Energy
Science ou Applied Energy. Nature Energy et Joule ne sont pas atteignables avec ce périmètre
(verdict Codex, partagé).

Antériorités à traiter dans l'introduction : Global Wind Atlas 3 (DTU, 250 m, WAsP) ; New
European Wind Atlas (3 km + couche micro 50 m, campagnes Alaiz ALEX17 et Kassel) ; atlite et
renewables.ninja (ERA5 direct → facteur de charge) ; Staffell & Pfenninger 2016 ; « Validation
of European-scale simulated wind speed and wind generation time series » (Applied Energy 2022 :
ERA5, NEWA, GWA, 32 mâts) et sa suite 2025 sur sillages et ajustements micro-échelle ; KNW,
NORA3, CERRA comme réanalyses régionales ; FuXi-CFD ; Le Toumelin 2024 (mécanisme de
calibration) ; WAsP (pourquoi un surrogate CFD apporte plus qu'une mise à l'échelle linéarisée).

## 2. État de départ (inventaire du 2026-09-16, vérifié par Codex)

Acquis :
- Surrogate v3 0-200 m, 32 niveaux ; correction M_I8 multi-hauteurs ; config
  `configs/training/devine_style_M_I8_multiheight_v3.yaml`. Bat le brut aux 7 hauteurs.
  Checkpoints sur Aqua, pas dans le dépôt local (à vérifier avant tout job).
- Matérialisation d'une grille d'entrée 180×180 à (lat, lon, t) arbitraire :
  `services/module2b-surrogate/extract_v2_input_at_coords.py::build_one` (un patch 6 km).
- Motif surrogate + correction : `eval_devine_style.py:74-78`, `run_masts_M_K2.py`.
- Stores ERA5 horaires Europe par saison sur Aqua (lon −10→20, lat 35→49), année continue
  déc. 2022 → nov. 2023.
- Métriques par strates : `services/validation/utils/strata.py` — attention, `class_topo`
  classe par altitude seule, pas par relief.

Manquants :
- Inférence plein domaine ou tuilée : tous les scripts jettent le champ et gardent le pixel
  central. Débit actuel 3.9 appariements/s, limité par la matérialisation CPU.
- Courbes de puissance, facteur de charge, SCADA, GWA : aucun code.
- Validation hors échantillon à hauteur de moyeu : tours ICOS et Perdigão sont dans M_I8. Il
  faut des mâts nouveaux, ingérés et pré-enregistrés (critères écrits avant les scores).
- Masque « terrain complexe » cohérent avec l'entraînement (écart-type d'altitude, pente,
  position topographique) : à écrire ; `class_topo` ne suffit pas.

## 3. Phases

### P0 — Moteur, données, accès (8-12 j de travail supervisé, GPU < 5 h)

Ordre imposé par la relecture : ce qui casse tout d'abord.

1. Moteur régional minimal : `services/module2b-surrogate/infer_regional.py`. Cache statique
   par tuile (terrain, z0, lat) construit une fois via `build_one` ; par heure, vecteur ERA5
   3×3 × niveaux × variables pour toutes les tuiles, lot GPU ≥ 64, `surrogate(ANN(era5))`.
   Sorties limitées : (a) colonnes 32 niveaux aux points de validation ; (b) climatologie
   33 m en ligne (moyenne, Weibull k et A, percentiles) par hauteur ; (c) champs horaires
   denses uniquement sur la région pilote, à 100 m, hauteurs {100, 150 m}, float16, chunks
   par mois. Mesure de débit sur 1 000 tuile-heures (H100, A100, A6000). Cible ≥ 50/s ;
   sous 10/s, P3 passe en heures échantillonnées.
2. Accès et licences des jeux de validation, vérifiés et notés dans le plan avant toute
   ingestion : Alaiz ALEX17 (CENER, portail CKAN, public à confirmer) ; Rödeser Berg / Kassel
   (Fraunhofer, accès moins clair) ; Penmanshiel et Kelmarsh (Cubico, Zenodo, CC-BY 4.0) ;
   La Haute Borne (ENGIE, ouvert). Inventaire d'éoliennes : OPSD (licence permissive) ; OSM
   seulement si la base dérivée peut rester sous ODbL, sinon exclu.
3. Test minimal Alaiz + Penmanshiel : ERA5 téléchargé pour les deux boîtes et les années
   (`ingest_era5_europe_hourly.py`, file CDS 1-3 j), inférence aux mâts, vitesse au moyeu
   observée vs ERA5 direct vs nous. C'est le test qui décide de l'angle.
4. Fenêtre valide du patch : erreur surrogate vs CFD en fonction de la distance au bord sur le
   split test (script `analysis/patch_edge_error.py`, réutilise `evaluate_v2_physical.py`).

Porte P0 : débit mesuré ; données de validation accessibles avec licence claire ; à Alaiz
et Penmanshiel, notre vitesse au moyeu bat ERA5 direct. Sinon l'angle s'arrête ici.

### P1 — Validation hors échantillon complète (25-35 j)

1. Ingestion et contrôle qualité des mâts et SCADA (alignement des hauteurs, nettoyage des
   indisponibilités, filtrage des sillages internes du parc par secteur de direction).
2. Métriques pré-enregistrées : vitesse au moyeu (MAE, biais, pente pred~obs), facteur de
   charge mensuel (biais, corrélation), Weibull, par classe de vent. Trois colonnes partout :
   ERA5 direct, atlas climatologique (GWA/NEWA, climatologie seulement), nous. Intervalles par
   rééchantillonnage. Tours ICOS reportées à part, étiquetées « en échantillon ».
3. Masque terrain complexe cohérent avec la campagne v2 (écart-type d'altitude 6 km, pente,
   position topographique), appliqué aux sites de validation pour stratifier.
4. Si Perdigão hors échantillon est voulu dans ce papier : réentraînement M_I9 avec les mâts
   exclus (partagé avec le papier 2, 40 h H100).

Porte P1 : gain hors échantillon à hauteur de moyeu, avec intervalles, sur au moins deux
sites de relief et un contrôle plat.

### P2 — Chaîne éolienne et région pilote (20 j)

1. `wind_energy/power.py` : courbes ouvertes (windpowerlib, turbine de référence IEA 3.4 MW),
   correction de densité, facteur de charge. Tests unitaires.
2. Baseline ERA5 direct à la façon d'atlite : même courbe, seule la vitesse change.
3. Région pilote Navarre-Aragon (parc dense en relief, mât Alaiz) : inférence horaire 2023
   ou 1 000 heures stratifiées ; écart de ressource ERA5 direct / nous / GWA pour les
   éoliennes OPSD de la région ; agrégats confrontés à la production régionale seulement comme
   ordre de grandeur (sillages et indisponibilités non modélisés, à écrire).

### P3 — Extension (optionnelle, après acceptation interne des figures P1-P2)

Un deuxième massif (Alpes du Sud ou Massif central) en heures échantillonnées ; pas d'atlas
Europe horaire dense (stockage et calcul non tenables : 10-20 k tuiles × 8 760 h). Une carte
climatologique Europe de relief à 33 m reste possible en heures échantillonnées si le débit
P0 le permet ; elle est une figure, pas une revendication.

### P4 — Figures et rédaction (15-20 j de Guillaume)

Figure 1 : Alaiz et Penmanshiel, vitesse au moyeu et facteur de charge mensuel, trois produits.
Figure 2 : Navarre-Aragon, écart de ressource cartographié, éoliennes superposées.
Figure 3 : une vallée un jour de vent fort, ERA5 constant vs champ à 33 m.
Figure 4 : gain par classe de vent et de relief. Méthodes : opérateur, calibration (Le Toumelin
2024 en antériorité), protocole pré-enregistré, limites.

## 4. Risques et limites à écrire

- Débit et E/S non mesurés ; P0.1 les mesure avant tout engagement.
- Qualité à 100-150 m hors échantillon inconnue avant P0.3 ; c'est le risque de fond.
- ERA5 hors domaine actuel (Écosse pour Penmanshiel) : téléchargement CDS obligatoire.
- Surrogate résiduel sur ERA5 : biais et moyennes robustes, rampes horaires non revendiquées.
- Direction non calibrée explicitement ; sillages, givrage, indisponibilité, rafales,
  intensité turbulente absents → aucune revendication de charges ni de production réelle.
- Terrain très raide (pente > 20°) exclu de l'entraînement CFD : à signaler sur les cartes.
- Politique de code : Nature exige le code central disponible aux relecteurs ; pipeline CFD
  privé acceptable seulement si l'inférence, la calibration et les sorties sont déposées avec
  DOI et suffisent à vérifier les conclusions. À trancher avant d'écrire.

## 5. Budget indicatif (révisé)

P0 8-12 j ; P1 25-35 j ; P2 20 j ; P3 optionnel ; P4 15-20 j de Guillaume. GPU : P0 < 5 h,
P1 < 50 h, P2 50-100 h. Stockage : < 100 Go hors champs horaires de la région pilote (≈ 100 Go
de plus). Durée calendaire : 4-5 mois jusqu'au brouillon.

## 6. Décisions à prendre avant P0

1. Accepter la revendication réduite (méthode validée hors échantillon) plutôt que l'atlas.
2. Région pilote : Navarre-Aragon (recommandé, validation Alaiz sur place).
3. Publier les sorties (climatologies, séries aux points) avec DOI, pipeline CFD privé.
4. Journal : Nature Communications en premier choix, Wind Energy Science en repli.

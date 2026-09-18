# Relectures adverses — hypothèse « petit réseau seul » (ablation du surrogate) — 2026-09-18

Deux relecteurs sur un dossier commun : (1) spécialiste du downscaling statistique + éditeur Nature (Claude, antériorités vérifiées sur le web), (2) arbitrage Codex (codex-cli 0.133.0, lecture seule du dépôt, vérification du code du bras A). Le dossier est en anglais (langue de travail des agents).

---

# Dossier commun

# Brief — adversarial review of the "small network only" project (2026-09-18)

Background facts: see /private/tmp/claude-501/-Users-guillaume-Documents-Recherche-downscalewind/7729ca38-08c6-45eb-9a63-12da0219a963/scratchpad/brief_adversarial_2026-09-18.md
and the three prior reviews in docs/adversarial_reviews_perimeter_2026-09-18.md (repo). Key fact driving this:
at 10 m, the raw CFD surrogate loses to ERA5 everywhere (bias -1.3 m/s) and ALL the measured gain over ERA5
(+20 % in the top relief quintile, +32 % for stations > 300 m above their ERA5 cell, ~0 elsewhere, -57 % in calm
strong-relief hours) comes from a 92 k-parameter correction network placed upstream of the surrogate.

## The PI's new hypothesis (his words, paraphrased)
"Is the direct modification of ERA5 by a small network enough? Do we really need a surrogate with tens of millions
of parameters and 12 000 CFD simulations? If the small network alone matches the full chain, that is a stronger
result: it shows FuXi-CFD-style surrogates are unnecessary for point predictions, and even DEVINE's upstream
correction is not the interesting part — the small network is. It could be deployed on any free tier VM behind a
simple API. The downscaler should be source-agnostic (ERA5, IFS, AROME...). It must use the real relief: either
the terrain patch itself, or only relief METRICS. Metrics lose 'where the wind comes from', but maybe wind-relative
metrics (ridge orientation vs wind direction, upwind slope, exposure) are enough."

## What is being run right now (ablation arm A, job 25437506 on the HPC)
ANNDirect: same inputs as the M_I8 correction network (ERA5 3x3 cells x 10 pressure levels x {u,v,T,q} + surface
t2m/d2m/u10/v10 + lat + z0, 12 topo/physics features [mean & std of elevation in 1 km, z0, lat, hour/month sin-cos,
T gradients 850-surface and 500-850, RH, q], the 4-channel 180x180 terrain patch at 33 m through a small CNN encoder,
plus the target height AGL). Output = (du, dv) residual to ERA5 u10/v10 at the station. Same data (766 k rows,
674 stations/towers, 10-200 m), same station-id split, same regime loss, same 4 epochs. 88 k parameters.
Planned arm B: same without the CNN, with relief metrics only (to be designed — this review should say which).

## Known prior art (verify and extend)
Wind-Topo (Dujardin & Lehning 2022, QJRMS): COSMO-1 + DEM patch CNN -> 10 m wind at 261 Swiss stations.
TerraWind (2024, GRL). DEVINE / Le Toumelin 2024 (NPG): AROME + CNN downscaler trained on ARPS LES, obs correction upstream.
Winstral Sx (upwind slope index), Sx/exposure descriptors used in snow/wind literature. WindNinja (mass-conserving /
momentum solver). FuXi-CFD (Lin et al. 2026 NatComms). Kim et al. NWP-CFD Korea. Neural operators 3D wind (Comms Phys 2026).
Classic statistical downscaling of station wind with terrain predictors (e.g. Salameh 2009, Pryor & Barthelmie).

## Questions for the reviewers
1. If arm A matches M_I8 (within ~2 skill points) — what exactly is the publishable claim, and is it new given
   Wind-Topo/TerraWind? If arm A LOSES clearly — what does that prove about the surrogate?
2. Design of arm B: which relief metrics, at which scales, wind-relative or not, and what would make "metrics only"
   fail where a CNN succeeds? What is the cheapest experiment that separates the two?
3. Where does a point-only small network structurally fail (spatial fields, extremes, heights, calm regime,
   out-of-distribution forecasts as input) and does any of that matter for the PI's goals (fire weather ISI at 33 m,
   hub-height wind, a public point downscaler)?
4. "Source agnostic": what is the minimal test that would justify feeding IFS/AROME to a network trained on ERA5?
5. Venue and framing if the small network wins; venue and framing if it loses.

---

# Relecture adverse — hypothèse « petit réseau seul » (2026-09-18) — relecteur 1 (Claude, web vérifié)

## Antériorités vérifiées

- Le Toumelin et al. 2024, NPG 31:75 (https://npg.copernicus.org/articles/31/75/2024/). ANNspeed = 17 entrées dont 5
  topographiques (élévation 30 m, TPI500m, laplacien, pente, courbure) + élévation AROME, 2 couches [50, 10] ≈ 1 400
  paramètres. 218 stations d'entraînement / 55 de test (Alpes). Table 3, MAE vitesse : AROME 1,34 ; DEVINE seul 1,29 ;
  ANN seul 1,21 ; ANN+DEVINE 1,16 ; analyse AROME 1,18. Direction : 44°/43°/35°/35°/37°. Table 4, transfert
  Corse+Pyrénées (39 stations jamais vues) : AROME 1,53 ; ANN seul 1,69 (dégrade) ; ANN+DEVINE 1,64.
- Wind-Topo, Dujardin & Lehning 2022, QJRMS (https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4265). Entrées :
  COSMO-1 (1,1 km) u,v,w à 10/89/293/589/1164 m + DEM 53 m, CNN. 261 stations d'entraînement, 60 de validation.
  Stations alpines : MAE 1,77 → 1,21 m/s (−32 %), biais 0,72 → −0,07. Code AGPL.
- Hu et al. 2023, ERL (https://iopscience.iop.org/article/10.1088/1748-9326/aceb0a). XGBoost, ERA5 + élévation/pente/
  aspect/TPI 5 km et 75 km/TDI, 388 stations CH+DE ; classe la plus rugueuse RMSE 4,22 → 2,05. Pas de réseau, pas de patch.
- TerraWind, Lian 2024, GRL : CNN terrain + GNN inter-stations, Chine de l'Est ; comparé uniquement à des interpolations
  (−42,6 % MAE), pas à la NWP. Peu utile comme référence.
- Helbig et al. 2017, JGR : laplacien, pente quadratique moyenne, sky view factor comme descripteurs de downscaling en Suisse.
- FuXi-CFD 2026 : 12 000 CFD, 3 tours EU, CC BY-NC-ND, ablation uniquement contre d'autres architectures profondes —
  jamais contre un petit réseau sans CFD.

Conclusion factuelle : « petit ANN sur NWP + 5 descripteurs ≈ chaîne complète » est déjà publié (Le Toumelin 2024,
écart 4 % en faveur du downscaler). Ce n'est pas une hypothèse nouvelle, c'est une réplication attendue.

## Q1 — Si le bras A égale M_I8 / s'il perd

Si A ≈ M_I8 (±2 points) : revendication = « à l'échelle station, 674 sites, le surrogate CFD n'apporte rien de mesurable
au-delà d'ERA5 + relief + 88 k paramètres ». Pas nouveau (Le Toumelin Table 3, Wind-Topo, Hu 2023). Résultat négatif
honnête, pas un papier NatComms. Deux réserves : (a) le seuil de 2 points est sous le bruit — IC 90 % du quintile fort
[10 ; 27] ; il faut un bootstrap apparié par station sur les résidus A vs M_I8 ; (b) M_I8 et A ont la même capacité
(88–92 k) : l'égalité peut dire « le goulot est l'ANN dans les deux cas », pas « le surrogate est inutile ».

Si A perd nettement (> 5 points) : le surrogate est un prior utile au point (répond à l'attaque D) — à condition que le
CNN de A ait été un test loyal. Il ne l'est pas : l'encodeur (ann_correction.py l.106-123) fait un AdaptiveAvgPool2d(1)
sur le patch sans connaissance du vent ; le latent 48-d est indépendant de la direction et sans position. Un Sx amont
est structurellement difficile à apprendre. Une défaite de A prouve peu ; il faut le bras C (patch tourné dans le repère
du vent) avant toute conclusion.

## Q2 — Bras B : descripteurs (≤ 12, classés)

Calculables une fois par station sur le patch 180×180 à 33 m, < 1 s par station (estimé) ; les directionnels précalculés
sur 16 secteurs et interpolés à la direction ERA5 (u10/v10 et 850 hPa) à l'apprentissage.
1. Sx Winstral (dir.) : max amont de atan((z_i − z_0)/dist), d = 100, 300, 1000 m.
2. dz_cell (non dir.) : z_station − orographie de la maille source. Déjà utilisé ; le +32 % y vit.
3. TPI (non dir.), r = 300, 1000, 3000 m.
4. Dénivelé amont/aval (dir.) : moyenne de z dans le secteur amont ±45° à 1 et 3 km moins z_0 ; idem aval.
5. α de Le Toumelin (dir.) : arctan(tan(pente)·cos(dir − aspect)) à 100 et 300 m.
6. Courbure le long / travers vent (dir.), 300 m.
7. Laplacien de z (non dir.), 300 m et 1 km.
8. Écart-type du relief (non dir.), 1 km et 6 km.
9. Orientation de crête vs vent (dir.) : cos(2·(dir − axe principal du tenseur de structure à 1–3 km)).
10. Sky view factor / openness (non dir.), 500 m.
11. z0 amont pondéré (dir.), 1 km.
12. Distance à la crête amont (dir.), plafonnée à 3 km.
≈ 25 scalaires. Échec attendu des métriques : effets non locaux hors axe (sillage d'un sommet décalé, canalisation d'une
vallée à 3 km), interactions Sx×stabilité. Le CNN n'y réussit que s'il voit la direction — d'où le bras C.

Expérience la moins chère (mêmes données/split/perte/4 époques que A) : B = descripteurs directionnels à la direction
ERA5 ; B′ = les mêmes moyennés sur 16 secteurs (B − B′ isole « d'où vient le vent ») ; C = patch tourné amont à gauche +
CNN de A (C − B isole ce que le CNN apprend au-delà des métriques). Diagnostic sans entraînement : corréler le résidu de
A par station avec Sx(dir ERA5) ; corrélation > 0,3 = A ne voit pas l'amont.

## Q3 — Où un réseau ponctuel échoue structurellement

- Hauteurs : 766 k lignes dominées par le 10 m ; aucune supervision à 33 m (ISI feu), presque aucune à hauteur de moyeu ;
  le surrogate est la seule source d'un profil 10→33 m contraint physiquement.
- Calme : le −57 % est une propriété de l'ANN dans les deux chaînes.
- Extrêmes : extrapolation sans garde-fou (le surrogate n'y est guère mieux : 34 cas > 8 m/s).
- Transfert : Le Toumelin Table 4, ANN seul 1,21 (Alpes) → 1,69 (Corse/Pyrénées), sous AROME brut 1,53.
- Champ spatial : rien pour la propagation de feu, la disposition d'un parc, les cartes.
Pour les objectifs du PI : ISI 33 m et moyeu dépendent des hauteurs → un réseau ponctuel n'y répond pas ; un downscaler
public 10 m relief fort, oui, avec abstention hors domaine.

## Q4 — « Source agnostic » : test minimal

Le réseau apprend l'incrément station − maille 31 km ; avec IFS 9 km ou AROME 1,3 km l'objet est différent. Test : trois
mois, 100 stations jamais vues, relief Q5 : (i) entrer IFS analyse ou t+0–6 h (ECMWF open data, CC BY 4.0) dans le
réseau ERA5, dz_cell recalculé sur l'orographie de la source ; (ii) comparer à IFS brut et à la compétence ERA5 ;
(iii) classifieur de domaine ERA5 vs IFS sur le vecteur d'entrée (AUC > 0,7 = décalage). Critère : compétence IFS
≥ 0,8 × compétence ERA5 en Q5, aucune dégradation ailleurs.

## Q5 — Revue et cadrage

Petit réseau gagne : réplication de Le Toumelin 2024 + outil ; GMD ou EMS, ERL possible ; NatComms exclu.
Petit réseau perd, et l'écart croît avec la hauteur, le relief et le transfert (B, C, IFS) : ablation qui répond à
l'attaque D ; cadrage « prior physique 3D transférable », GMD solide, NatComms plausible seulement avec les items 2, 4, 6
de l'arbitrage Codex. Écart entre 2 et 5 points : non concluant, lancer C avant d'écrire.

## Non vérifié
Paramètres et patch de Wind-Topo ; stations et baseline NWP de TerraWind ; Winstral 2017, Long 2025, Kim Corée,
Salameh 2009, Pryor & Barthelmie non consultés. Aucun downscaler ponctuel public 2025-2026 trouvé. Coût des descripteurs
estimé, non mesuré.

---

# Relecteur 2 — arbitrage Codex

**1. Bras A**

Non, pas comme test définitif de « le surrogate est-il nécessaire ? ». C’est un bon premier stress-test ponctuel, mais il est biaisé contre le petit réseau sur plusieurs points.

Inputs : l’ANN de A reçoit bien les mêmes entrées que l’ANN de correction M_I8 : `era5_flat` 408 composantes, topo/physique 12, patch terrain 4 canaux, mêmes niveaux AGL et mêmes données/config (`configs/training/ablation_ann_only_A_terrain_cnn.yaml:14-79`, `devine_style_M_I8_multiheight_v3.yaml:90-165`; construction ERA5/terrain : `dataset_v2_obs_centered.py:269-311`; topo 12 : `dataset_v2_obs_centered.py:189-205`). Mais la chaîne M_I8 a ensuite le surrogate avec `geo` 3D, `terrain`, et `k_obs`; A n’a qu’un scalaire de hauteur.

Loss : même perte vitesse régime (`train_v2_devine_style.py:330-338`, config `:56-64`). Attention : elle supervise la vitesse, pas la direction ni `u/v`.

Données/split : mêmes parquets, poids, split station-id et filtres (`train_v2_devine_style.py:376-418`, `dataset_v2_obs_centered.py:385-397`, split `:589-617`). C’est juste pour A vs M_I8, mais pas une preuve robuste externe : le split n’est pas géographique.

Convention résiduelle : correcte. A prédit `(du,dv)` ajouté à ERA5 `u10/v10` brut (`train_v2_devine_style.py:304-313`; `ann_correction.py:218-225`). M_I8 ajoute le résidu surrogate à la baseline ERA5 corrigée par l’ANN (`train_v2_devine_style.py:315-327`). Donc la comparaison est « ERA5 + réseau direct » contre « ERA5 corrigé + surrogate résiduel », pas strictement deux estimateurs du même objet interne.

Hauteur : asymétrie importante. A reçoit `agl_levels[k_obs]/200` (`train_v2_devine_style.py:309`; `ann_correction.py:261-263`), donc une hauteur scalaire quantifiée au niveau AGL le plus proche (`dataset_v2_obs_centered.py:543-544`). Le surrogate, lui, voit le champ `geo=[z,agl]` sur toute la grille verticale (`dataset_v2_obs_centered.py:278-284`) et sort au niveau `k_obs`. Si A perd aux hauteurs, cela ne prouve pas seul que la physique CFD est nécessaire.

Zero-init : oui, A démarre bien à ERA5 : dernière couche mise à zéro (`ann_correction.py:248-252`), puis ajout à `u10/v10` (`train_v2_devine_style.py:310-311`).

Paramètres : `ANNCorrection` M_I8 = 92 576 ; `ANNDirect` A = 88 160. Comparable.

Bug/unfairness : M_I8 est fine-tuné depuis M_I7b (`devine_style_M_I8_multiheight_v3.yaml:136-139`) alors que A repart de zéro (`ablation_ann_only_A_terrain_cnn.yaml:8-9`). Quatre époques peuvent donc désavantager A. Autre unfairness : l’encodeur CNN termine par `AdaptiveAvgPool2d(1)` (`ann_correction.py:106-123`) ; il conserve des statistiques de patch mais perd fortement la position relative des formes. Ce n’est pas totalement aveugle à la direction, car il reçoit `slope_x/slope_y` et le vent ERA5, mais il ne sait pas proprement « ce relief est en amont ». Le reviewer 1 a raison sur le besoin du bras C.

Oui, la comparaison à M_I8 doit être appariée par station : bootstrap sur les différences de résidus/MAE A−M_I8, pas deux IC séparés.

**2. Arbitrage des cinq réponses de review 1**

Q1 confirmé : si A égale M_I8, le claim est un résultat négatif utile mais peu nouveau ; si A perd, cela ne prouve le surrogate qu’après correction des biais hauteur/CNN.

Q2 confirmé : B doit inclure métriques directionnelles ; B seul ne suffit pas à isoler l’information spatiale du patch.

Q3 confirmé : le petit réseau ponctuel échoue structurellement sur champs, profils verticaux, transfert et extrêmes ; c’est central pour feu 33 m et hub-height.

Q4 confirmé : « source agnostic » exige un vrai test IFS/AROME hors domaine ; sinon c’est non démontré.

Q5 confirmé mais durcir : même si A gagne, cadrage GMD/EMS outil-ablation, pas NatComms ; si A perd nettement après C/B, le surrogate redevient défendable.

**3. Valeur scientifique**

Avec Le Toumelin 2024, l’hypothèse « ANN seul presque égal à ANN+downscaler » est déjà attendue, pas neuve : ANN seul 1,21 vs ANN+DEVINE 1,16. Et le transfert ANN seul se dégrade : 1,69 vs AROME 1,53. Ce qui reste scientifiquement valable est une question de domaine : sur 674 sites européens, avec ERA5 multi-niveaux, terrain haute résolution et hauteurs 10-200 m, où un petit modèle ponctuel suffit-il, et où un prior 3D devient-il nécessaire ? Le cadrage défendable est une ablation de nécessité et de domaine d’application, pas « FuXi-CFD est inutile ».

**4. Plan après A**

1. Analyse appariée A vs M_I8 : isole la contribution mesurée du surrogate au point ; 1 jour ; stop si l’écart ± IC station est <2 points partout.

2. Bras B métriques directionnelles : isole patch CNN vs descripteurs physiques calculés ; 2-3 jours ; stop si B égale A/M_I8 en Q5 et `dz_cell>300`.

3. Bras C patch tourné vent : isole l’information amont perdue par le pooling ; 3-5 jours ; nécessaire si A perd ou si B échoue ; stop si C n’améliore pas A de >2 points appariés.

4. Re-run équitable init/compute : A et M_I8 depuis zéro ou tous deux fine-tunés, même budget ; 2-4 jours + queue ; stop si classement inchangé.

5. Test transfert source : ERA5-trained sur IFS/AROME analyses, stations relief Q5 ; 5-7 jours ; stop si skill <80 % du skill ERA5 ou dégradation hors relief.

Bras B et C sont tous deux nécessaires : B teste « métriques suffisent », C teste « le CNN actuel est mal posé ».

**5. Outil**

Si le petit réseau gagne, on peut honnêtement déployer un downscaler ponctuel ERA5, gardé, pour vitesse à station/point dans des régimes validés, surtout relief fort et vent établi, avec retour ERA5 ou abstention hors domaine. Il faut refuser les promesses de champ 3D, hub-height généralisé, fire-weather 33 m validé, prévision IFS/AROME source-agnostic, et utilité partout.
# Limitations et hypothèses du pipeline DownscaleWind

Ce fichier documente les hypothèses sous-jacentes, les conditions de validité,
et les limites connues de chaque module. Il est mis à jour au fur et à mesure
du développement.

**Principe :** une limitation non documentée est une limitation dangereuse.

---

## Données d'entrée

### ERA5 (résolution ~25 km, 6 h)

- **Lissage spatial** : les effets orographiques sous-maille (crêtes, vallées < 25 km)
  sont absents de l'entrée. Le pipeline doit les reconstruire — c'est précisément
  son objectif, mais cela implique que toute erreur de grande échelle dans ERA5
  se propage dans toutes les sorties.

- **Résolution temporelle 6 h** : les cycles diurnes rapides (brise de mer/montagne,
  courants de pente) sont partiellement capturés. Les fronts qui traversent le domaine
  en < 6 h seront mal représentés en interpolation temporelle.

- **ERA5 réanalyse ≠ prévision IFS** : le pipeline est entraîné sur ERA5 (réanalyse
  assimilant des observations) et utilisé en inférence sur IFS (prévision sans
  assimilation). Le domain shift est non caractérisé en V1 — à quantifier via
  IFS HRES Open-Meteo sur la période 2017.

- **Humidité spécifique q** : non disponible via Open-Meteo IFS. Approximée depuis
  l'humidité relative (RH) par la relation de Tetens. Erreur typique < 5% sur q
  pour T > 0°C, plus grande en conditions très froides.

---

## Module 1 — Downscaling temporel

- **Interpolation NeuralODE** : le modèle interpolle dans l'espace latent entre
  t et t+6h. Il ne peut pas créer d'événements météorologiques qui ne sont pas
  présents dans les snapshots ERA5 encadrants (pas de génération de nouveaux
  fronts ou de nouvelles perturbations).

- **MC Dropout comme proxy d'incertitude** : quantifie l'incertitude épistémique
  (variabilité du modèle) mais pas l'incertitude aléatoire (variabilité naturelle
  non capturable à 6 h de résolution). L'enveloppe d'incertitude est donc
  probablement sous-estimée.

- **Généralisation hors-distribution** : entraîné sur ERA5 2016–2017, performance
  dégradée attendue sur des situations synoptiques extrêmes non représentées
  dans la période d'entraînement.

---

## Module 2 — Reconstruction du profil vertical

- **Profil logarithmique en couche de surface (0–100 m)** : valide pour conditions
  neutres à légèrement stables. Invalide pour :
  - Stabilité très stable (Ri_b > 0.25) : la loi log surestime u*
  - Conditions très instables (Ri_b < -1) : couche limite convective, profil non-log
  - Terrain complexe dans les 100 premiers mètres (recirculations)
  Un avertissement est loggué quand Ri_b > 0.25 ou < -1.

- **Paramétrisation de Monin-Obukhov (Businger-Dyer)** : valide pour |z/L| < 1
  (L = longueur d'Obukhov). En stabilité très stable (z/L >> 1), plusieurs
  formulations existent (Beljaars-Holtslag implémentée comme option) avec des
  comportements divergents.

- **Projection de Helmholtz 1D** : l'hypothèse de colonne 1D pour satisfaire ∇·u = 0
  néglige les termes horizontaux de divergence. Valide loin du terrain, approximative
  dans les zones de fort gradient horizontal de vitesse (crêtes, vallées encaissées).

---

## Module 2A — CFD OpenFOAM

- **RANS stationnaire** : `buoyantSimpleFoam` résout les équations de Navier-Stokes
  moyennées (RANS). Il ne capture pas :
  - La turbulence résolue (fluctuations instantanées)
  - Les processus instationnaires (oscillations de sillage, vortex de Kelvin-Helmholtz)
  - Les effets thermiques transitoires (courants de pente nocturnes en transition)

- **k-ω SST** : meilleur que k-ε pour les gradients de pression adverses et les zones
  de décollement, mais calibré sur des écoulements de laboratoire. En ABL, les
  constantes du modèle sont ajustées empiriquement — la littérature montre des
  divergences en terrain très complexe (pentes > 20°).

- **Domaine 50 km, maillage 1 km** : à cette résolution, les effets de crête fine
  (largeur < 1 km) ne sont pas résolus. La double crête de Perdigão (~2 km de
  largeur) est partiellement résolue.

- **Conditions aux limites (3 faces inlet)** : la configuration left/front/right → inlet,
  back + top → outlet est une approximation pour les directions obliques (±90° max
  par rapport à l'axe principal). Au-delà, des instabilités numériques sont possibles.
  En pratique, la batch couvre 16 directions avec 22.5° de pas.

- **Dérive en atmosphère libre** : sans nudging volumique, le champ CFD au-dessus
  de 3 km peut dériver du profil ERA5 d'entrée. A mesurer sur le premier batch
  (objectif : dérive < 10% sur |u|).

- **Temps de calcul** : estimé 20–40 min/run sur 8 cœurs. Les runs en stabilité
  forte (Ri_b > 0.1) convergent plus lentement. Le benchmark sur un run unique
  doit précéder le batch complet.

---

## Module 2B — Surrogate GNN

- **Champ réceptif limité** : avec N couches GATv2, chaque nœud CFD ne voit que
  ses N-voisins dans le graphe. Pour couvrir 50 km à 1 km de résolution, il faut
  ~7 couches minimum. Les ablation studies détermineront le nombre optimal.

- **Généralisation à d'autres sites** : entraîné sur Perdigão, le surrogate a
  potentiellement appris des effets topographiques spécifiques à la double crête.
  La généralisation à d'autres configurations (plateaux, côtes, alpins) n'est
  pas garantie sans données CFD supplémentaires sur ces sites.

- **Incertitude d'ensemble** : 3–5 modèles avec initialisations différentes donnent
  une enveloppe d'incertitude. Cette enveloppe représente la variabilité du modèle,
  pas l'incertitude physique complète (erreurs CFD, erreurs ERA5, variabilité
  atmosphérique).

- **Pruning des arêtes bipartite** : le k-NN k=5 pour les arêtes ERA5→CFD est
  un choix empirique. Avec k < 3, risque de sous-contrainte synoptique ; avec
  k > 8, dominance des nœuds ERA5 sur la topographie locale.

---

## Validation

- **Biais site-spécifique de Perdigão** : les mâts de mesure sont concentrés sur
  et autour des deux crêtes principales. Les zones de vallée et les secteurs sous
  le vent sont sous-représentés dans la validation.

- **CERRA comme vérité terrain intermédiaire** : CERRA à 5.5 km résout partiellement
  les crêtes de Perdigão (~2–3 km de largeur). Son utilisation comme proxy de
  "vérité terrain méso-échelle" suppose que les erreurs CERRA sont plus petites que
  les erreurs ERA5, ce qui est généralement vrai mais pas garanti en terrain
  très complexe.

- **Critère de succès PoC** (réduction RMSE ≥ 30% vs baseline ERA5 interpolé) :
  ce seuil est ambitieux mais atteignable selon la littérature sur des cas similaires.
  À documenter après la première évaluation sur données réelles.

---

*Dernière mise à jour : initialisation du projet*
*Prochaine mise à jour : après le premier run CFD de test*

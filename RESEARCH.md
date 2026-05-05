# Module 2B — Surrogate micro-échelle : État de l'art, stratégie et plan

> Document de référence pour le surrogate GNN vent+température.
> À convertir en `RESEARCH.md` dans le repo une fois validé.

---

## 1. Paysage concurrentiel

### 1.1 Foundation models méso-échelle (25km → 2km) — entrées potentielles

| Modèle | Org | Résolution | Variables | Open source | Notes |
|--------|-----|-----------|-----------|-------------|-------|
| **CorrDiff** | NVIDIA | 25km→2km | U, T, precip | Oui (PhysicsNeMo) | Diffusion, 500× plus rapide que NWP. US + Taiwan. |
| **Prithvi WxC** | IBM/NASA | ~50km→~12km | 160 vars MERRA-2 | Oui (HuggingFace) | 2.3B params, fine-tunable. Voir détail ci-dessous. |
| **GenCast** | Google | ~25km global | U, T, Z, q | Poids (non-commercial) | Probabiliste, diffusion sur graphe. |
| **Aurora** | Microsoft | ~25km global | Multi-source | Poids (non-commercial) | Foundation model, pré-entraîné multi-datasets. |

#### Prithvi WxC en détail — potentiel comme entrée du surrogate
- **Architecture** : Vision Transformer encoder-decoder (25 encoder + 5 decoder blocks), attention locale (fenêtre 30×32 pixels) + globale (MaxViT), 2 560 dim interne, 16 heads
- **Pré-entraînement** : 160 variables MERRA-2 (0.5°×0.625°, ~50-60km), 20 vars surface + 10 vars × 14 niveaux verticaux, 1980-2019, 3h de résolution temporelle
- **Downscaling démontré** : MERRA-2 6× (300km→50km, RMSE=0.73K sur T2m) et CORDEX 12× (150km→12.5km, RMSE=0.44K)
- **Vérité terrain du downscaling** : la donnée native à pleine résolution (MERRA-2 ou CORDEX) sert de cible — **pas d'observations in-situ**
- **Fine-tuning** : backbone gelé + couches de tâche entraînées (embedding, upscaling module). Entraîné sur 64 A100 pour le pré-entraînement, 16-48 pour le fine-tuning
- **Code** : [github.com/NASA-IMPACT/Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC), poids sur [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M)
- **Limite clé** : le downscaling va de ~50km à ~12km — reste loin de notre cible 100m. Mais pourrait servir d'**entrée améliorée** (au lieu d'ERA5 brut) pour notre surrogate CFD. Le gain serait une meilleure représentation des conditions à méso-échelle, surtout T et humidité.
- **Question ouverte** : est-ce que fine-tuner Prithvi pour descendre plus bas (12km→2km ou 1km) est réaliste ? Probablement pas sans données d'entraînement haute résolution (AROME, HRES), qui sont souvent propriétaires.

**Verdict** : Ces modèles couvrent le méso-échelle. Ils sont des **entrées potentielles** pour notre surrogate, pas des concurrents directs. Le scénario intéressant : Prithvi/CorrDiff comme pré-processing → nos BC CFD sont meilleures → meilleur résultat micro-échelle. Mais ERA5 brut en altitude (>500m AGL) est déjà fiable, donc le gain marginal est à quantifier.

### 1.2 Surrogates micro-échelle (CFD → DL) — concurrents directs

#### WindSeer (ETH Zürich, Nature Communications 2024)
- **Le plus proche de notre approche.**
- Architecture : U-Net 3D encoder-decoder (conv 3D kernel 3, skip connections)
- CFD : OpenFOAM simpleFoam, **k-ε standard** (pas Parente), snappyHexMesh
- Dataset : **563 patches** terrain suisse (1.5×1.5 km), **7 361 runs** (866 terrain/dir × 15 vitesses), 93% convergence
- Entrées : distance euclidienne au terrain + mesures éparses (trajectoires UAV)
- Sorties : **Ux, Uy, Uz, TKE** (pas k ni ε séparément, **pas de température**)
- Grille : rééchantillonné sur grille régulière **91×91×96** (16.5m horiz, 11.5m vert) — perte d'info maillage
- Validation Perdigão : MAE = **1.87 m/s**, corrélation = 0.69 (9 120 prédictions, 38 mâts)
- Inférence : **21 ms** (64³) sur Jetson Orin
- Code : [github.com/ethz-asl/WindSeer](https://github.com/ethz-asl/WindSeer), licence **BSD-3**
- **Poids pré-entraînés : NON PUBLIÉS**
- Limites : domaine 1.5×1.5 km fixe, neutre seulement, pas de canopée/Coriolis, pas de T, terrains suisses uniquement

#### DEVINE (Dujardin et al., AMS AIES 2023)
- Architecture : CNN fully-convolutional
- Entraîné sur **7 279 simulations ARPS** sur topographies **gaussiennes synthétiques**
- Downscale NWP → 30m en utilisant DEM haute résolution
- MAE cross-val = **0.16 m/s** (sur données synthétiques)
- Réduction biais AROME de 27% sur 61 stations alpines
- **Vent seul, pas de température**
- [DEVINE — AMS](https://journals.ametsoc.org/view/journals/aies/2/1/AIES-D-22-0034.1.xml)

#### TerraWind (Lian et al., GRL 2024)
- Combine CNN (corrélations spatiales vent-topo) + **GNN** (liens inter-stations) + AdaptNet
- Réduction RMSE vitesse de 24-33% vs interpolation
- Intéressant pour l'architecture hybride CNN+GNN
- [TerraWind — GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL112124)

#### Correction NWP + downscale (Le Toumelin et al., NPG 2024)
- 2 ANN corrigent AROME 1.3km (direction + vitesse) en utilisant **218 stations météo** comme vérité terrain
- DEVINE (gelé) downscale ensuite vers 30m
- Gradient rétro-propagé à travers DEVINE → correction optimisée pour la sortie finale
- Pertinent si on veut aller de NWP → micro-échelle, **mais notre approche est différente** : on prend les conditions en altitude (ERA5 fiable au-dessus de 500m AGL) comme BC du surrogate CFD, pas le vent de surface
- [Le Toumelin 2024 — NPG](https://npg.copernicus.org/articles/31/75/2024/)

### 1.3 Industriels micro-échelle

| Outil | Résolution | Méthode | Prix | ML intégré ? |
|-------|-----------|---------|------|-------------|
| **Meteodyn WT** | 50m | CFD RANS | 1 500€/point | En cours |
| **WindSim 11** | ~50m | CFD + ML weather classif. | Consulting | Oui (classification) |
| **Vortex** | Variable | Méso + micro | Abonnement | Non |

Marché éolien établi, cher, lent. Un surrogate rapide et moins cher peut disrupter.

### 1.4 Fondations GNN pour simulation physique (émergent)

| Travail | Contribution clé | Pertinence |
|---------|-----------------|------------|
| **MeshGraphNets** (DeepMind, ICLR 2021) | Message-passing sur maillage non structuré, résolution-indépendant | Architecture de base candidate |
| **X-MeshGraphNet** (2024) | Partitionnement graphe + halo pour scalabilité mémoire | Nécessaire si >100k cellules |
| **SGUNET** (Apple, 2025) | Pré-entraînement + transfer learning, fine-tune avec 1/16 des données | Stratégie de pré-entraînement |
| **PIGNN-CFD** (2023) | GNN physics-informed, embed RANS dans l'architecture | Physics-informed loss |
| **Graph Diffusion urbaine** (2025) | Génératif conditionné par géométrie sur maillage non structuré | Généralisation topographique |
| **PhysicsNeMo** (NVIDIA) | Framework open source GNN + neural operators pour CFD | Infrastructure d'entraînement |

---

## 2. Question clé : Température — couplé ou découplé ?

### 2.1 Les trois options

#### Option A — simpleFoam (isotherme) + T analytique post-hoc
- simpleFoam résout U, k, ε sans température
- T est estimé après coup par un profil analytique : gradient adiabatique (-6.5°C/km) + correction stabilité (Monin-Obukhov)
- **Pour :** rapide, pipeline existant, 50k runs faisable
- **Contre :** T ne tient pas compte de l'advection par le vent local (pas de cold air pooling en vallée, pas d'effets de recirculation sur T)
- **Cas d'usage :** suffisant si l'application cible est le vent (éolien). Insuffisant pour agriculture/gel

#### Option B — buoyantBoussinesqSimpleFoam dès le départ
- Résout U, k, ε, **T** avec couplage flottabilité (Boussinesq : ρ = ρ₀(1 - β(T - T₀)))
- La température modifie le vent (flottabilité) et le vent modifie la température (advection)
- **Pour :** physiquement correct, capture les effets thermiques sur le vent (katabatiques, cold air pooling, inversions)
- **Contre :** plus lent (~30-50% overhead), convergence plus difficile en stabilité forte, plus de paramètres d'entrée (T_surface, flux radiatif ou gradient T), setup BC plus complexe
- **Cas d'usage :** nécessaire pour agriculture/gel et pour les runs stratifiés

#### Option C — simpleFoam d'abord, T en 2ème passe (scalaire passif)
- Phase 1 : simpleFoam → champ U convergé
- Phase 2 : résoudre l'équation de transport de T avec U figé (scalarTransportFoam ou ajout d'un scalaire passif)
- **Pour :** la convergence vent est plus facile sans couplage, on peut quand même avoir l'advection de T par le vent
- **Contre :** **pas de rétroaction T→U** (pas de flottabilité). Le vent ne "sait pas" que l'air froid s'accumule en fond de vallée. En conditions stables, cette rétroaction est critique (le cold air pooling renforce la stratification qui supprime la turbulence qui renforce le pooling → feedback positif)

### 2.2 Incompressible / Boussinesq / Anélastique / Compressible — le vrai sujet

Tu as absolument raison : **l'hypothèse incompressible est un problème fondamental** pour un domaine de 3-5 km de hauteur. Voici l'analyse complète :

#### Les 4 niveaux d'approximation

| Approche | Densité | Pression | Quand valide ? | Solver OpenFOAM |
|----------|---------|----------|----------------|----------------|
| **Incompressible** | ρ = const | p dynamique seule | ΔT ≈ 0, H << H_s | `simpleFoam` |
| **Boussinesq** | ρ₀(1-βΔT) dans g seulement | p dynamique | ΔT < 15-20K, **H << H_s ≈ 8.5 km** | `buoyantBoussinesqSimpleFoam` |
| **Anélastique** | ρ̄(z) variable, ∇·(ρ̄u)=0 | p̄(z) + p' | H ~ H_s, pas d'ondes acoustiques | Pas natif OF (WRF, PALM) |
| **Compressible** | ρ(p,T) gaz parfait | p total | Toujours valide | `buoyantSimpleFoam` (steady), `buoyantPimpleFoam` (transient) |

**H_s** = hauteur d'échelle de pression ≈ **8.5 km** pour l'atmosphère standard.

#### Variation de densité avec l'altitude (atmosphère standard)

| Altitude | ρ/ρ₀ | T (°C) | p/p₀ | Erreur Boussinesq |
|----------|-------|--------|------|-------------------|
| 0 m | 1.000 | 15.0 | 1.000 | 0% |
| 1 km | 0.908 | 8.5 | 0.882 | ~9% |
| 2 km | 0.822 | 2.0 | 0.785 | ~18% |
| 3 km | 0.742 | -4.5 | 0.692 | **~26%** |
| 5 km | 0.601 | -17.5 | 0.534 | **~40%** |

**Pour un domaine de 3 km de hauteur** : la densité varie de ~26% entre le sol et le sommet. Boussinesq suppose ρ constant partout sauf dans le terme de flottabilité — c'est une **erreur significative** sur la vitesse (la masse conservée est ρu, pas u).

#### Impact concret sur le vent

- **Effet d'accélération manqué** : l'air qui monte une crête se décomprime. En incompressible, il garde la même densité → pas d'accélération par expansion. En compressible, il s'allège et accélère davantage.
- **Thermiques** : la convection est fondamentalement liée à ρ(z). Sans profil de densité réaliste, les thermiques ne se forment pas correctement en RANS (et de toute façon, RANS est mauvais pour la convection résolue — c'est LES/DNS).
- **Ondes de gravité** : piégées par la stratification ρ(z). Non capturées en Boussinesq.
- **Cold air pooling** : l'air froid dense s'accumule en fond de vallée. Boussinesq capture le mécanisme (terme βΔT·g) mais sous-estime l'amplitude quand ΔT > 15K.

#### Recommandation révisée pour le solver

**Pour un domaine de 3 km de hauteur, buoyantSimpleFoam (compressible) est préférable à buoyantBoussinesqSimpleFoam**, car :
1. La densité varie de 26% — hors critère Boussinesq (Δρ/ρ << 1)
2. L'équation d'état ρ = p/(RT) est exacte, pas une approximation
3. Le coût supplémentaire vs Boussinesq est modeste (~20-30% plus lent)
4. La pression hydrostatique p̄(z) est naturellement incluse

**Mais** : l'alternative pragmatique est de **limiter la hauteur du domaine** à ~1.5 km AGL (au lieu de 3 km), ce qui est suffisant pour les applications vent de surface et rend Boussinesq acceptable. WindSeer utilise un domaine de min 1.1 km — et ils ne font même pas de température.

#### Scénarios solver recommandés

| Scénario | Hauteur domaine | Solver | Température | Usage |
|----------|----------------|--------|-------------|-------|
| **Éolien neutre** | 1.5-2 km | simpleFoam | Non | Micro-siting, TI |
| **Éolien + stabilité** | 1.5-2 km | buoyantBoussinesqSimpleFoam | Oui | Production nocturne |
| **Agriculture / gel** | 1-1.5 km | buoyantBoussinesqSimpleFoam | Oui | Cold air pooling en vallée |
| **Général haute qualité** | 3 km | **buoyantSimpleFoam** | Oui | Référence, conditions extrêmes |

### 2.3 Ce que dit la littérature (performance vent)

**L'impact du couplage sur le vent dépend de la stabilité :**

| Condition | ΔU due à la flottabilité | Couplage nécessaire ? |
|-----------|-------------------------|----------------------|
| **Neutre** (L→∞) | ~0% | Non — simpleFoam suffit |
| **Faiblement stable** (L > 500m) | 5-10% | Marginal |
| **Modérément stable** (L ~ 200m) | 15-30% | **Oui** — suppression turbulence significative |
| **Très stable** (L < 100m) | 30-50%+ | **Oui** — katabatiques, inversions, cold air pooling |
| **Instable** (L < 0) | 10-20% | Oui pour convection, mais RANS est mauvais ici (LES nécessaire) |

Référence : l'étude WES 2024 montre qu'un déficit de vitesse derrière une éolienne varie de 30% (stable) à 41% (instable) → la stratification change significativement le champ de vent.

**Pour le gel/agriculture** : les événements de gel radiatif se produisent en conditions **très stables** (nuit claire, vent faible). C'est exactement le régime où le couplage T↔U est le plus fort. Sans couplage, on ne peut pas capturer le cold air pooling dans les vallées — qui est LE mécanisme principal du gel localisé.

### 2.4 Recommandation révisée

**Partir directement sur buoyantSimpleFoam (compressible) pour tout le dataset.**

Justification :
1. La densité varie trop sur 3 km pour ignorer (26%). Même sur 1.5 km, c'est ~14% — non négligeable.
2. On veut la température de toute façon (agriculture, stabilité). Autant l'avoir dès le début.
3. Le surcoût est ~20-30% par run vs simpleFoam — acceptable sur HPC.
4. Un dataset homogène (même solver partout) est plus simple à gérer qu'un dataset hybride.
5. L'équation d'état ρ = p/(RT) est exacte — pas d'approximation à justifier dans un papier.

**Paramètres de stabilité à varier :**
- Flux de chaleur au sol : -50 W/m² (stable nocturne), 0 (neutre), +200 W/m² (instable diurne)
- Ou gradient de température : +5 K/km (stable), -6.5 K/km (neutre adiabatique), -9 K/km (instable)
- L_mo comme paramètre dérivé pour le conditioning du GNN

**Hauteur de domaine :** 3 km semble un bon compromis. 5 km = ondes de gravité piégées en altitude (intéressant mais RANS les capture mal). 1.5 km = trop bas pour les effets compressibles et les interactions vallée-atmosphère libre.

**Note sur les thermiques :** RANS steady-state ne résout pas les thermiques individuels (phénomène instationnaire turbulent → LES). Mais il capture la **structure moyenne** de la CLA convective : profil de T bien mélangé, couche d'inversion, vent de gradient. C'est suffisant pour prédire le risque de gel (inversion nocturne) et le profil moyen de vent (éolien).

### 2.5 Setup buoyantSimpleFoam (compressible) pour ABL — changements vs simpleFoam

**Note importante** : À Perdigão, Venkatraman et al. (WES 2023) ont testé simpleFoam ET buoyantBoussinesqSimpleFoam, mais **tous les effets thermiques étaient négligés** dans les deux cas. Personne n'a fait de CFD compressible avec température résolue à Perdigão.

#### Champs requis dans `0/`

| Champ | simpleFoam (actuel) | buoyantSimpleFoam (nouveau) | Notes |
|-------|--------------------|-----------------------------|-------|
| `U` | ✓ | ✓ | Même BC (atmBoundaryLayer compatible) |
| `p` | ✓ (dynamique) | ✓ (pression totale) | **Change de sémantique** — p total, pas p dynamique |
| `p_rgh` | — | **NOUVEAU** | p - ρ·g·h (perturbation hydrostatique). C'est p_rgh qui est résolu, pas p |
| `T` | — | **NOUVEAU** | Température [K]. Profil initial : T₀ - Γ·z (gradient adiabatique ou stratifié) |
| `k` | ✓ | ✓ | Identique |
| `epsilon` | ✓ | ✓ | Identique |
| `nut` | ✓ | ✓ | Identique |
| `alphat` | ✓ (template existe) | ✓ | Diffusivité thermique turbulente. Template `alphat.j2` existe déjà |

#### Fichiers `constant/` modifiés

| Fichier | simpleFoam | buoyantSimpleFoam | Changement |
|---------|-----------|-------------------|-----------|
| `transportProperties` | ν seul | **Supprimé** | Remplacé par thermophysicalProperties |
| `thermophysicalProperties` | — | **NOUVEAU** | Type de mélange, eq. d'état, transport |
| `turbulenceProperties` | ✓ | ✓ → `momentumTransport` | Même contenu (OF10 Foundation) |
| `g` | — | **NOUVEAU** | `(0 0 -9.81)` — accélération gravitationnelle |
| `fvOptions` | ✓ | ✓ | Identique (Coriolis, canopée) |

#### `thermophysicalProperties` — contenu type pour ABL

```
thermoType
{
    type            hePsiThermo;       // psi-based (p = rho*R*T)
    mixture         pureMixture;
    transport       const;             // mu, Pr constants
    thermo          hConst;            // cp constant
    equationOfState perfectGas;        // rho = p/(R*T) — EXACT
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie      { molWeight 28.96; }   // air
    thermodynamics { Cp 1005; Hf 0; }
    transport   { mu 1.5e-5; Pr 0.71; }
}
```

#### BC pour `T` (température)

| Face | BC type | Valeur | Notes |
|------|---------|--------|-------|
| **Inlet** | `fixedProfile` ou `codedFixedValue` | T(z) = T₀ - Γ·z + correction MO | Profil log modifié selon stabilité |
| **Outlet** | `inletOutlet` | T ambiante | Permet retour de flux |
| **Terrain** | `fixedValue` ou `fixedFluxTemperature` | T_sol ou q_sol | `fixedFlux` = -50 W/m² (stable), 0 (neutre), +200 (instable) |
| **Top** | `fixedValue` | T(z_top) = T₀ - Γ·z_top | Maintient le gradient |
| **Sides** | Comme inlet/outlet selon la direction du vent | | |

#### BC pour `p_rgh`

| Face | BC | Notes |
|------|-----|-------|
| Inlet | `fixedFluxPressure` | Compatible avec U fixé |
| Outlet | `fixedValue 0` | Référence de pression |
| Terrain | `fixedFluxPressure` | Imperméable |
| Top | `fixedValue 0` ou `fixedFluxPressure` | |

#### Profil initial T(z) pour la stabilité

```
Neutre :     T(z) = 288.15 - 0.0065·z     (gradient adiabatique sec -6.5 K/km)
Stable :     T(z) = 288.15 - 0.0035·z     (inversion partielle, -3.5 K/km → sub-adiabatique)
Instable :   T(z) = 288.15 - 0.0098·z     (super-adiabatique, -9.8 K/km → conditions convectives)
```

#### Coût supplémentaire estimé

- buoyantSimpleFoam résout 1 équation de plus (énergie) → ~20-30% plus lent que simpleFoam
- La convergence peut être plus délicate en stabilité forte (couplage T↔U via flottabilité)
- Avec k-ε Parente, le couplage est indirect (via ρ dans l'équation de quantité de mouvement)
- Estimation : 8-20 min/run sur 8 cores (vs 5-15 min pour simpleFoam)

---

## 3. Notre positionnement — différenciation

| Aspect | WindSeer | DEVINE | Meteodyn | **Notre projet** |
|--------|----------|--------|----------|-----------------|
| Dataset CFD | 7 361 runs | 7 279 (ARPS) | Privé | **50 000 runs** |
| Terrains | Suisse | Gaussien synth. | Variable | **SRTM mondial** |
| Turbulence | k-ε standard | ARPS | Propriétaire | **k-ε Parente** |
| Source terms | Aucun | Aucun | ? | **Coriolis + canopée** |
| Compressibilité | **Non** (incompressible) | Non | ? | **Oui** (gaz parfait ρ=p/RT) |
| Stratification | Non | Non | Oui | **Oui** (flux sol + gradient T) |
| Température | **Non** | Non | ? | **Oui** (couplée, buoyantSimpleFoam) |
| Land cover | Roughness seule | Aucun | Variable | **WorldCover + ETH Canopy** |
| Architecture DL | U-Net 3D grille rég. | CNN grille rég. | N/A | **GNN maillage non structuré** |
| Sorties | U + TKE | U | U | **U + k + ε + T + ρ** |
| Poids publiés | Non | Non | Non | **Oui (prévu)** |
| Licence | BSD-3 (code) | ? | Commercial | **Open source + API** |
| Inférence | 21 ms (Jetson) | ~100 ms | Minutes-heures | **Cible < 1s GPU** |
| Applications | UAV en vol | NWP→30m | Éolien | **Multi (éolien, agri, qualité air)** |

---

## 4. Applications B2B et marchés cibles

| Application | Besoin micro-échelle | Variable clé | Concurrent | Notre avantage |
|-------------|---------------------|-------------|------------|----------------|
| **Agriculture / gel** | Carte T_min par parcelle, risque gel J+1→J+4 | **T** + vent | CNNs station (MAE ~2°C) | Physique 3D (cold air pooling) |
| **Éolien micro-siting** | AEP, TI, cisaillement au mât | **U, k, ε** | Meteodyn (1500€), WindSim | 100× plus rapide, 10× moins cher |
| **Qualité de l'air** | Dispersion polluants en vallée/ville | **U** + stabilité | Quasi-inexistant en ML | Champ 3D + stratification |
| **Météo locale / outdoor** | Prévision vent/T à 100m, temps réel | U, T | Apps météo (NWP brut) | Résolution micro |
| **Drones / UAV** | Vent 3D temps réel basse altitude | **U 3D** | WindSeer | Dataset + physique supérieurs |

---

## 5. Plan d'exécution

### Phase 1 — Validation pipeline CFD compressible (en cours → 1 mois)
- **Objectif** : 1 run buoyantSimpleFoam convergé avec maillage adaptatif rapide (~50k cellules) sur Perdigão
- **Changement clé vs plan précédent** : passer de simpleFoam à buoyantSimpleFoam dès maintenant
- **Fichiers à adapter** : `generate_mesh.py` (ok), `openfoam_runner.py` (changer solver), `prepare_inflow.py` (ajouter profil T), templates OpenFOAM (ajouter T, p_rgh, alphat, transportProperties→physicalProperties)
- **Templates à créer** : `0/T.j2`, `0/p_rgh.j2`, `constant/physicalProperties.j2` (avec β, Pr_t)
- **Critère** : convergence résidus < 1e-4, vitesses aux mâts ~O(10 m/s), profil T réaliste (~-6.5K/km en neutre)
- **Statut** : évolution de l'étape 8 du plan `buzzing-splashing-phoenix.md`

### Phase 2 — Pipeline de génération de terrains mondiaux (2 mois)
- Script auto : coordonnées aléatoires → SRTM download → STL → WorldCover + ETH Canopy → z₀/LAD raster
- Sélection de ~2 000 patches terrain (diversité TPI, continents, types de terrain)
- Taille patches : 3×3 km (compromis contexte vs coût)
- Validation : inspection visuelle d'un échantillon, check statistiques de diversité

### Phase 3 — Dataset à grande échelle (3-4 mois, parallélisable HPC)
- 25 runs/patch (8 dir × 3 vit + 1 condition thermique variable) = ~50k runs
- **Solver** : buoyantSimpleFoam + k-ε Parente + Coriolis + canopée
- **Domaine** : 3 km de hauteur, maillage adaptatif 30-80k cellules
- **Variables de sortie** : U, k, ε, T, p, ρ sur tous les cell centres
- Conditions thermiques variées : neutre (gradient adiabatique), stable (inversion), instable (flux positif)
- Post-traitement → Zarr/HDF5 + construction graphe (cellules→nœuds, faces→arêtes)
- **Coût estimé** : 150-400k CPU-hours (buoyantSimpleFoam ~30% plus lent que simpleFoam)

### Phase 4 — Architecture GNN et entraînement (2-3 mois)
- MeshGraphNets (PyTorch Geometric) sur maillage non structuré
- Conditioning global : direction, vitesse, flux chaleur sol (ou L_mo) via FiLM
- Node features : position, distance terrain, z₀, LAD, altitude, pression hydrostatique
- Loss : MSE pondérée sur (U, k, ε, T) + continuité (∇·(ρU) ≈ 0) optionnel
- Sorties : **Ux, Uy, Uz, k, ε, T** (6 canaux)
- Baselines : U-Net 3D (à la WindSeer), profil log+adiabatique, interpolation

### Phase 5 — Validation multi-sites (1-2 mois)
- Perdigão (48 mâts, U + T), Bolund, Askervein
- 1-2 wind farms si données accessibles
- Vallée alpine pour validation gel (données Trento si disponibles)
- Métriques : MAE, RMSE, profils verticaux (U et T), diagrammes de Taylor
- **Cible** : MAE vitesse < 1.5 m/s sur Perdigão (vs 1.87 WindSeer), MAE T < 1.5°C

### Phase 6 — Produit et API
- Pipeline inférence : GPS + rayon + conditions NWP/ERA5 en altitude → auto-download DEM/landcover → graphe → GNN → champ 3D
- Entrée NWP possible via : ERA5 brut, IFS Open-Meteo, ou Prithvi WxC fine-tuné (si gain démontré)
- FastAPI + GPU cloud
- Sorties : GeoTIFF 2D (surface), NetCDF/Zarr 3D, GeoJSON points, rapport PDF (éolien)

---

## 6. Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|-----------|
| Taux convergence CFD sur terrains extrêmes | Perte de runs (~10%) | Critère relaxé, filtrage, fallback mesh plus grossier |
| Scalabilité GNN >100k cellules | Limite résolution | X-MeshGraphNet, partitionnement graphe |
| Maillage adaptatif rapide = moins précis | Erreur de discrétisation | Valider sur quelques cas haute-résolution, quantifier l'erreur |
| simpleFoam isotherme insuffisant pour gel | Application agriculture bloquée | buoyantBoussinesq en phase 5 |
| NVIDIA/Google descendent à la micro-échelle | Concurrence soudaine | Publier vite, dataset ouvert comme moat, niche applications |
| Généralisation GNN hors distribution | Erreurs sur terrains très différents | Diversité dataset maximale, TPI sampling, transfer learning |

---

## 7. Décisions ouvertes à trancher

- **D-NEW-1** : Taille des patches terrain — 3×3 km (comme WindSeer++) ou 5×5 km (plus de contexte mais plus cher) ?
- **D-NEW-2** : Nombre de directions par patch — 8 (45° pas) ou 16 (22.5° pas) ?
- **D-NEW-3** : Architecture GNN — MeshGraphNets pur ou hybride avec neural operator (FNO) pour les grandes échelles ?
- **D-NEW-4** : Budget HPC disponible → détermine le compromis patches × runs/patch × résolution maillage
- **D-NEW-5** : Publication — un seul gros papier (Nature Comms) ou deux papiers (1: dataset + méthode CFD, 2: GNN + validation) ?

---

## 8. Références clés

### CFD ABL terrain complexe
- Venkatraman et al. (WES 2023) — Source terms at Perdigão, [DOI](https://doi.org/10.5194/wes-8-85-2023)
- Parente et al. (2011) — Modified k-ε for consistent ABL
- Palma et al. (WES 2020) — Mesh resolution threshold 40m

### Surrogates DL pour le vent
- WindSeer — [Nature Comms 2024](https://www.nature.com/articles/s41467-024-47778-4), [GitHub](https://github.com/ethz-asl/WindSeer)
- DEVINE — [AMS AIES 2023](https://journals.ametsoc.org/view/journals/aies/2/1/AIES-D-22-0034.1.xml)
- TerraWind — [GRL 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL112124)
- Le Toumelin et al. — [NPG 2024](https://npg.copernicus.org/articles/31/75/2024/)

### GNN pour CFD
- MeshGraphNets — [arXiv 2010.03409](https://arxiv.org/abs/2010.03409)
- X-MeshGraphNet — [arXiv 2411.17164](https://arxiv.org/html/2411.17164v2)
- SGUNET (Apple) — [arXiv 2502.06848](https://arxiv.org/abs/2502.06848)
- PIGNN-CFD — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360132323000835)

### Foundation models météo
- CorrDiff — [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)
- Prithvi WxC — [HuggingFace IBM/NASA](https://newsroom.ibm.com/2024-09-23-ibm-and-nasa-release-open-source-ai-model-on-hugging-face-for-weather-and-climate-applications)

### Température / gel / stratification
- CFD stratified ABL + MOST — [ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0360132324011260)
- Mountain-valley thermal winds (OpenFOAM) — [MDPI 2023](https://www.mdpi.com/2071-1050/15/2/1387)
- Frost downscaling CNN (Trento) — [Atmosphere 2025](https://www.mdpi.com/2073-4433/16/1/38)

### Métriques de performance surrogates CFD
- MAPE typique : 1.7%–15% selon complexité
- Speedup vs RANS/LES : 10×–280×
- Inférence : millisecondes à secondes

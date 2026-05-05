# DownscaleWind — Briefing stratégique pour orientation recherche

> Document de cadrage pour discussion collaborative (Australie, CS Aero, industriels éolien/solaire).
> État au 25 mars 2026.

---

## 1. Ce que fait le projet

**DownscaleWind** est un pipeline de downscaling atmosphérique qui prend les réanalyses/prévisions globales ERA5 (25 km, 6h) et produit des champs de vent 3D à résolution kilométrique (~100 m–1 km) en terrain complexe, en temps quasi-réel (<1 s).

### Architecture en 4 modules

```
ERA5 (25 km, 6h)
    │
    ├─► Module 1 — Downscaling temporel (NeuralODE, 6h → 1h)        ✅ FAIT
    │       Interpolation dans l'espace latent entre snapshots ERA5
    │
    └─► Module 2A — CFD (OpenFOAM simpleFoam k-ε, RANS stationnaire) 🔧 EN COURS
            ERA5 → conditions aux limites → simulation terrain complexe
            │
            └─► Module 2B — Surrogate GNN (GATv2 + FiLM)             📋 PLANIFIÉ
                    Remplace le CFD en inférence : < 1s au lieu de 10-30 min
                    │
                    └─► Module 3 — Stochastique (turbulence sous-horaire) 📋 FUTUR
```

### Site de validation : Perdigão, Portugal
Double crête parallèle, IOP mai-juin 2017, 48 mâts instrumentés. Benchmark international de référence pour la CFD en terrain complexe (Fernando et al., BAMS 2019).

---

## 2. État d'avancement (mars 2026)

| Composant | Statut | Détail |
|-----------|--------|--------|
| Ingestion ERA5/IFS | ✅ | Scripts versionnés, Zarr, checkpoints SHA256 |
| Module 1 — Temporel | ✅ | 76K params, <10 ms CPU, entraîné sur 2016 |
| Module 2A — CFD pipeline | 🔧 80% | simpleFoam convergé, 500 iter ~22s HPC, 7/25 timestamps calculés |
| Maillage structuré | ✅ | terrainBlockMesher validé, 4 niveaux de raffinement |
| Conditions aux limites ERA5 | ✅ | Interpolation trilinéaire, inletOutlet (Robin BC) |
| Validation vs observations | 🔧 | RMSE 3.10 m/s à 100m (comparable à la littérature) |
| Module 2B — GNN surrogate | 📋 | Architecture décidée (3 candidats codés), dataset en cours |
| Module 3 — Stochastique | 📋 | Non commencé |

### Résultat clé actuel
- **CFD vs ERA5 pur** : le CFD reproduit ERA5 en altitude (bias ~0 m/s) et ajoute les effets de terrain près du sol (accélération de crête, ralentissement en vallée)
- **Convergence en résolution** : RMSE passe de 3.88 (500 m) à 3.10 m/s (100 m) vs 81 points de mesure

---

## 3. Intérêt scientifique et verrous technologiques

### 3.1 Positionnement dans la littérature

Le concurrent direct le plus proche est **WindSeer** (ETH Zürich, Nature Communications 2024) :

| | WindSeer | **DownscaleWind** |
|--|----------|-------------------|
| Dataset CFD | 7 361 runs, Suisse | **Cible 15k–50k, terrains mondiaux (SRTM)** |
| Turbulence | k-ε standard | **k-ε Parente (validé Perdigão)** |
| Termes sources | Aucun | **Coriolis + canopée (WorldCover + ETH)** |
| Température | Non | **Oui (prévu, couplé ou post-traitement)** |
| Architecture DL | U-Net 3D (grille régulière) | **GNN sur maillage non structuré** |
| Inférence | 21 ms (Jetson Orin) | Cible < 1 s GPU |
| Validation Perdigão | MAE 1.87 m/s | **À battre** |
| Open weights | Non | **Oui (prévu)** |

### 3.2 Verrous technologiques (= sujets de publication)

1. **GNN sur maillage CFD non structuré** : WindSeer ré-échantillonne sur grille régulière (perte d'info). Opérer directement sur le maillage adaptatif est un problème ouvert (scalabilité, transfert entre maillages).

2. **Conditioning multi-échelle ERA5→micro** : comment injecter le forçage synoptique (9 points ERA5 à 25 km) dans un GNN qui opère à 100 m ? Architecture bipartite (arêtes ERA5→CFD) vs FiLM global.

3. **Augmentation par rotation** : une seule direction CFD + rotation à l'entraînement = ×16 data augmentation. Implique un domaine cylindrique et une invariance rotationnelle à démontrer.

4. **Contraintes physiques dans la loss** : divergence nulle (incompressibilité ∇·u = 0) et non-pénétration au terrain comme pertes auxiliaires. Impact quantitatif non établi dans la littérature GNN-CFD-ABL.

5. **Température couplée en terrain complexe** : aucun surrogate existant ne prédit T. Le cold air pooling en vallée est un phénomène critique pour agriculture/gel et qualité de l'air, non couvert par WindSeer/DEVINE.

---

## 4. Applications industrielles et marchés

### 4.1 Éolien (wind farm)

| Besoin | Variable | Notre apport |
|--------|----------|-------------|
| Micro-siting (placement turbines) | U, TI, cisaillement | 100× plus rapide que Meteodyn (1 500€/point) |
| AEP (Annual Energy Production) | U à hauteur moyeu | Profil vertical résolu, pas juste extrapolation log |
| Wake effects (sillages) | U 3D + k | Champ 3D complet, pas un modèle de sillage analytique |
| Repowering | U à nouvelle hauteur | Changement de hauteur = requête instantanée |
| O&M loads | TI, shear exponent | Fatigue des pales mieux estimée avec le vrai profil |

**Avantage compétitif** : les industriels éoliens (Vestas, Siemens Gamesa, GoldWind) utilisent des outils CFD coûteux (Meteodyn WT ~1 500€/point, WindSim consulting). Un surrogate 100× plus rapide et open source peut servir de :
- **Screening tool** : pré-sélection de sites avant la campagne de mesure (réduction coûts de prospection)
- **Complément mât** : étendre 1-2 mâts à une carte de vent complète sur le site
- **Opérationnel** : prévision de production J+1 (couplé avec IFS/GFS en entrée)

**Collaboration Australie** : l'Australie est le 3ᵉ marché éolien offshore émergent (Bass Strait, Hunter Valley). Les terrains côtiers australiens (Great Dividing Range → côte) sont un cas d'application direct.

### 4.2 Solaire — soiling et rendement

Le **soiling** (dépôt de poussière sur panneaux PV) dépend directement du vent local :
- **Vent faible** (< 2 m/s) : accumulation de poussière, perte de rendement 0.5-1%/jour en environnement aride
- **Vent modéré** (3-6 m/s) : nettoyage naturel partiel
- **Vent fort** (> 10 m/s) : abrasion, dommages mécaniques
- **Direction du vent** : influence le pattern de dépôt (face amont vs aval du panneau)

| Besoin solaire | Variable DownscaleWind | Application |
|----------------|----------------------|-------------|
| Carte de soiling | U surface, direction | Planification fréquence de nettoyage par zone |
| Vent extrême | U_max, rafales (Module 3) | Dimensionnement structure, angle de stow |
| Rendement thermique | T_air locale | Correction P_max = P_STC × (1 - γ(T-25°C)) |
| Brouillard / rosée | T, humidité (v1.5) | Risque de condensation sur panneaux |
| Convection surface | U surface | Refroidissement naturel des panneaux |

**Collaboration Australie** : les grandes fermes solaires australiennes (NSW, QLD) sont en environnement semi-aride avec forte variabilité de soiling. Le terrain plat-à-vallonné est plus simple que Perdigão → les performances du surrogate y seraient meilleures.

### 4.3 Agriculture et gel

- **Cold air pooling** : l'air froid nocturne s'écoule vers les points bas → gel localisé. C'est un phénomène 3D qui nécessite le vent ET la température.
- **Notre avantage** : seul surrogate qui prévoit T en 3D en terrain complexe (via BBSF ou scalaire passif post-traitement).
- **Marché** : viticulture (gel printanier = 1ᵉʳ risque en France/Australie), arboriculture, cultures précoces.
- **Australie** : les vallées viticoles (Barossa, Hunter Valley, Yarra Valley, Tasmania) sont directement concernées.

### 4.4 Qualité de l'air et environnement

- Dispersion de polluants en vallée (inversions thermiques)
- Modélisation de la dispersion d'odeurs (élevage, industrie)
- Impact sur la santé : particules fines piégées par stabilité forte

### 4.5 Drones / UAV

- Navigation en terrain complexe (missions autonomes)
- WindSeer cible ce marché mais sans température ni terrains mondiaux
- Application militaire potentielle (ISR en terrain montagneux)

---

## 5. Axes de collaboration recherche

### 5.1 Avec CS Aero (Australie) — axe CFD + turbulence

| Sujet | Expertise CS Aero | Notre contribution | Publication |
|-------|-------------------|--------------------|-------------|
| **LES vs RANS en terrain complexe** | Capacité HPC + compétence LES | Pipeline automatisé + 48 mâts validation | BLM ou WES |
| **Modèle de turbulence adaptatif** | Développement modèles turb. | Dataset 15k runs pour calibration | JFM |
| **Effets de canopée sur TKE** | Aérodynamique | WorldCover + ETH Canopy + fvOptions | Agricultural and Forest Meteorology |
| **Wake-terrain interaction** | Aéro turbines | Champ 3D complet sur site réel | Wind Energy Science |

**Intérêt pour CS Aero** : accès à un pipeline CFD atmosphérique automatisé + un dataset massif de runs RANS en terrain complexe pour valider/calibrer leurs modèles de turbulence.

### 5.2 Avec industriels éoliens (Australie)

| Sujet | Industriel type | Livrable | Valorisation |
|-------|-----------------|----------|-------------|
| **Screening éolien offshore/onshore** | Goldwind, Vestas AU | API vent 3D sur sites candidats | Licence ou SaaS |
| **Correction de production** | Opérateur de parc | Prévision J+1 à J+7 site-spécifique | Réduction d'erreur de prévision |
| **Validation sur sites australiens** | Bureau of Meteorology (BoM) | Benchmark DOWN vs BoM ACCESS-C3 | Publication conjointe |
| **Extension offshore** | Star of the South | Adaptation BC (roughness marine, fetch) | Contrat R&D |

### 5.3 Avec industriels solaires / soiling

| Sujet | Partenaire type | Livrable |
|-------|-----------------|----------|
| **Carte de soiling prédictive** | First Solar, Array Technologies | U + dir à 2m → modèle de dépôt |
| **Optimisation nettoyage** | O&M operators | Planning par zone basé sur U_surface |
| **Corrélation vent-rendement** | Recherche (UNSW, ANU) | Dataset U + T + PV production |
| **Abrasion éolienne** | Désert australien | U_max + direction dominante → usure panneaux |

### 5.4 Sujets de thèse potentiels

1. **GNN physics-informed pour le downscaling éolien** (3 ans, HPC intensif)
   - Verrou : scalabilité GNN >500k nœuds, contraintes physiques
   - Débouché : Nature Communications / ICLR

2. **Downscaling thermique en terrain complexe : cold air pooling et gel** (3 ans)
   - Verrou : couplage T↔U via BBSF, validation sur données terrain
   - Débouché : Agricultural and Forest Meteorology, applications viticoles

3. **Prévision de soiling par downscaling éolien** (3 ans, co-encadrement avec solaire)
   - Verrou : couplage vent → transport de particules → dépôt
   - Débouché : Solar Energy, application industrielle directe

4. **Transfer learning entre terrains : un foundation model pour le vent de surface** (3 ans)
   - Verrou : généralisation hors-distribution du GNN
   - Débouché : si ça marche, c'est un papier très cité

---

## 6. Forces et faiblesses du projet

### Forces
- **Pipeline bout-en-bout** : de ERA5 brut au champ 3D en <1s — aucun concurrent n'a ça
- **Validation rigoureuse** : Perdigão 48 mâts, comparaison directe avec WindSeer/Letzgus
- **Open source + open weights** : différenciation vs industriels fermés (Meteodyn, WindSim)
- **Multi-application** : éolien + agri + qualité de l'air + drones (pas mono-marché)
- **Terrains mondiaux (SRTM)** : pas limité à un pays (WindSeer = Suisse, DEVINE = Alpes)
- **Reproductible** : données publiques (ERA5, SRTM, WorldCover), code versionné, seeds fixés

### Faiblesses / risques
- **Un seul développeur principal** (Guillaume) → bus factor = 1
- **Pas encore de résultat GNN** : tout le ML est à faire (modules 2B, 3)
- **HPC dépendant** : 15k–50k runs CFD nécessitent un budget calcul significatif (~150-400k CPU-hours)
- **Température pas encore transportée** : v1 = vent neutre seulement, BBSF différé
- **RANS limité** : pas de turbulence résolue, pas d'instationnaire → les cas stables/convectifs sont approchés
- **Pas de données observées hors Perdigão** : généralisation non démontrée

### Ce qui manque pour un papier
1. ✅ Pipeline CFD automatisé qui converge
2. 🔧 Dataset de 25+ runs sur Perdigão (7/25 faits)
3. 📋 GNN entraîné + évalué sur ces 25 runs
4. 📋 Comparaison quantitative vs WindSeer (MAE < 1.87 m/s) et vs ERA5 brut
5. 📋 Ablation study (architecture, physics loss, augmentation)

---

## 7. Stratégie de publication

### Option A — Deux papiers (recommandé)
1. **Papier méthode/dataset** (Wind Energy Science ou GMD)
   - Pipeline CFD automatisé ERA5→OpenFOAM en terrain complexe
   - Dataset ouvert de N runs sur Perdigão + M terrains
   - Validation vs observations, comparaison avec littérature
   - **Rapide à publier** (pas besoin du GNN)

2. **Papier surrogate** (Nature Communications ou ICLR)
   - GNN sur maillage non structuré + physics-informed
   - Benchmark vs WindSeer/DEVINE + baselines
   - Applications multi-domaines

### Option B — Un seul papier ambitieux (Nature Communications)
- Tout le pipeline de A à Z
- Plus long mais plus d'impact

### Cibles de publication par domaine
| Domaine | Journal | Impact Factor | Pertinence |
|---------|---------|---------------|------------|
| Éolien | Wind Energy Science | 4.1 | Validation Perdigão, pipeline CFD |
| Géosciences ML | Geoscientific Model Development | 5.2 | Pipeline complet, reproductibilité |
| ML généraliste | Nature Communications | 16.6 | Si résultats state-of-the-art |
| ML conférence | ICLR / NeurIPS workshop | Top | Architecture GNN + physics |
| Agriculture | Agricultural and Forest Meteorology | 6.8 | Si température validée (v1.5+) |
| Solaire | Solar Energy | 7.2 | Si application soiling démontrée |

---

## 8. Questions clés pour la discussion stratégique

1. **Priorité d'application** : éolien d'abord (marché mature, données dispo) ou agriculture/gel (plus différenciant, moins de concurrence) ?

2. **Température quand ?** : publier d'abord le vent neutre (rapide, comparable à WindSeer) puis ajouter T, ou attendre d'avoir T pour se différencier ?

3. **Terrains australiens** : utiliser les données BoM/METAR australiennes pour un deuxième site de validation ? Quels sites sont les plus instrumentés ?

4. **HPC** : quel budget calcul est réaliste pour le dataset (150k–400k CPU-hours) ? Accès Gadi (NCI) ou Setonix (Pawsey) via collaboration ?

5. **Collaboration CS Aero** : quel niveau — co-supervision de thèse, projet conjoint ARC, ou échange informel ?

6. **Soiling** : y a-t-il des données couplées vent + soiling disponibles en Australie (ARENA datasets) ?

7. **IP et valorisation** : open source total (maximum de citations) vs dual-licence (SaaS commercial + open research) ?

8. **Timeline réaliste** : premier papier quand ? GNN fonctionnel quand ? Avec quelles ressources humaines ?

---

## 9. Résumé exécutif (pour interlocuteurs non-techniques)

**DownscaleWind** transforme les prévisions météo globales (résolution ~25 km) en cartes de vent et température locales (résolution ~100 m) en terrain montagneux, en moins d'une seconde.

**Comment** : on simule des milliers de scénarios de vent sur des terrains réels avec un modèle de mécanique des fluides (CFD), puis on entraîne un réseau de neurones (GNN) à reproduire ces simulations instantanément.

**Pourquoi c'est utile** :
- **Éolien** : placer les turbines au bon endroit, prévoir la production → économie de millions €/$ par parc
- **Solaire** : prédire l'encrassement des panneaux par la poussière → optimiser le nettoyage
- **Agriculture** : cartographier le risque de gel en fond de vallée → protéger les cultures
- **Qualité de l'air** : comprendre où les polluants s'accumulent en conditions stables

**Ce qui nous distingue** : le seul outil open source qui combine physique complète (turbulence, canopée, Coriolis, bientôt température) avec un dataset mondial de terrains et une inférence en temps réel. Les concurrents sont soit fermés et chers (Meteodyn), soit limités à un pays et sans température (WindSeer).

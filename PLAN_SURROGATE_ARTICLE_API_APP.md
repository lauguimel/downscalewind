# Plan — Surrogate, article, API vitrine et application meteo

Generated: 2026-04-27

## Objectif global

Construire une trajectoire coherente a partir de l'etat actuel du projet :

1. **Surrogate v2** : produire un modele robuste et exploitable sur la campagne v2.
2. **Article vent + T + RH + precipitations** : publier une vitrine scientifique honnete.
3. **API show-off** : exposer gratuitement une version limitee pour attirer des contacts.
4. **Integration app meteo** : brancher le downscaling dans l'application deja en cours.

Principe directeur : le surrogate est le socle. L'article, l'API et l'app ne doivent
pas avancer avec des resultats non filtres ou non versionnes.

---

## Etat de depart

### Actifs disponibles

- Campagne v2 `complex_terrain_v1` : environ 9k solves exploitables, multi-sites, vent + T + q.
- Pipeline HPC fonctionnel en deux phases : mesh puis solve.
- Prototype API `plume-api` avec FNO3D/TorchScript et endpoint `/v1/forecast`.
- Scripts de validation ICOS/FWI, comparaison ERA5/CFD, correction precipitations IMERG_QM.
- Dossier valorisation deja structure dans `valo/`.

### Limites connues

- Les partiels et early-converged doivent etre recuperes proprement.
- Les cas SIGFPE terrain raide doivent etre exclus ou traites comme chantier cfMesh separe.
- La convergence OpenFOAM ne suffit pas : il faut filtrer sur coherence physique.
- Le site Puéchabon n'est pas une bonne vitrine pour le gain vent : site plat, convection thermique, canopee.
- L'API FWI actuelle est une demonstration instantanee, pas un FWI operationnel journalier.
- Les README et docs doivent etre realignes : FNO3D deja present, GNN encore option recherche.

---

## Phase 0 — Nettoyage et gel du dataset v2

**Objectif :** transformer la campagne v2 en dataset d'entrainement fiable.

### Actions

- Relancer les 80 sites partiels avec le job phase 2.
- Modifier la collecte/export pour accepter le dernier timestep ecrit au lieu de requerir uniquement `300/U`.
- Recuperer les 16 early-converged.
- Tagger les 158 SIGFPE comme `terrain_steep_mesher_failed`.
- Produire une table de statut par site et par solve :
  - `solved`;
  - `partial`;
  - `early_converged`;
  - `diverged`;
  - `mesh_failed`;
  - `rejected_by_qa`.
- Definir trois niveaux qualite :
  - `gold` : utilisable article + surrogate ;
  - `silver` : utilisable training avec poids faible ou ablation ;
  - `rejected` : non utilise.

### QA minimale

- Bornes physiques :
  - `U_max < 50 m/s`, sauf justification orographique ;
  - `T` dans une plage meteorologique plausible ;
  - `q >= 0`.
- Coherence CFD/inflow :
  - erreur profil amont acceptable ;
  - direction top-domain coherente ;
  - pas de speed-up aberrant non explique ;
  - residuals U acceptables.
- Split strict par site :
  - jamais le meme site en train et test ;
  - split geographique documente.

### Livrables

- `dataset_v2_manifest.yaml`
- `dataset_v2_status.csv`
- `dataset_v2_qa_summary.md`
- `dataset_v2_splits.yaml`

### Critere de sortie

Le dataset peut etre utilise par une commande unique d'entrainement et une commande
unique d'evaluation, avec nombre de cas `gold/silver/rejected` clairement trace.

---

## Phase 1 — Surrogate v2

**Objectif :** obtenir un modele exploitable pour l'article, l'API vitrine et l'app.

### Choix technique court terme

Priorite : **FNO3D v2**.

Raison : l'API actuelle est deja construite autour d'un modele FNO3D TorchScript.
Le GNN reste pertinent scientifiquement, mais il ne doit pas bloquer la vitrine.

### Entrees modele

- Terrain.
- Rugosite `z0`.
- ERA5 profile 1D : `u`, `v`, `T`, `q`, `k`.
- Si disponible : ERA5 3x3 ou gradients horizontaux.
- Surface anchors : `u10`, `v10`, `t2m`, `d2m`.

### Sorties modele

- `u`, `v`, `w`.
- `T`.
- `q`, puis RH derivee.
- Optionnel : `k`, `epsilon`, `nut` pour diagnostic et article.

### Baselines obligatoires

- ERA5 brut/interpole.
- CFD teacher.
- Surrogate v1 ou FNO3D 9k si disponible.
- Ablation sans surface anchors.
- Ablation 1D ERA5 vs 3D/gradients ERA5.

### Metriques

- RMSE/MAE de vitesse par hauteur : 10, 60, 100, 150 m.
- RMSE/MAE `u`, `v`, direction.
- RMSE/MAE `T` et RH.
- Skill vs ERA5.
- Skill par famille de site :
  - fire terrain ;
  - mountain ;
  - wind onshore ;
  - morpho ;
  - paragliding si conserve.
- Taux de cas ou le modele degrade ERA5 : metrique `do_no_harm`.
- Latence CPU et GPU.
- Tests de robustesse hors domaine.

### Livrables

- `models/surrogate_v2_fno3d/`
- `surrogate_v2_metrics.json`
- `surrogate_v2_case_metrics.csv`
- `surrogate_v2_eval_report.md`
- `surrogate_v2.ts.pt`

### Critere de sortie

- Le surrogate bat ERA5 sur les terrains complexes.
- Les erreurs restent bornees sur les sites plats.
- Les champs sont physiquement plausibles.
- La latence est compatible avec une API gratuite limitee.

---

## Phase 2a — Article downscaling vent + T + RH + precipitations

**Objectif :** produire une vitrine scientifique solide sans exposer tout l'actif proprietaire.

### Message scientifique

Le papier ne doit pas dire : "nous faisons un FWI local parfaitement fiable".

Message recommande :

> Un pipeline hybride ERA5 + CFD/surrogate + correction precipitation ameliore la
> micro-meteorologie en terrain complexe. La correction pluie explique une partie
> majeure du gain FWI, et le downscaling vent ajoute de la valeur sur les sites
> ou le relief modifie vraiment l'ecoulement.

### Structure proposee

1. Introduction :
   - limites ERA5/CEMS en terrain complexe ;
   - besoin local vent, T, RH, precipitation pour feu et micro-meteo.
2. Methode :
   - ERA5/IFS ;
   - CFD teacher ;
   - surrogate ;
   - correction precipitation IMERG_QM ;
   - calcul FWI.
3. Dataset :
   - campagne v2 ;
   - familles de sites ;
   - splits ;
   - QA.
4. Validation meteorologique :
   - vent vs ICOS/Perdigao ;
   - T/RH ;
   - limites thermiques.
5. Validation FWI :
   - Puéchabon comme cas ablation pluie, pas comme preuve vent ;
   - FR-OHP / ES-LJu / autre site relief comme cas terrain complexe ;
   - comparaison ERA5, ERA5+IMERG_QM, CFD/surrogate+IMERG_QM.
6. Discussion :
   - quand le CFD aide ;
   - quand il n'aide pas ;
   - thermique/canopee comme etape suivante ;
   - limites RANS neutre.

### Figures cles

- Carte d'un site relief montrant acceleration vent et heterogeneite FWI.
- Ablation Puéchabon montrant que le gain vient surtout de la pluie.
- Skill vent par categorie de terrain.
- Skill FWI sur sites terrain complexe.
- Schema pipeline ERA5 -> CFD -> surrogate -> indice metier.

### Politique open data

- Publier code de calcul FWI et evaluation.
- Publier quelques cas demo reproductibles.
- Ne pas publier tout le dataset CFD ni les poids haute resolution si l'objectif est valorisation.
- Decrire clairement ce qui est ouvert et ce qui reste proprietaire.

### Critere de sortie

Le papier peut soutenir une demande BPI/i-Lab et attirer des prospects sans
surpromettre la performance operationnelle feu.

---

## Phase 2b — API show-off gratuite

**Objectif :** faire venir des contacts qualifiés, pas generer directement du revenu.

### Positionnement public

Ne pas dire : "modele degrade expres".

Dire :

> Demonstration gratuite basee sur un modele generique basse resolution. Elle
> sert a explorer les effets du relief. Les etudes dediees utilisent une
> resolution, une physique et une validation adaptees au site.

### Fonctionnalites gratuites

- Point forecast :
  - vitesse vent ;
  - direction ;
  - temperature indicative ;
  - RH indicative.
- Hauteurs fixes :
  - 60 m ;
  - 100 m ;
  - 150 m.
- Carte ou volume 3D de demonstration.
- Quelques sites vitrines pre-calcules.
- Badge clair : `experimental`, `not safety critical`.

### Ne pas fournir gratuitement

- Haute resolution pres du sol.
- Thermique avancee.
- Inversions / cold-air pooling.
- Canopee explicite.
- Export GeoTIFF, NetCDF, Zarr.
- Batch sur polygone.
- Historique long.
- SLA.
- Validation locale.
- Modele prive.

### Call-to-action

- "Demander une etude de site".
- "Envoyer des observations pour benchmark".
- "Discuter d'un cas eolien, feu, dispersion ou agriculture".

### Contrat API minimal

```http
POST /v1/forecast
```

Entree :

```json
{
  "latitude": 43.0,
  "longitude": 5.0,
  "timestamp": "2026-06-15T12:00:00Z",
  "height_m": 100,
  "variables": ["wind", "temperature", "rh"]
}
```

Sortie :

```json
{
  "wind_speed_ms": 6.2,
  "wind_direction_deg": 315,
  "temperature_K": 298.4,
  "relative_humidity_pct": 35.0,
  "model_version": "surrogate-v2-demo",
  "quality_flags": {
    "experimental": true,
    "terrain_complexity": "moderate",
    "thermal_regime_supported": false,
    "outside_training_domain": false
  }
}
```

### Critere de sortie

L'API doit donner envie de contacter, pas fournir une meteo operationnelle complete.

---

## Phase 2c — Integration dans l'application meteo

**Objectif :** brancher le downscaling dans l'app sans brouiller la difference entre meteo officielle et couche experimentale.

### Ordre d'integration

1. Brancher l'API sur une zone test.
2. Ajouter cache local.
3. Ajouter fallback ERA5/Open-Meteo si l'API est indisponible.
4. Afficher un badge `experimental terrain downscaling`.
5. Ajouter les flags de qualite.
6. Ajouter une page "demander une etude detaillee".

### Ce que l'app doit afficher

- Vent downscale a 60/100/150 m.
- Difference vs forecast standard.
- Indicateur terrain :
  - plat ;
  - colline ;
  - relief complexe ;
  - hors domaine.
- Mention claire :
  - pas d'alerte officielle ;
  - pas de decision securite ;
  - couche experimentale.

### Ce que l'app ne doit pas afficher au debut

- FWI operationnel complet.
- Alertes feu.
- Alertes gel.
- Promesses sur thermique/inversion.
- Scores de precision non valides.

### Critere de sortie

L'app montre la valeur visuelle et exploratoire du downscaling, tout en envoyant
les cas serieux vers une demande d'etude dediee.

---

## Trajectoire valorisation associee

### Role de chaque brique

- Surrogate : actif technique.
- Article : preuve scientifique.
- API : vitrine publique et generation de leads.
- App meteo : usage visible et recurrent.
- Etudes payantes : revenu initial.
- LBM/LES GPU : ambition i-Lab et montee en gamme.

### Offres payantes possibles

| Offre | Prix indicatif | Livrable |
|---|---:|---|
| Diagnostic site | 5-10 kEUR | baseline, relief, potentiel de gain |
| Etude CFD/surrogate standard | 15-30 kEUR | cartes vent/T/RH, benchmark ERA5 |
| Etude avancee thermique/canopee | 40-80 kEUR | physique dediee, validation locale |
| Surrogate/API privee | 80-150 kEUR+ | modele dedie, integration client |

### Segments prioritaires

1. Eolien onshore terrain complexe.
2. Feu/FWI comme vitrine scientifique et institutionnelle.
3. ICPE / dispersion atmospherique.
4. Agriculture gel, apres thermique credible.
5. Outdoor/parapente, seulement comme demonstration.

---

## Ordre recommande

1. Nettoyer et geler dataset v2.
2. Entrainer surrogate FNO3D v2.
3. Evaluer contre ERA5/obs/CFD teacher.
4. Selectionner 3 cas vitrines.
5. Stabiliser API gratuite.
6. Integrer l'API dans l'app meteo sur zone test.
7. Rediger article avec resultats figes.
8. Chercher 5-10 entretiens prospects.
9. Transformer les demandes recurrentes en offres payantes.
10. Utiliser traction + article + API pour BPI/i-Lab.

---

## Jalons

### Jalon A — Dataset pret

- `dataset_v2_manifest.yaml` existe.
- QA terminee.
- Splits figes.
- Cas rejetes documentes.

### Jalon B — Surrogate pret

- Modele entraine.
- Evaluation complete.
- Latence mesuree.
- Export TorchScript valide.

### Jalon C — Vitrine prete

- 3 sites demo.
- API deployee.
- Page publique claire.
- CTA et formulaire de contact.

### Jalon D — Article pret

- Figures figees.
- Ablations terminees.
- Limites assumees.
- Dataset/code ouvert partiellement defini.

### Jalon E — App branchee

- API consommee par l'app.
- Cache/fallback.
- Flags qualite affiches.
- Pas de promesse operationnelle non validee.

---

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Surrogate apprend des artefacts CFD | Mauvaise generalisation | QA stricte + split site + do-no-harm |
| API gratuite donne des resultats trop mauvais | Perte de credibilite | Sites vitrines choisis + flags + cadrage |
| Article survend le feu | Risque scientifique | Puéchabon en ablation honnete, relief pour claim vent |
| Trop de chantiers en parallele | Perte de vitesse | FNO3D d'abord, GNN/LBM apres |
| Confusion open source/proprietaire | Perte d'actif valorisable | Ouvrir code evaluation, garder dataset complet/poids avances |
| Thermique non resolue | Limite agri/gel/feu | Positionner comme future physique payante/i-Lab |

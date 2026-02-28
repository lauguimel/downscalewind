# services/validation

**Statut : TODO — V1**

## Responsabilité

Calcul des métriques de validation et génération du rapport de publication.

## Sources de vérité terrain (par ordre de qualité)

| Source | Type | Couverture | Accès |
|--------|------|-----------|-------|
| LIDAR Doppler Perdigão | Profils verticaux | IOP 2017 | NEWA dataset |
| Mâts Perdigão (~50 tours) | Points multi-hauteurs | IOP 2017 | NEWA dataset |
| Parcs éoliens open source | Points terrain complexe | Variable | OpenEnergy, NEWA |
| Stations OMM SYNOP | Points surface | Mondial | NOAA ISD |
| Radiosondages IGRA2 | Profils verticaux | Lisbonne (~200 km) | NOAA IGRA2 |

## Métriques calculées

### Par source d'observations (mâts, LIDAR)
- RMSE sur u, v, |u|, direction (circulaire) par hauteur de mât
- MBE (Mean Bias Error) — biais systématique
- Pearson R sur |u| et direction
- **Speed-up error** : ε = |(U/U_ref)_modèle − (U/U_ref)_obs|
  (U_ref = vitesse à une hauteur de référence en terrain plat amont)

### Décomposition d'erreur ERA5 → CERRA → mâts
- **Erreur 1** : ERA5 25 km → CERRA 5.5 km (erreur méso-échelle)
- **Erreur 2** : CERRA 5.5 km → mâts (erreur orographique)
→ Argument de publication : quantifier la contribution de chaque module

### Cohérence physique (CFD, sans observations)
- Résidu de divergence, conformité loi log, speed-up Jackson-Hunt
- Voir Module 2A README pour les critères détaillés

### Enveloppe d'incertitude
- Taux de couverture à 95% : fraction des observations dans l'enveloppe prédite
- Dispersion relative de l'ensemble GNN

## Comparaison systématique

Trois colonnes obligatoires dans chaque figure :
1. **ERA5 brut interpolé** (baseline)
2. **Sortie pipeline complet**
3. [Optionnel] WRF de référence si disponible dans la littérature Perdigão

## Fichiers à créer

- `load_perdigao.py` : chargement des données mâts NEWA (format NetCDF)
- `metrics.py` : RMSE, MBE, R², speed-up error, FSS, taux de couverture
- `plots.py` : figures publication-ready (profils verticaux, cartes speed-up,
  distributions d'erreur par classe de vent)
- `validation_pipeline.py` : script principal end-to-end
- `requirements.txt`, `Dockerfile`

## Notebook

`notebooks/validation_report.ipynb` — rapport reproductible avec toutes les
figures et métriques. Exécutable via `jupyter nbconvert --execute`.

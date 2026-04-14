# Puéchabon — Honest Results (2026-04-14)

## ⚠️ Correction du 2026-04-14 : le "155m" était un hack

Mon analyse précédente extrayait le CFD à 155m AGL pour matcher le vent ERA5.
C'était un ajustement artificiel qui **cachait la vraie performance CFD à Puéchabon**.

À **hauteur physique 11m** (même que le capteur ICOS) :
- CFD ws = **1.90 m/s**, bias **-1.06 m/s** (trop lent)
- ERA5 u10 = 2.87 m/s, bias -0.08 (quasi parfait)
- OBS = 2.96 m/s

→ Le CFD **sous-estime le vent** à Puéchabon à la hauteur du capteur.

## Ablation study — d'où vient le gain ?

Même jours (n=30), hauteur physique obs.

| Scénario                    | BUI MAE | ISI MAE | FWI MAE |
|-----------------------------|---------|---------|---------|
| ERA5 (baseline ECMWF)       | 8.39    | 4.89    | 8.59    |
| **ERA5 + IMERG_QM**         | **1.30**| **3.75**| **3.29**|
| CFD + ERA5-Land             | 8.42    | 7.08    | 9.78    |
| CFD + IMERG_QM ("ours")     | 1.30    | 6.14    | 5.41    |

**Leçons** :
1. **IMERG_QM seul bat ERA5 de -62% sur FWI** (8.59 → 3.29)
2. **Ajouter le CFD dégrade l'ISI** à Puéchabon (4.89 → 6.14)
3. À site plat, ERA5 u10 est déjà bien calibré

## Pourquoi le CFD perd à Puéchabon

### 1. Site plat = pas de signal topographique à résoudre
Puéchabon : plateau calcaire à 270m, pente faible. ERA5 à 25km lisse peu
(c'est un sol relativement homogène sur 25km). Rien pour le CFD à "gagner".

### 2. z0 WorldCover trop élevé
La forêt dense de chênes verts méditerranéens → WorldCover "tree cover" →
z0 ≈ 1m. Dans le CFD cela crée un profil logarithmique qui sous-estime le
vent aux 10-15 premiers mètres au-dessus du sol.

### 3. Pas de modèle de canopée explicite
Le CFD intègre la canopée dans z0 uniquement (loi de paroi). Une canopée
réelle a un displacement height (~4m) et un profil non-log au sein.
Sans modèle explicite, le CFD traite les 11m obs comme au-dessus d'un sol
très rugueux.

### 4. CFD neutre = pas de thermique
Biais T CFD -3.2°C. En été méditerranéen, la convection diurne chauffe la
surface de 3-4°C au-dessus de ERA5. Pas modélisé en simpleFoam neutre.

## Ce qu'il faut pour que le CFD gagne

### Court terme : ajouter sites à terrain complexe
Le CFD brille quand le terrain modifie le vent. Sites prioritaires :
- **FR-OHP** (Observatoire Haute-Provence, 684m, pré-Alpes)
- **ES-LJu** (Sierra Nevada, 1600m)
- **IT-Ren** (Alpes italiennes)
- **Corse** (Mistral + terrain)
- Perdigão existant (référence terrain complexe)

Attendu : gain ISI de 30-50% comme sur OPE 120m, SAC 100m, TRN 180m.

### Moyen terme : améliorer le CFD lui-même
1. **Raffinement vertical** : 2-5-10-15-20m au lieu de 0-20m
2. **Canopy drag modèle** (fvOptions plant canopy) au lieu de z0 seul
3. **buoyantSimpleFoam** pour thermique diurne
4. **Displacement height** explicite dans le mesh

### Long terme : LES sur jours de feu
~10 jours par site où la turbulence résolue ajoute de la valeur.

## Résultat pour le papier (version honnête)

**Claim 1** (robuste, démontré) :
> La correction des précipitations IMERG par QM stratifié réduit l'erreur
> FWI de **-62%** vs ERA5-Land officiel, due au biais drizzle massif
> (+972% de pluie fictive en été méditerranéen).

**Claim 2** (à démontrer, nécessite sites à terrain) :
> Sur terrain complexe, le downscaling CFD réduit l'erreur vent de 30-50%,
> apportant un gain ISI supplémentaire aux jours de Mistral/Tramontane.

**Claim 3** (limite, honnête) :
> Sur sites plats (Puéchabon), le CFD neutre n'apporte pas de valeur — le
> vrai gain fire-weather vient de la correction précipitation.

## Fichiers associés

- `data/validation/fwi_hybrid/puechabon_ABLATION.csv` — toutes les variantes
- `data/validation/fwi_hybrid/puechabon_FINAL_h155.csv` — ancien (hack 155m)
- `data/validation/RESULTS_STRUCTURE.md` — vue d'ensemble

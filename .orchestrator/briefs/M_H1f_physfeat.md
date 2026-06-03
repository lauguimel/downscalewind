# DEPARTMENT BRIEF — M_H'1f : features physiques de stabilité (JJA test ciblé) + re-éval Perdigão

## 0. CONTEXTE — pourquoi ce test, pourquoi JJA seul
M_H'1a (JJA, NOAA plaine) validé : vitesse −32.5%, direction améliorée, propagation Perdigão
100% lisse. MAIS sur Perdigão IOP la justesse au centre DÉGRADE (MAE corr 2.27 vs raw 1.37,
biais +1.92) = sur-correction. Hypothèse user : l'ANN, entraîné sur régime « vent fort
sous-estimé » (plaine été), ne sait pas « ne pas corriger » en régime vent faible/stable
(Perdigão). **Donner à l'ANN un signal physique de stabilité** (gradient T, RH, q) devrait
réduire cette sur-correction SANS nouvelle source obs.

**Séquencement décidé (user 2026-06-02)** : tester l'hypothèse VITE et PAS CHER d'abord —
features dérivées du cache JJA EXISTANT (60971 grid.zarr, zéro re-matérialisation), re-train
JJA seul (~9h), re-éval Perdigão. Si la sur-correction baisse → GO M_H'1b 4-saisons. Sinon →
le plafond est l'étage B (surrogate frozen ne sait pas le terrain raide) = chantier surrogate v3.

## 1. Fait technique VÉRIFIÉ par le Boss (ne pas re-deviner)
Les 4 features sont dérivables à la LECTURE du grid.zarr existant — `load_grid_inputs`
(dataset_v2_obs_centered.py ~L110-172) lit déjà `era5_flat` contenant t2m, d2m, et TOUS les
niveaux pression (T aux plevels). Donc :
- `gradient_T_850_surf` = T(850 hPa) − t2m
- `gradient_T_500_850` = T(500 hPa) − T(850 hPa)
- `RH_surface` = Magnus-Tetens(t2m, d2m)
- `q_surface` = humidité spécifique de (d2m, sp/pressure surface)
se calculent depuis le grid.zarr SANS re-matérialiser. `compute_topo_features` (~L52) construit
le vecteur 8-dim actuel [mean_topo, std_topo, z0_eff, lat, hour_sin/cos, month_sin/cos].

## 2. Mission ID + type
- **ID**: M_H'1f · **Type**: Implement (features + norm + ANN topo_dim) + Validate (re-train JJA ~9h + re-éval Perdigão)

## 3. Required reading
- `/Users/guillaume/Documents/Recherche/downscalewind/.orchestrator/memory/department.md`
  (Codex sans SSH Aqua → Department scp+qsub ; override store ERA5 par période ; dry-run avant qsub)
- `services/module2b-surrogate/src/dataset_v2_obs_centered.py` (compute_topo_features L52,
  load_grid_inputs L110, __getitem__ — où injecter les 4 features ; vérifier où T aux plevels
  est accessible : `g["input/era5_pressure_levels"]` + le bloc `u,v,T,q × plevels × 3×3` de era5_flat)
- `services/module2b-surrogate/src/ann_correction.py` (topo_dim L46, déjà configurable — vérifier
  que rien d'autre ne hardcode 8)
- `services/module2b-surrogate/train_v2_devine_style.py` (instancie ANN avec cfg topo_dim ;
  charge norm — voir où ajouter la normalisation des 4 features)
- `services/module2b-surrogate/eval_devine_loso.py` + `services/validation/audit_devine_perdigao.py`
  (doivent consommer le MÊME compute_topo_features → topo_dim 12 cohérent à l'éval)
- `configs/training/devine_style_full_M_H1.yaml` (topo_dim: 8 → 12 ; ajouter output_dir dédié)
- `configs/training/eval_perdigao_M_H1c.yaml` (re-utilisé pour re-éval Perdigão avec le nouveau best.pt)

## 4. Implémentation
1. **dataset** : étendre `compute_topo_features` (ou un helper appelé dans __getitem__) pour ajouter
   les 4 features → vecteur 12-dim. Les valeurs T(850), T(500) viennent du grid.zarr
   (era5_flat ou un accès direct `g["input/..."]`). Documenter l'ordre exact des 12 dims.
   ⚠️ topo_dim devient 12 SEULEMENT si cfg le demande — garder le défaut 8 pour rétrocompat
   (M_H'1a best.pt reste chargeable/évaluable).
2. **normalisation** : les 4 nouvelles features ont des échelles physiques (gradient T en K,
   RH en %, q en kg/kg) → DOIVENT être normalisées (centrage/scale) sinon gradients dominent.
   Calculer mean/std sur un échantillon du train JJA (ou constantes physiques raisonnables
   documentées) et les appliquer. NE PAS laisser des K bruts entrer dans l'ANN.
3. **ANN** : `ann_correction.py` topo_dim déjà param → juste passer 12 via cfg. Vérifier aucun
   autre hardcode 8.
4. **config** : `configs/training/devine_style_M_H1f.yaml` = clone M_H1 avec topo_dim: 12,
   `enable_phys_features: true` (flag), output_dir `surrogate_v2_devine_M_H1f_physfeat`,
   reste identique (lr 5e-4, grad_clip 1.0, τ 0.6/0.4, epochs 8, batch_size 4, val_frac 0.2,
   seed 42, exclude perdigao, cache JJA existant, overwrite_cache false).
5. **PBS** : `configs/hpc/devine_style_M_H1f.pbs` (clone du H1 PBS, gpu_id H100, walltime 14h).

## 5. Validation + run
- **dry-run CPU 1-batch** : vérifier topo_features.shape == (B,12), ANN forward OK, les 4 features
  finies (pas NaN), normalisées (ordres de grandeur ~O(1)).
- **re-train JJA** : qsub ~9h. MÊME split watertight que M_H'1a → val-43 directement comparable.
- **re-éval LOSO** : val mae_corrected/raw + comparer au M_H'1a (1.223). Les features ne doivent
  PAS dégrader le gain NOAA.
- **re-éval Perdigão** : relancer `audit_devine_perdigao.py` avec le nouveau best.pt (config
  override 2017 + topo_dim 12). Comparer MAE centre corr vs M_H'1a (2.27). C'est LE chiffre clef.

## 6. Allowed edit zones
- `services/module2b-surrogate/src/dataset_v2_obs_centered.py` (modify : +4 features, additive)
- `services/module2b-surrogate/src/ann_correction.py` (modify SEULEMENT si hardcode 8 résiduel)
- `services/module2b-surrogate/train_v2_devine_style.py` (modify : norm des 4 features si besoin)
- `services/module2b-surrogate/eval_devine_loso.py`, `services/validation/audit_devine_perdigao.py`
  (modify : consommer topo_dim 12 cohérent)
- `configs/training/devine_style_M_H1f.yaml`, `configs/hpc/devine_style_M_H1f.pbs` (new)
- `data/models/surrogate_v2_devine_M_H1f_physfeat/`, `data/validation/phase_H_prime_*` (sorties)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`
- Aqua : scp scripts/configs + écriture models/validation

## 7. Forbidden
- ZÉRO re-matérialisation (cache JJA reste 60971 ; features dérivées à la lecture).
- PAS de modif model_vit_v2.py / dataset_v2_vit.py / surrogate v2 best.pt / M_H'1a best.pt (frozen).
- Rétrocompat : défaut topo_dim=8 inchangé (M_H'1a doit rester évaluable).
- PAS de re-download CDS (M_H'1b'' hors scope). PAS de commit/push. qsub ≤14h.
- Perdigão = test set immuable, READ-ONLY.

## 8. Exit criterion
- dataset produit topo_features (B,12) finies+normalisées ; dry-run CPU OK.
- re-train JJA exit 0, best.pt produit ; val mae_corrected ≈ M_H'1a 1.223 (±0.1, ne doit pas dégrader NOAA).
- **re-éval Perdigão : MAE centre corr comparée à 2.27** → baisse = hypothèse validée.
- cache JJA 60971 inchangé. Diff zones autorisées only.

## 9. Runner + ops
`codex` via run-engineer.sh, single-spawn. Codex SANS SSH Aqua → Engineer livre code + dry-run
CPU local ; Department scp + qsub (~9h, ne pas attendre in-turn) + récup. SSH conda : `module load
Miniconda3/24.9.2-0; eval "$(conda shell.bash hook)"; conda activate fuxicfd`. Shell stdout vide
→ rediriger /tmp + Read. Write `.engineer_brief.md` PUIS run-engineer.sh (séparé).

## 10. Report (≤300 mots)
- Implémentation : 4 features (ordre des 12 dims), normalisation appliquée, rétrocompat 8 OK.
- dry-run : shape (B,12), features finies/normalisées.
- Si run fini in-turn : val-43 mae_corrected (vs 1.223, NOAA dégradé ?) + **Perdigão MAE centre
  (vs 2.27, hypothèse validée ?)** → verdict GO/NO-GO M_H'1b 4-saisons.
- Sinon : "code done + scp'd + job <ID>" pour Boss hpc-watch (re-éval Perdigão = 2e job court après).
- cache 60971. Job ID. proposals (≤3) + memory candidates (≤3).

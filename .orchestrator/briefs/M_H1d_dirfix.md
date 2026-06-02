# DEPARTMENT BRIEF — M_H'1e : fix bug direction de l'eval + re-run val + verdict agrégé

## 0. CONTEXTE — vitesse DÉJÀ GREEN, seule la direction est en jeu
Job 22162041 (exit 0, 1h09) a confirmé la VITESSE de façon décisive (val-43 mae_corrected
**1.223** vs raw 1.812 vs era5 1.315, −32.5%, biais +0.008, bat ERA5). NE PAS retoucher la
vitesse. Cette mission corrige UNIQUEMENT le volet direction, qui est buggé.

## 1. Bug localisé par le Boss (ne pas re-deviner)
Dans `services/module2b-surrogate/eval_devine_loso.py` :
- la VITESSE compare au `speed_obs` **déjà apparié dans le parquet** → correct.
- la DIRECTION fait un SECOND appariement séparé `_attach_obs_direction` (L314-353) qui
  rejoint l'obs store par nearest-time, ET compare contre `dir_era5 = _dir_from_uv(era5_u10_raw,
  era5_v10_raw)` où `era5_u10_raw` vient des arrays du forward (pas du parquet).
- Résultat : TOUTES les erreurs dir gonflées d'un ~+33° quasi constant. eval donne
  raw 67° / corr 71° / **era5 74°**, biais +32 à +40°.
- **Preuve du bug** : recalcul direct du parquet (Boss) avec UNE convention cohérente
  `_dir_from_uv` pour pred ET obs → **raw 43° / era5 34°** (ALL val, n=12488). Calm>=1 :
  raw 40.5° / era5 31.2°. ERA5 (aucun modèle, juste u10/v10 vs obs) ne peut valoir 34 ET 74°.
- Le `obs_uv_dir_check` du JSON (mae 3e-6) prouve seulement que le store `wind_dir` ==
  `_dir_from_uv(obs_u,obs_v)` DANS le store — il ne valide PAS l'appariement aux pairings.

## 2. Le fix (précis)
Calculer la direction à partir du MÊME obs apparié que la vitesse :
- direction obs = `_dir_from_uv(u_obs, v_obs)` (colonnes parquet déjà appariées) — PAS de
  join obs-store séparé.
- direction raw = `_dir_from_uv(u_pred, v_pred)` (parquet).
- direction ERA5 = `_dir_from_uv(u10_era5_baseline, v10_era5_baseline)` (parquet, PAS les
  arrays forward).
- direction corrigée = `_dir_from_uv(u_pred_corr, v_pred_corr)` (sortie du forward ANN+surrogate
  au pixel central — la SEULE quantité qui exige le GPU).
- erreur angulaire = `_angular_diff_deg` (wrap ±180°), MÊME `_dir_from_uv` partout → toute
  constante de convention s'annule.
- **Supprimer / contourner** `_attach_obs_direction` pour la métrique direction (le garder
  seulement si une colonne obs manque, mais ici u_obs/v_obs existent dans le parquet).

## 3. Deliverables
1. **Persister un parquet par-pairing** `data/validation/phase_H_prime_loso/pairing_dir.parquet` :
   `[station_id, timestamp, split, u_obs, v_obs, u_pred_raw, v_pred_raw, u_pred_corr, v_pred_corr,
   u10_era5_baseline, v10_era5_baseline]` → rend toute ré-analyse direction CPU-only et ré-vérifiable.
2. **`loso_summary_dir.json`** avec le verdict EN AGRÉGÉ sur la distribution (val ET train) :
   - `mae_dir_deg` et `median_dir_deg` et `bias_dir_deg` pour corrected / raw / era5,
     calculés sur TOUTES les val pairings (pas moyennés par station).
   - distribution : buckets [0-15,15-30,30-45,45-90,>90] en **% des pairings** (pas # stations).
   - stratification par classe de vent obs (calme<1, faible 1-3, moyen 3-7, fort>7) :
     l'erreur directionnelle en vent calme est physiquement non informative → la reporter à part.
   - **garde-fou user** : NE PAS produire de verdict per-station comme preuve. Une station
     isolée à grande erreur = microrelief sous-maille non capté, attendu. Le verdict se lit
     sur la distribution agrégée, et l'arbitre est corrigé **vs OBS**, jamais vs ERA5.
3. Sanity : imprimer côte-à-côte le recalcul "depuis parquet brut" (doit retrouver raw 43° /
   era5 34° sur ALL val) pour prouver que le fix élimine le +33° fantôme.

## 4. Re-run
- Forward-only, **VAL stations seulement** (~12061 pairings, ~1/5 du run précédent → ~13-15 min
  GPU) suffit pour le verdict direction. Réutiliser le cache (require_cached=True, ZERO
  matérialisation). PBS `configs/hpc/eval_devine_loso.pbs` (A100, 2h, déjà OK) — ajouter un
  flag `--val-only` ou équivalent si simple ; sinon run complet accepté.
- Cache DOIT rester 60971 dirs avant ET après.

## 5. Allowed edit zones
- `services/module2b-surrogate/eval_devine_loso.py` (modify : volet direction + dump parquet)
- `configs/hpc/eval_devine_loso.pbs` (modify : éventuel flag val-only)
- `data/validation/phase_H_prime_loso/` (sorties)
- `.engineer_logs/`, `scratch/`, `tmp/`, `test/scratch/`
- Aqua : scp script(s) + écriture sous `data/validation/phase_H_prime_loso/`

## 6. Forbidden
- NE PAS toucher au calcul VITESSE (déjà GREEN) ni à `_step`/forward de la vitesse.
- AUCUNE matérialisation (cache reste 60971). AUCUN re-train. AUCUN qsub >2h.
- PAS de modif model_vit_v2.py / dataset_v2_vit.py / surrogate best.pt / M_H'1a best.pt /
  train split logic. PAS de commit/push.

## 7. Exit criterion
- `pairing_dir.parquet` écrit (val au moins) avec les 4 jeux u,v + obs.
- `loso_summary_dir.json` : mae/median/bias dir corrected/raw/era5 EN AGRÉGÉ val, distribution
  en %, stratifié classe vent.
- Sanity : le recalcul "parquet brut" retrouve raw≈43° / era5≈34° (ALL val) → preuve fix OK.
- Cache == 60971 avant ET après. Diff zones autorisées only.

## 8. Runner + validation
`codex` via run-engineer.sh, single-spawn. ⚠️ Codex sandbox N'A PAS SSH Aqua
(`Could not resolve hostname aqua` constaté M_H'1d″) → l'Engineer livre le CODE + dry-run CPU
local 1-batch ; **c'est le Department (toi) qui scp + qsub sur Aqua** (tu as SSH). Process :
Write `.engineer_brief.md` PUIS run-engineer.sh (séparé). SSH conda : `module load
Miniconda3/24.9.2-0; eval "$(conda shell.bash hook)"; conda activate fuxicfd`. Shell stdout
rend souvent vide → rediriger vers /tmp et Read.

## 9. Report (≤300 mots)
- Fix appliqué (direction depuis parquet, convention unique, dump pairing parquet).
- DIRECTION agrégée val : mae_dir corrected vs raw(~43°) vs era5(~34°) ; médiane ; biais ;
  distribution %. Stratifié vent (calme vs fort).
- Sanity recalcul parquet OK (raw 43 / era5 34 retrouvés).
- **VERDICT (agrégé, vs OBS, garde-fou microrelief)** : l'ANN speed-only AMÉLIORE / NEUTRE /
  DÉGRADE la direction vs OBS en agrégé ? → (a) dégrade nettement → justifie mission
  ANN_direction (DEVINE 2-réseaux speed+dir) ; (b) neutre/améliore → speed-only OK → M_H'1c Perdigão.
- Cache 60971 confirmé. Job ID. proposals (≤3) + memory candidates (≤3).

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module 1 — Notebook opérationnel : entraînement, évaluation, visualisation

    **Usage :** Modifier la cellule **[CONFIG]**, puis exécuter toutes les cellules dans l'ordre.

    | Section | Cellules |
    |---------|----------|
    | A — Setup | CONFIG, imports, statut des données |
    | B — Entraînement | Lancer train.py, courbes MLflow |
    | C — Évaluation | Charger modèle, métriques test set |
    | D — Visualisation | Champs interpolés sur une fenêtre ERA5 |
    | E — Inférence batch | Produire un store Zarr 1h complet |
    | F — Ablation (optionnel) | Comparer plusieurs grid_size |
    """)
    return


@app.cell
def _():
    # ═══════════════════════════════════════════════════════════════════════════
    # [CONFIG] ← SEULE CELLULE À MODIFIER
    # ═══════════════════════════════════════════════════════════════════════════

    # Chemins vers les données
    ZARR_6H  = "../../data/raw/era5_perdigao.zarr"
    ZARR_1H  = "../../data/raw/era5_hourly_perdigao.zarr"
    OUT_DIR  = "../../data/models/module1"      # modèles + norm_stats.json
    ZARR_OUT = "../../data/interim/era5_1h_module1_perdigao.zarr"  # infer.py output

    # Hyperparamètres d'entraînement
    GRID_SIZE  = 7       # ablation: 3, 5, 7, 9, 11 (None = tout le domaine)
    N_HIDDEN   = 48      # canaux cachés du CNN résiduel
    BATCH_SIZE = 32
    LR         = 3e-4
    MAX_EPOCHS = 300
    PATIENCE   = 30
    DEVICE     = "auto"  # 'cpu', 'cuda', 'mps', ou 'auto'

    # MLflow
    EXPERIMENT = "module1-temporal"
    RUN_NAME   = f"grid{GRID_SIZE}_h{N_HIDDEN}"

    # Fenêtre pour la visualisation (cellules D)
    # Format YYYY-MM-DD. S0 = t0, S1 = t0+6h.
    VIZ_DATE   = "2017-05-15"   # doit être dans les données disponibles
    VIZ_HOUR   = "06:00"        # heure de S0 (00, 06, 12, ou 18)

    # Période pour l'inférence batch (cellule E)
    INFER_START = "2017-05-01"
    INFER_END   = "2017-06-15"
    return (
        BATCH_SIZE,
        DEVICE,
        EXPERIMENT,
        GRID_SIZE,
        INFER_END,
        INFER_START,
        LR,
        MAX_EPOCHS,
        N_HIDDEN,
        OUT_DIR,
        PATIENCE,
        RUN_NAME,
        VIZ_DATE,
        VIZ_HOUR,
        ZARR_1H,
        ZARR_6H,
        ZARR_OUT,
    )


@app.cell
def _(OUT_DIR, ZARR_1H, ZARR_6H, ZARR_OUT):
    # ── Imports ───────────────────────────────────────────────────────────────────
    import sys
    import subprocess
    import json
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import torch

    ROOT    = (Path(".").resolve() / "../.." ).resolve()
    MODULE1 =  Path(".").resolve()
    for p in [str(ROOT), str(MODULE1)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from src.model import AdvectionResidualInterpolator
    from src.normalization import NormStats, VARIABLE_ORDER

    # Résoudre les chemins
    ZARR_6H_P  = Path(ZARR_6H).resolve()
    ZARR_1H_P  = Path(ZARR_1H).resolve()
    OUT_DIR_P  = Path(OUT_DIR).resolve()
    ZARR_OUT_P = Path(ZARR_OUT).resolve()

    TRAIN_PY   = MODULE1 / "train.py"
    EVAL_PY    = MODULE1 / "evaluate.py"
    INFER_PY   = MODULE1 / "infer.py"

    print(f"ROOT    : {ROOT}")
    print(f"Python  : {sys.executable}")
    return (
        AdvectionResidualInterpolator,
        EVAL_PY,
        INFER_PY,
        NormStats,
        OUT_DIR_P,
        Path,
        TRAIN_PY,
        VARIABLE_ORDER,
        ZARR_1H_P,
        ZARR_6H_P,
        ZARR_OUT_P,
        mcolors,
        np,
        plt,
        subprocess,
        sys,
        torch,
    )


@app.cell
def _(OUT_DIR_P, Path, ZARR_1H_P, ZARR_6H_P, ZARR_OUT_P):
    # ── Statut des données et artefacts ──────────────────────────────────────────
    def _status(path, label):
        p = Path(path)
        if p.exists():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
            size_str = f"{size/1e9:.2f} GB" if size > 1e9 else (f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.0f} KB")
            print(f"  ✓  {label:<30s} {size_str:>10s}   {p}")
            return True
        else:
            print(f"  ✗  {label:<30s} {'manquant':>10s}   {p}")
            return False

    print("═" * 70)
    print("STATUT DES DONNÉES ET ARTEFACTS")
    print("═" * 70)
    ok_6h   = _status(ZARR_6H_P,  "Zarr ERA5 6h (entrées)")
    ok_1h   = _status(ZARR_1H_P,  "Zarr ERA5 1h (cibles train)")
    ok_norm = _status(OUT_DIR_P / "norm_stats.json", "norm_stats.json")
    ok_model= _status(OUT_DIR_P / "best_model.pt",   "best_model.pt")
    ok_infer= _status(ZARR_OUT_P, "Zarr inférence 1h (sortie)")
    print("═" * 70)

    if not ok_6h:
        print("\n⚠  ERA5 6h manquant. Lancer :")
        print(f"   python services/data-ingestion/ingest_era5.py --site perdigao --start 2016-01 --end 2017-06 --output {ZARR_6H_P}")
    if not ok_1h:
        print("\n⚠  ERA5 1h manquant. Lancer :")
        print(f"   python services/data-ingestion/ingest_era5_hourly.py --site perdigao --start 2016-01 --end 2017-06 --output {ZARR_1H_P}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## B — Entraînement
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DEVICE,
    EXPERIMENT,
    GRID_SIZE,
    LR,
    MAX_EPOCHS,
    N_HIDDEN,
    OUT_DIR_P,
    PATIENCE,
    RUN_NAME,
    TRAIN_PY,
    ZARR_1H_P,
    ZARR_6H_P,
    subprocess,
    sys,
):
    # ── Lancer train.py (streaming de la sortie en temps réel) ───────────────────
    # Cette cellule lance train.py comme processus externe et affiche la sortie
    # en continu. Elle peut prendre plusieurs heures sur CPU.
    cmd = [sys.executable, str(TRAIN_PY), '--zarr-6h', str(ZARR_6H_P), '--zarr-1h', str(ZARR_1H_P), '--output', str(OUT_DIR_P), '--experiment', EXPERIMENT, '--run-name', RUN_NAME, '--device', DEVICE, '--max-epochs', str(MAX_EPOCHS), '--patience', str(PATIENCE), '--batch-size', str(BATCH_SIZE), '--lr', str(LR), '--n-hidden', str(N_HIDDEN)]
    if GRID_SIZE is not None:
        cmd += ['--grid-size', str(GRID_SIZE)]
    print('Commande :')
    print(' '.join(cmd))
    print('=' * 70)
    _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for _line in _proc.stdout:
        print(_line, end='', flush=True)
    _proc.wait()
    ret = _proc.returncode
    print(f"\n{('✓ Entraînement terminé' if ret == 0 else f'✗ Erreur (code {ret})')}")
    return


@app.cell
def _(EXPERIMENT, OUT_DIR_P, mo):
    # ── Charger les runs MLflow et tracer les courbes ────────────────────────────
    import mlflow

    MLRUNS_DIR = OUT_DIR_P.parent / "mlruns"
    mlflow.set_tracking_uri(str(MLRUNS_DIR))

    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT],
            order_by=["metrics.val_rmse_u ASC"],
        )
        if runs.empty:
            print("Aucune run MLflow trouvée.")
        else:
            print(f"{len(runs)} run(s) trouvée(s) dans l'expérience '{EXPERIMENT}'")
            cols = ["tags.mlflow.runName", "metrics.val_rmse_u", "metrics.best_epoch",
                    "params.grid_size", "params.n_hidden", "params.n_params"]
            mo.output.replace(mo.ui.table(runs[[c for c in cols if c in runs.columns]].head(10)))
    except Exception as e:
        print(f"MLflow non disponible : {e}")
        runs = None
    return mlflow, runs


@app.cell
def _(mlflow, np, plt, runs):
    # ── Courbes de loss : train_loss et val_rmse_u ────────────────────────────────
    # Charge les métriques de la meilleure run (val_rmse_u le plus bas)
    if runs is not None and (not runs.empty):
        best_run_id = runs.iloc[0]['run_id']
        client = mlflow.MlflowClient()
        train_hist = client.get_metric_history(best_run_id, 'train_loss')
        val_hist = client.get_metric_history(best_run_id, 'val_rmse_u')
        epochs_tr = [m.step for m in train_hist]
        losses_tr = [m.value for m in train_hist]
        epochs_val = [m.step for m in val_hist]
        rmse_val = [m.value for m in val_hist]
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        ax1.plot(epochs_tr, losses_tr, lw=2, color='steelblue')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Loss pondérée')
        ax1.set_title('Train loss')
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs_val, rmse_val, lw=2, color='darkorange')
        best_idx = int(np.argmin(rmse_val))
        ax2.axvline(epochs_val[best_idx], ls='--', color='red', alpha=0.6, label=f'Best epoch {epochs_val[best_idx]} ({rmse_val[best_idx]:.4f})')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('RMSE (σ)')
        ax2.set_title('Val RMSE u (850 hPa moyen)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        run_name = runs.iloc[0].get('tags.mlflow.runName', best_run_id[:8])
        plt.suptitle(f'Run : {run_name}', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print("Pas de runs MLflow — lancer l'entraînement d'abord.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## C — Évaluation
    """)
    return


@app.cell
def _(AdvectionResidualInterpolator, NormStats, OUT_DIR_P, torch):
    # ── Charger le meilleur modèle ────────────────────────────────────────────────
    MODEL_PATH = OUT_DIR_P / "best_model.pt"
    NORM_PATH  = OUT_DIR_P / "norm_stats.json"

    if not MODEL_PATH.exists():
        print(f"✗ {MODEL_PATH} introuvable — entraîner le modèle d'abord.")
    else:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model = AdvectionResidualInterpolator(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        ns = NormStats.load(NORM_PATH)

        print("Modèle chargé :")
        kw = checkpoint["model_kwargs"]
        print(f"  Paramètres   : {model.count_parameters():,}")
        print(f"  n_hidden     : {kw['n_hidden']}")
        print(f"  dx_m_eff     : {kw['dx_m']:.0f} m/σ")
        print(f"  Meilleure époque : {checkpoint['epoch']}  val_rmse_u={checkpoint['val_rmse_u']:.4f}")

        std_u, std_v = ns.get_wind_std()
        print(f"\nNormStats : std_u={std_u:.2f} m/s, std_v={std_v:.2f} m/s")
    return MODEL_PATH, NORM_PATH, model, ns


@app.cell
def _(
    EVAL_PY,
    GRID_SIZE,
    MODEL_PATH,
    NORM_PATH,
    ZARR_1H_P,
    ZARR_6H_P,
    subprocess,
    sys,
):
    # ── Lancer evaluate.py sur le test set ────────────────────────────────────────
    if MODEL_PATH.exists() and ZARR_6H_P.exists() and ZARR_1H_P.exists():
        cmd_eval = [sys.executable, str(EVAL_PY), '--zarr-6h', str(ZARR_6H_P), '--zarr-1h', str(ZARR_1H_P), '--model', str(MODEL_PATH), '--norm-stats', str(NORM_PATH), '--split', 'test']
        if GRID_SIZE is not None:
            cmd_eval += ['--grid-size', str(GRID_SIZE)]
        print('Évaluation sur le test set (IOP Perdigão 2017-05 → 2017-06)…')
        print('=' * 70)
        _proc = subprocess.Popen(cmd_eval, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        eval_output = []
        for _line in _proc.stdout:
            print(_line, end='', flush=True)
            eval_output.append(_line)
        _proc.wait()
    else:
        print('Données ou modèle manquants — vérifier le statut en cellule A.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## D — Visualisation : inférence sur une fenêtre ERA5
    """)
    return


@app.cell
def _(VARIABLE_ORDER, VIZ_DATE, VIZ_HOUR, ZARR_6H_P, model, np, ns, torch):
    # ── Charger une fenêtre ERA5 réelle ──────────────────────────────────────────
    # Cherche t0 = VIZ_DATE VIZ_HOUR dans le store 6h
    import zarr as zarr_lib

    if not ZARR_6H_P.exists():
        print("Zarr 6h manquant — cellule de visualisation ignorée.")
        viz_ok = False
    else:
        store_6h = zarr_lib.open(str(ZARR_6H_P), mode="r")
        lats = store_6h["coords/lat"][:]
        lons = store_6h["coords/lon"][:]
        levels_hpa = [int(l) for l in store_6h["coords/level"][:]]
        N_LAT, N_LON = len(lats), len(lons)

        # Trouver l'index de t0
        def _to_h(arr):
            a = np.asarray(arr)
            return a.astype("datetime64[ns]").astype("datetime64[h]") if a.dtype.kind in ("i","u") else a.astype("datetime64[h]")

        times_6h = _to_h(store_6h["coords/time"][:])
        t0_target = np.datetime64(f"{VIZ_DATE}T{VIZ_HOUR.replace(':', '')[:2]}", "h")
        idx_candidates = np.where(times_6h == t0_target)[0]

        if len(idx_candidates) == 0:
            print(f"✗ Timestamp {t0_target} introuvable dans le store 6h.")
            print(f"  Premiers timestamps : {times_6h[:8].tolist()}")
            viz_ok = False
        elif idx_candidates[0] + 1 >= len(times_6h):
            print(f"✗ Pas de S1 disponible après {t0_target} (dernier pas de temps).")
            viz_ok = False
        else:
            idx_s0 = int(idx_candidates[0])
            idx_s1 = idx_s0 + 1
            t0 = times_6h[idx_s0]
            t1 = times_6h[idx_s1]

            if t1 != t0 + np.timedelta64(6, "h"):
                print(f"✗ t1={t1} n'est pas t0+6h (manque dans les données).")
                viz_ok = False
            else:
                print(f"✓ Fenêtre ERA5 chargée : {t0} → {t1}")

                snap0_raw = np.stack(
                    [store_6h[f"pressure/{v}"][idx_s0].astype(np.float32) for v in VARIABLE_ORDER], axis=0
                )  # (V, L, H, W)
                snap1_raw = np.stack(
                    [store_6h[f"pressure/{v}"][idx_s1].astype(np.float32) for v in VARIABLE_ORDER], axis=0
                )
                snap0_norm = ns.normalize_snapshot(snap0_raw)
                snap1_norm = ns.normalize_snapshot(snap1_raw)

                S0t = torch.from_numpy(snap0_norm).unsqueeze(0)  # (1, V, L, H, W)
                S1t = torch.from_numpy(snap1_norm).unsqueeze(0)

                with torch.no_grad():
                    preds = model(S0t, S1t)  # (1, 5, V, L, H, W)

                print(f"  Grille : {N_LAT}×{N_LON}, niveaux : {levels_hpa}")
                viz_ok = True
    return (
        N_LAT,
        N_LON,
        viz_ok,
        lats,
        levels_hpa,
        lons,
        preds,
        snap0_norm,
        snap1_norm,
        t0,
        t1,
    )


@app.cell
def _(
    MODEL_PATH,
    VIZ_DATE,
    viz_ok,
    lats,
    levels_hpa,
    lons,
    mcolors,
    np,
    ns,
    plt,
    preds,
    snap0_norm,
    snap1_norm,
    t0,
    t1,
):
    # ── Figure principale : vent 850 hPa sur 7 heures (t0 … t+6h) ────────────────
    if not viz_ok or not MODEL_PATH.exists():
        print('Visualisation ignorée (données ou modèle manquants).')
    else:
        LEV_IDX = levels_hpa.index(850) if 850 in levels_hpa else 2
        taus = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
        XX, YY = np.meshgrid(lons, lats)

        def _denorm_uv(snap_norm, level_idx):  # Construire les 7 frames : S0, τ×5, S1
            u_p = ns.denormalize(snap_norm[0], 'u')[level_idx]
            v_p = ns.denormalize(snap_norm[1], 'v')[level_idx]
            return (u_p, v_p)
        frames = []
        hours = []
        frames.append(_denorm_uv(snap0_norm, LEV_IDX))
        hours.append(str(t0)[-5:])
        for k, tau in enumerate(taus):
            pred_k_norm = preds[0, k].numpy()
            frames.append(_denorm_uv(pred_k_norm, LEV_IDX))
            t_k = t0 + np.timedelta64(k + 1, 'h')  # (V, L, H, W)
            hours.append(str(t_k)[-5:])
        frames.append(_denorm_uv(snap1_norm, LEV_IDX))
        hours.append(str(t1)[-5:])
        ws_all = [np.sqrt(u ** 2 + v ** 2) for u, v in frames]
        vmin, vmax = (0, max((ws.max() for ws in ws_all)))
        _fig, _axes = plt.subplots(2, 7, figsize=(22, 6), gridspec_kw={'height_ratios': [1, 0.05]})  # Calculer les limites de vitesse
        cmap = plt.cm.RdBu_r
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for _col, ((u_f, v_f), hour, ws) in enumerate(zip(frames, hours, ws_all)):
            _ax = _axes[0, _col]
            im = _ax.pcolormesh(lons, lats, ws, cmap=cmap, norm=norm)
            _ax.quiver(XX, YY, u_f, v_f, scale=200, width=0.006, color='k', alpha=0.7)
            _ax.set_title(hour, fontsize=9)
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _col in (0, 6):
                for spine in _ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2.5)
            _axes[1, _col].axis('off')
        plt.colorbar(im, ax=_axes[0, :], orientation='horizontal', pad=0.01, fraction=0.02, label='Vitesse vent m/s')  # Encadrer S0 et S1 en rouge (données connues)
        plt.suptitle(f'Vent {levels_hpa[LEV_IDX]} hPa — interpolation Module 1  ({VIZ_DATE})\n[Cadre rouge = données connues, autres = prédictions du modèle]', fontsize=11)
        plt.tight_layout()
        plt.show()
    return hours, taus


@app.cell
def _(
    MODEL_PATH,
    N_LAT,
    N_LON,
    VIZ_DATE,
    viz_ok,
    hours,
    lats,
    levels_hpa,
    lons,
    np,
    ns,
    plt,
    preds,
    snap0_norm,
    snap1_norm,
    taus,
):
    # ── Figure secondaire : profils verticaux u, v, T au centre du domaine ────────
    if viz_ok and MODEL_PATH.exists():
        ci, cj = (N_LAT // 2, N_LON // 2)
        p_levels = np.array(levels_hpa, dtype=float)
        _fig, _axes = plt.subplots(1, 3, figsize=(13, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, 7))
        labels = [hours[0]] + [f'τ={t:.2f}' for t in taus] + [hours[-1]]
        for _col, (var_idx, var_name, _ax) in enumerate([(0, 'u (m/s)', _axes[0]), (1, 'v (m/s)', _axes[1]), (3, 'T (K)', _axes[2])]):
            var_key = ['u', 'v', 'z', 't', 'q'][var_idx]
            for fi, (frame_norm, label, color) in enumerate(zip([snap0_norm] + [preds[0, k].numpy() for k in range(5)] + [snap1_norm], labels, colors)):
                phys = ns.denormalize(frame_norm[var_idx], var_key)
                profile = phys[:, ci, cj]
                lw = 2.5 if fi in (0, 6) else 1.0
                ls = '-' if fi in (0, 6) else '--'
                _ax.plot(profile, p_levels, color=color, lw=lw, ls=ls, label=label if fi in (0, 3, 6) else None)
            _ax.invert_yaxis()
            _ax.set_ylabel('Pression (hPa)')
            _ax.set_xlabel(var_name)
            _ax.grid(True, alpha=0.3)
            _ax.legend(fontsize=7)
        plt.suptitle(f'Profils verticaux au centre ({lats[ci]:.2f}°N, {lons[cj]:.2f}°E)  {VIZ_DATE}', fontsize=11)  # (L, H, W)
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## E — Inférence batch : produire un store Zarr 1h
    """)
    return


@app.cell
def _(
    DEVICE,
    INFER_END,
    INFER_PY,
    INFER_START,
    MODEL_PATH,
    NORM_PATH,
    ZARR_6H_P,
    ZARR_OUT_P,
    subprocess,
    sys,
):
    # ── Lancer infer.py ───────────────────────────────────────────────────────────
    if MODEL_PATH.exists() and ZARR_6H_P.exists():
        cmd_infer = [sys.executable, str(INFER_PY), '--zarr-6h', str(ZARR_6H_P), '--zarr-out', str(ZARR_OUT_P), '--model', str(MODEL_PATH), '--norm-stats', str(NORM_PATH), '--start', INFER_START, '--end', INFER_END, '--device', DEVICE, '--overwrite']
        print(f'Inférence : {INFER_START} → {INFER_END}')
        print(f'Sortie : {ZARR_OUT_P}')
        print('=' * 70)
        _proc = subprocess.Popen(cmd_infer, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for _line in _proc.stdout:
            print(_line, end='', flush=True)
        _proc.wait()
        print(f"{('✓ Done' if _proc.returncode == 0 else f'✗ Erreur (code {_proc.returncode})')}")
    else:
        print('Modèle ou Zarr 6h manquant.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## F — Ablation : comparer plusieurs grid_size (optionnel)

    Cette section lance un entraînement rapide (`--max-epochs 50`) pour chaque `grid_size`
    et compare les `val_rmse_u` finaux. Décommenter pour activer.
    """)
    return


@app.cell
def _():
    # ── Ablation grid_size ────────────────────────────────────────────────────────
    # Décommenter et exécuter séparément — peut prendre plusieurs heures.

    # GRID_SIZES_ABLATION = [3, 5, 7]   # ajouter 9, 11 si les données le permettent
    # EPOCHS_ABLATION = 50
    # ablation_results = {}   # {grid_size: val_rmse_u}
    #
    # for gs in GRID_SIZES_ABLATION:
    #     out_gs = OUT_DIR_P.parent / f"module1_g{gs}"
    #     cmd_gs = [
    #         sys.executable, str(TRAIN_PY),
    #         "--zarr-6h",    str(ZARR_6H_P),
    #         "--zarr-1h",    str(ZARR_1H_P),
    #         "--output",     str(out_gs),
    #         "--grid-size",  str(gs),
    #         "--max-epochs", str(EPOCHS_ABLATION),
    #         "--patience",   "15",
    #         "--experiment", EXPERIMENT,
    #         "--run-name",   f"ablation_g{gs}",
    #         "--device",     DEVICE,
    #     ]
    #     print(f"\n{'='*60}\ngrid_size={gs}\n{'='*60}")
    #     proc = subprocess.Popen(cmd_gs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    #                             text=True, bufsize=1)
    #     last_rmse = None
    #     for line in proc.stdout:
    #         if "val_rmse" in line:
    #             print(line, end="", flush=True)
    #             try:
    #                 last_rmse = float(line.split("val_rmse_u=")[1].split()[0])
    #             except Exception:
    #                 pass
    #     proc.wait()
    #     ablation_results[gs] = last_rmse
    #
    # # Graphe comparatif
    # fig, ax = plt.subplots(figsize=(7, 4))
    # gs_list = sorted(ablation_results.keys())
    # rmse_list = [ablation_results[gs] for gs in gs_list]
    # ax.plot(gs_list, rmse_list, "o-", lw=2, ms=8)
    # for gs, rmse in zip(gs_list, rmse_list):
    #     ax.annotate(f"{rmse:.4f}", (gs, rmse), textcoords="offset points", xytext=(0, 8))
    # ax.set_xlabel("grid_size (pixels ERA5)")
    # ax.set_ylabel("val RMSE u (σ)")
    # ax.set_title("Ablation : taille de contexte spatial")
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
    # print(ablation_results)

    print("Cellule d'ablation commentée — décommenter pour activer.")
    return


if __name__ == "__main__":
    app.run()

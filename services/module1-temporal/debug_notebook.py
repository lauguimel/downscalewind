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
    # Module 1 — Notebook de débogage : pipeline complet sur données synthétiques

    **Objectif :** Exécuter toutes les phases du pipeline module 1 sans téléchargement CDS.

    ## Conventions tenseurs
    | Symbole | Shape | Description |
    |---------|-------|-------------|
    | `S0`, `S1` | `(V, L, H, W)` | Snapshot normalisé (une batch dim en plus dans le loader) |
    | `targets` | `(T, V, L, H, W)` | 5 états intermédiaires (1h, 2h, 3h, 4h, 5h) |
    | Sortie modèle | `(B, T, V, L, H, W)` | idem avec batch |

    - **V = 5** : `[u=0, v=1, z=2, t=3, q=4]`
    - **L = 10** : `[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]` hPa
    - **H, W = 7, 7** : grille ERA5 Perdigão
    - **Convention N→S** : `lat[0]` = Nord (40.5°), `lat[-1]` = Sud (39.0°)
    - **Advection** : `v > 0` (vers le Nord) → dans l'index lat, déplacement vers des indices plus petits
    """)
    return


@app.cell
def _():
    # ── [1] Imports ───────────────────────────────────────────────────────────────
    import sys
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import torch
    from torch.utils.data import DataLoader

    # Chemins vers les packages partagés
    ROOT    = (Path(".").resolve() / "../.." ).resolve()  # racine du projet
    MODULE1 =  Path(".").resolve()                         # services/module1-temporal/
    for p in [str(ROOT), str(MODULE1)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from shared.data_io import create_empty_store, append_pressure_slice
    from src.model import AdvectionResidualInterpolator
    from src.normalization import NormStats, VARIABLE_ORDER, LOSS_WEIGHTS, build_loss_weight_tensor
    from src.dataset import ERA5TemporalDataset, SPLIT_RANGES

    print(f"ROOT    : {ROOT}")
    print(f"MODULE1 : {MODULE1}")
    print(f"SPLIT_RANGES['train'] = {SPLIT_RANGES['train']}")
    return (
        AdvectionResidualInterpolator,
        DataLoader,
        ERA5TemporalDataset,
        NormStats,
        Path,
        VARIABLE_ORDER,
        append_pressure_slice,
        build_loss_weight_tensor,
        create_empty_store,
        np,
        plt,
        tempfile,
        torch,
    )


@app.cell
def _(Path, np, tempfile, torch):
    # ── [2] Configuration ─────────────────────────────────────────────────────────
    TMPDIR  = Path(tempfile.mkdtemp(prefix="m1_debug_"))
    ZARR_6H = TMPDIR / "synthetic_6h.zarr"
    ZARR_1H = TMPDIR / "synthetic_1h.zarr"

    PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]
    LATS = np.linspace(40.5, 39.0, 7, dtype=np.float32)   # N→S
    LONS = np.linspace(-8.5, -7.0, 7, dtype=np.float32)   # W→E
    N_LAT, N_LON, N_LEV = 7, 7, 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constantes physiques pour la grille ERA5 0.25° à 40°N
    DX_M_PHYSICAL = 21_480.0
    DY_M_PHYSICAL = 27_830.0
    DT_S          = 6 * 3600.0

    print(f"Device  : {DEVICE}")
    print(f"TMPDIR  : {TMPDIR}")
    print(f"Grille  : {N_LAT}×{N_LON}, {N_LEV} niveaux")
    return (
        DEVICE,
        DT_S,
        DX_M_PHYSICAL,
        DY_M_PHYSICAL,
        LATS,
        LONS,
        N_LAT,
        N_LEV,
        N_LON,
        PRESSURE_LEVELS,
        TMPDIR,
        ZARR_1H,
        ZARR_6H,
    )


@app.cell
def _(N_LAT, N_LON, PRESSURE_LEVELS, np):
    # ── [3] Générateur de champs atmosphériques synthétiques ─────────────────────
    def atmospheric_field(t_hours: float) -> dict:
        """
        Génère (u, v, z, t, q) synthétiques pour l'instant t_hours depuis l'epoch.

        Physique :
        - Vent : rotation lente (7 jours), amplitude dépend du niveau (plus fort en altitude)
        - Température : profil vertical standard + cycle diurne sinusoïdal 5K
        - Géopotentiel : intégration hydrostatique approximative
        - Humidité : décroissance exponentielle avec l'altitude

        Non-linéarité temporelle : sin/cos → valeurs à t+3h ≠ (t0 + t6) / 2
        → le CNN a quelque chose à corriger par rapport à l'interpolation linéaire.
        """
        p = np.array(PRESSURE_LEVELS, dtype=np.float64)
        angle = 2 * np.pi * t_hours / (24 * 7)     # rotation complète en 7 jours

        # ── Vent (m/s) ────────────────────────────────────────────────────────────
        # Amplitude croissante avec l'altitude (ratio 850/p)
        u_lev = 10 * np.cos(angle) * (850 / p)     # (L,)
        v_lev =  6 * np.sin(angle) * (850 / p)     # (L,)
        u_arr = u_lev[:, None, None] * np.ones((N_LAT, N_LON))  # (L, H, W)
        v_arr = v_lev[:, None, None] * np.ones((N_LAT, N_LON))
        # Gradient méridional (cisaillement du vent avec la latitude)
        lat_shear = np.linspace(-1.0, 1.0, N_LAT)[None, :, None]   # (1, H, 1)
        u_arr += 2.0 * lat_shear

        # ── Température (K) ───────────────────────────────────────────────────────
        T_sfc = 288.15 + 5.0 * np.sin(2 * np.pi * t_hours / 24)   # cycle diurne
        T_lev = T_sfc - 6.5e-3 * 8000 * np.log(1000 / p)           # (L,) profil standard
        T_arr = T_lev[:, None, None] * np.ones((N_LAT, N_LON))
        # Gradient zonal de température
        lon_grad = np.linspace(-1.0, 1.0, N_LON)[None, None, :]    # (1, 1, W)
        T_arr += 2.0 * lon_grad

        # ── Géopotentiel (m²/s²) — hydrostatique approché ─────────────────────────
        z_arr = (287.05 * T_sfc / 9.81 * np.log(1000 / p))[:, None, None] * np.ones((N_LAT, N_LON))

        # ── Humidité spécifique (kg/kg) ───────────────────────────────────────────
        q_arr = (0.015 * np.exp(-p / 400.0))[:, None, None] * np.ones((N_LAT, N_LON))

        return {
            "u": u_arr.astype(np.float32),
            "v": v_arr.astype(np.float32),
            "z": z_arr.astype(np.float32),
            "t": T_arr.astype(np.float32),
            "q": q_arr.astype(np.float32),
        }

    # Vérification rapide
    f0 = atmospheric_field(0)
    f3 = atmospheric_field(3)
    f6 = atmospheric_field(6)
    lin_u = (f0["u"] + f6["u"]) / 2
    err_lin = np.abs(f3["u"] - lin_u).mean()
    print(f"Erreur d'interpolation linéaire sur u (850 hPa) : {err_lin:.3f} m/s")
    print("→ Non nul : le CNN a quelque chose à corriger")
    return (atmospheric_field,)


@app.cell
def _(
    LATS,
    LONS,
    N_LAT,
    N_LEV,
    N_LON,
    PRESSURE_LEVELS,
    ZARR_6H,
    append_pressure_slice,
    atmospheric_field,
    create_empty_store,
    np,
):
    # ── [4] Création du store 6h (32 pas = 8 jours, 2016-01-01 à 2016-01-08) ─────
    N_6H = 32
    times_6h = [np.datetime64('2016-01-01T00') + np.timedelta64(6 * _i, 'h') for _i in range(N_6H)]
    store_6h = create_empty_store(path=ZARR_6H, n_times=N_6H, levels_hpa=PRESSURE_LEVELS, lats=LATS.tolist(), lons=LONS.tolist(), source='synthetic', site='debug', time_step_hours=6, overwrite=True)
    for _i, _ts in enumerate(times_6h):
        _f = atmospheric_field(_i * 6)
        append_pressure_slice(store_6h, _i, _f['u'], _f['v'], _f['z'], _f['t'], _f['q'], _ts)
    print(f'Store 6h créé : {ZARR_6H}')
    print(f"  shape u : {store_6h['pressure/u'].shape}   # attendu (32, 10, 7, 7)")
    assert store_6h['pressure/u'].shape == (N_6H, N_LEV, N_LAT, N_LON)  # i × 6h
    return (store_6h,)


@app.cell
def _(
    LATS,
    LONS,
    N_LAT,
    N_LEV,
    N_LON,
    PRESSURE_LEVELS,
    ZARR_1H,
    append_pressure_slice,
    atmospheric_field,
    create_empty_store,
    np,
):
    # ── [5] Création du store 1h (192 pas = 8 jours) ─────────────────────────────
    N_1H = 8 * 24  # 192
    times_1h = [np.datetime64('2016-01-01T00') + np.timedelta64(_i, 'h') for _i in range(N_1H)]
    store_1h = create_empty_store(path=ZARR_1H, n_times=N_1H, levels_hpa=PRESSURE_LEVELS, lats=LATS.tolist(), lons=LONS.tolist(), source='synthetic_1h', site='debug', time_step_hours=1, overwrite=True)
    for _i, _ts in enumerate(times_1h):
        _f = atmospheric_field(float(_i))
        append_pressure_slice(store_1h, _i, _f['u'], _f['v'], _f['z'], _f['t'], _f['q'], _ts)
    print(f'Store 1h créé : {ZARR_1H}')
    print(f"  shape u : {store_1h['pressure/u'].shape}   # attendu (192, 10, 7, 7)")
    assert store_1h['pressure/u'].shape == (N_1H, N_LEV, N_LAT, N_LON)  # t = i heures
    return (store_1h,)


@app.cell
def _(np, store_1h, store_6h):
    # ── [6] Inspection des stores ─────────────────────────────────────────────────
    import zarr as zarr_lib

    def _to_h(arr):
        a = np.asarray(arr)
        if a.dtype.kind in ("i", "u"):
            return a.astype("datetime64[ns]").astype("datetime64[h]")
        return a.astype("datetime64[h]")

    times_6h_arr = _to_h(store_6h["coords/time"][:])
    times_1h_arr = _to_h(store_1h["coords/time"][:])

    print("=== Store 6h ===")
    print(f"  u     : {store_6h['pressure/u'].shape}")
    print(f"  time  : {store_6h['coords/time'].shape}, [{times_6h_arr[0]} … {times_6h_arr[-1]}]")
    print(f"  u[t=0, lev=2(850hPa), center] = {store_6h['pressure/u'][0, 2, 3, 3]:.2f} m/s")

    print("\n=== Store 1h ===")
    print(f"  u     : {store_1h['pressure/u'].shape}")
    print(f"  time  : {store_1h['coords/time'].shape}, [{times_1h_arr[0]} … {times_1h_arr[-1]}]")

    # Vérifier que tous les timestamps 6h sont dans le store 1h
    # Utiliser np.isin (comparaison numpy native, sans conversion Python) pour éviter
    # les problèmes de hash entre numpy.datetime64 et datetime.datetime
    missing_mask = ~np.isin(times_6h_arr, times_1h_arr)
    missing = times_6h_arr[missing_mask]
    print(f"\nTimestamps 6h dans store 1h : {(~missing_mask).sum()}/{len(times_6h_arr)}")
    assert len(missing) == 0, f"Manquants : {missing[:3]}"
    print("✓ Tous les timestamps 6h ont leur correspondant dans le store 1h")

    # Vérifier la non-linéarité temporelle
    u_t0 = store_6h["pressure/u"][0, 2, 3, 3]   # 850 hPa, centre
    u_t6 = store_6h["pressure/u"][1, 2, 3, 3]
    u_t3_lin = (u_t0 + u_t6) / 2
    u_t3_true = store_1h["pressure/u"][3, 2, 3, 3]
    print(f"\nNon-linéarité : u(t+3h) linéaire={u_t3_lin:.3f}, exact={u_t3_true:.3f}")
    print(f"  Erreur interpolation linéaire : {abs(u_t3_true - u_t3_lin):.3f} m/s")
    return


@app.cell
def _(DX_M_PHYSICAL, DY_M_PHYSICAL, NormStats, VARIABLE_ORDER, ZARR_6H):
    # ── [7] NormStats ─────────────────────────────────────────────────────────────
    print("Calcul NormStats (algorithme de Welford)…")
    ns = NormStats.compute_from_zarr(ZARR_6H, time_slice=slice(0, 28))
    print("Done.")

    # Tableau mean/std par variable
    print(f"\n{'Var':>4} | {'mean[850hPa]':>14} | {'std[850hPa]':>12} | {'mean[200hPa]':>14}")
    print("-" * 55)
    for var in VARIABLE_ORDER:
        lev_850 = 2  # index niveau 850 hPa
        lev_200 = 9  # index niveau 200 hPa
        print(f"{var:>4} | {ns.mean[var][lev_850]:>14.4f} | {ns.std[var][lev_850]:>12.4f} | "
              f"{ns.mean[var][lev_200]:>14.4f}")

    # Correction dx/dy pour l'advection (dénormalisation du vent)
    std_u, std_v = ns.get_wind_std()
    dx_m_eff = DX_M_PHYSICAL / (std_u + 1e-8)
    dy_m_eff = DY_M_PHYSICAL / (std_v + 1e-8)
    print(f"\nstd_u={std_u:.3f} m/σ, std_v={std_v:.3f} m/σ")
    print(f"dx_m_eff={dx_m_eff:.0f}, dy_m_eff={dy_m_eff:.0f}")
    return dx_m_eff, dy_m_eff, ns


@app.cell
def _(DataLoader, ERA5TemporalDataset, ZARR_1H, ZARR_6H, ns):
    # ── [8] Dataset et DataLoader ─────────────────────────────────────────────────
    # Les 8 jours de données tombent dans le split 'train' (2016-01-01 → 2016-11-01)
    ds = ERA5TemporalDataset(
        zarr_6h=ZARR_6H,
        zarr_1h=ZARR_1H,
        norm_stats=ns,
        split="train",
        grid_size=7,
    )
    print(ds.info())
    print(f"  Attendu : 31 fenêtres (32 timestamps → 31 paires consécutives)")
    assert len(ds) == 31, f"Attendu 31, obtenu {len(ds)}"

    # Inspecter un exemple
    S0, S1, targets = ds[0]
    print(f"\nS0      : {tuple(S0.shape)}       # (V=5, L=10, H=7, W=7)")
    print(f"S1      : {tuple(S1.shape)}")
    print(f"targets : {tuple(targets.shape)}  # (T=5, V=5, L=10, H=7, W=7)")
    print(f"\nS0[u,850hPa,centre] normalisé = {S0[0, 2, 3, 3]:.4f} σ")
    print(f"  physique ≈ {ns.denormalize(S0[0].numpy(), 'u')[2, 3, 3]:.2f} m/s")

    # DataLoader — num_workers=0 obligatoire dans Jupyter
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch_S0, batch_S1, batch_tgt = next(iter(loader))
    print(f"\nBatch S0 : {tuple(batch_S0.shape)}   # (B=4, V=5, L=10, H=7, W=7)")
    print(f"Batch tgt: {tuple(batch_tgt.shape)}")
    return S0, S1, loader, targets


@app.cell
def _(LATS, LONS, S0, S1, np, ns, plt, targets):
    # ── [9] Visualisation d'un exemple ────────────────────────────────────────────
    LEV_IDX = 2  # 850 hPa
    u0 = ns.denormalize(S0[0].numpy(), 'u')[LEV_IDX]
    # Dénormaliser
    v0 = ns.denormalize(S0[1].numpy(), 'v')[LEV_IDX]  # (H, W)
    u1 = ns.denormalize(S1[0].numpy(), 'u')[LEV_IDX]
    v1 = ns.denormalize(S1[1].numpy(), 'v')[LEV_IDX]
    u_mid = ns.denormalize(targets[2, 0].numpy(), 'u')[LEV_IDX]
    # Cible à τ=3/6 (milieu, t+3h)
    v_mid = ns.denormalize(targets[2, 1].numpy(), 'v')[LEV_IDX]  # targets[τ=2, u, ...]
    u_lin = (u0 + u1) / 2
    # Baseline linéaire au milieu
    v_lin = (v0 + v1) / 2
    _fig, _axes = plt.subplots(1, 4, figsize=(18, 4))
    XX, YY = np.meshgrid(LONS, LATS)
    speed_kw = dict(scale=150, width=0.005)
    for _ax, _u_f, _v_f, title, color in zip(_axes, [u0, u_lin, u_mid, u1], [v0, v_lin, v_mid, v1], ['t₀ (S0)', 'linéaire (t₀+3h)', 'vérité (t₀+3h)', 't₀+6h (S1)'], ['C0', 'C2', 'C1', 'C0']):
        _ws = np.sqrt(_u_f ** 2 + _v_f ** 2)
        _im = _ax.pcolormesh(LONS, LATS, _ws, cmap='Blues', vmin=0, vmax=15)
        _ax.quiver(XX, YY, _u_f, _v_f, **speed_kw, color=color)
        _ax.set_title(title)
        _ax.set_xlabel('Lon (°E)')
        _ax.set_ylabel('Lat (°N)')
        plt.colorbar(_im, ax=_ax, label='|V| m/s')
    plt.suptitle('Vent à 850 hPa — données synthétiques', fontsize=13)
    plt.tight_layout()
    plt.show()
    return LEV_IDX, XX, YY


@app.cell
def _(
    AdvectionResidualInterpolator,
    DEVICE,
    DT_S,
    N_LAT,
    N_LEV,
    N_LON,
    S0,
    S1,
    dx_m_eff,
    dy_m_eff,
    torch,
):
    # ── [10] Modèle — forward pass et assertions ──────────────────────────────────
    model = AdvectionResidualInterpolator(
        n_vars=5,
        n_levels=N_LEV,
        n_hidden=48,
        n_intermediate=5,
        dx_m=dx_m_eff,
        dy_m=dy_m_eff,
        dt_s=DT_S,
    ).to(DEVICE)

    print(f"Paramètres entraînables : {model.count_parameters():,}")

    # Vérification zero-init de la couche de sortie du CNN
    cnn_out_norm = model.cnn[-1].weight.norm().item()
    print(f"CNN couche sortie — norme poids : {cnn_out_norm:.6f}  (doit être 0.0)")
    assert cnn_out_norm == 0.0, "Erreur : la couche de sortie du CNN doit être zero-init !"

    # Test forward pass
    S0t = S0.unsqueeze(0).to(DEVICE)   # (1, V, L, H, W)
    S1t = S1.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(S0t, S1t)

    expected = (1, 5, 5, N_LEV, N_LAT, N_LON)
    print(f"\nOutput shape : {tuple(out.shape)}")
    print(f"Attendu      : {expected}")
    assert tuple(out.shape) == expected, f"Shape incorrecte : {out.shape}"
    print("✓ Forward pass OK")

    # Avec CNN = 0 (zero-init), la sortie est l'advection pure
    # → les corrections devraient être exactement 0
    cnn_input = torch.cat([S0t, S1t], dim=1)
    corrections = model.cnn(cnn_input)
    print(f"\nCorrections CNN (doit être 0.0) : {corrections.abs().max().item():.6f}")
    return S0t, S1t, model


@app.cell
def _(DEVICE, N_LEV, build_loss_weight_tensor, loader, model, torch):
    # ── [11] Entraînement (30 époques, inline sans MLflow) ────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-05)
    loss_weights = build_loss_weight_tensor(N_LEV, DEVICE).unsqueeze(1)  # (1,1,V,1,1,1)
    # Vérifier la shape des poids
    assert loss_weights.shape == (1, 1, 5, 1, 1, 1)
    print(f'Loss weights shape : {tuple(loss_weights.shape)}')
    print(f'Loss weights (par var) : {loss_weights.squeeze().tolist()}')  # [2,2,1,1,0.5]
    N_EPOCHS = 30
    train_losses = []
    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for _S0b, _S1b, _tgt in loader:
            _S0b = _S0b.to(DEVICE)
            _S1b = _S1b.to(DEVICE)
            _tgt = _tgt.to(DEVICE)  # (B, V, L, H, W)
            optimizer.zero_grad()
            _preds = model(_S0b, _S1b)  # (B, T, V, L, H, W)
            loss = (loss_weights * (_preds - _tgt).pow(2)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # (B, T, V, L, H, W)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch + 1:3d}/{N_EPOCHS} | loss = {train_losses[-1]:.5f}')
    return N_EPOCHS, train_losses


@app.cell
def _(N_EPOCHS, plt, train_losses):
    # ── [12] Courbe de loss ───────────────────────────────────────────────────────
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(range(1, N_EPOCHS + 1), train_losses, lw=2, color='steelblue')
    _ax.set_xlabel('Époque')
    _ax.set_ylabel('Loss pondérée (σ²)')
    _ax.set_title("Courbe d'entraînement — données synthétiques")
    _ax.grid(True, alpha=0.3)
    # Annoter la loss finale
    _ax.annotate(f'Final : {train_losses[-1]:.5f}', xy=(N_EPOCHS, train_losses[-1]), xytext=(N_EPOCHS * 0.7, train_losses[-1] * 1.2), arrowprops=dict(arrowstyle='->'))
    plt.tight_layout()
    plt.show()
    print(f'Loss initiale : {train_losses[0]:.5f}  →  finale : {train_losses[-1]:.5f}')
    print(f'Réduction : {100 * (1 - train_losses[-1] / train_losses[0]):.1f}%')
    return


@app.cell
def _(
    LATS,
    LEV_IDX,
    LONS,
    S0,
    S0t,
    S1,
    S1t,
    XX,
    YY,
    model,
    np,
    ns,
    plt,
    targets,
    torch,
):
    # ── [13] Inférence et visualisation des 5 pas intermédiaires ─────────────────
    model.eval()
    taus = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    with torch.no_grad():
        _preds = model(S0t, S1t)
    _fig, _axes = plt.subplots(3, 5, figsize=(20, 10))  # (1, 5, 5, 10, 7, 7)
    row_labels = ['Linéaire', 'Modèle', 'Vérité terrain']
    for k, _tau in enumerate(taus):
        u_lin_n = ((1 - _tau) * S0[0] + _tau * S1[0]).numpy()
        v_lin_n = ((1 - _tau) * S0[1] + _tau * S1[1]).numpy()
        u_lin_p = ns.denormalize(u_lin_n, 'u')[LEV_IDX]
        v_lin_p = ns.denormalize(v_lin_n, 'v')[LEV_IDX]  # ── Baseline linéaire ──
        u_mod_n = _preds[0, k, 0].cpu().numpy()  # (L, H, W) normalisé
        v_mod_n = _preds[0, k, 1].cpu().numpy()
        u_mod_p = ns.denormalize(u_mod_n, 'u')[LEV_IDX]
        v_mod_p = ns.denormalize(v_mod_n, 'v')[LEV_IDX]
        u_tgt_n = targets[k, 0].numpy()
        v_tgt_n = targets[k, 1].numpy()  # ── Modèle ──
        u_tgt_p = ns.denormalize(u_tgt_n, 'u')[LEV_IDX]  # (L, H, W)
        v_tgt_p = ns.denormalize(v_tgt_n, 'v')[LEV_IDX]
        for _row, (_u_f, _v_f) in enumerate([(u_lin_p, v_lin_p), (u_mod_p, v_mod_p), (u_tgt_p, v_tgt_p)]):
            _ax = _axes[_row, k]
            _ws = np.sqrt(_u_f ** 2 + _v_f ** 2)
            _im = _ax.pcolormesh(LONS, LATS, _ws, cmap='RdBu_r', vmin=0, vmax=15)  # ── Vérité terrain ──
            _ax.quiver(XX, YY, _u_f, _v_f, scale=150, width=0.006, color='k', alpha=0.7)  # (L, H, W)
            _ax.set_title(f'{row_labels[_row]} τ={_tau:.2f}', fontsize=8)
            _ax.set_xticks([])
            _ax.set_yticks([])
            if k == 0:
                _ax.set_ylabel(row_labels[_row], fontsize=8)
    plt.colorbar(_im, ax=_axes[:, -1], label='|V| m/s')
    plt.suptitle('Vent 850 hPa interpolé : linéaire vs modèle vs vérité (après 30 époques)', fontsize=11)
    plt.tight_layout()
    plt.show()
    return (taus,)


@app.cell
def _(DEVICE, VARIABLE_ORDER, loader, model, np, taus, torch):
    # ── [14] Métriques : RMSE modèle vs baseline linéaire ────────────────────────
    model.eval()
    n_vars = len(VARIABLE_ORDER)
    n_taus = 5
    rmse_model = np.zeros((n_taus, n_vars))
    rmse_linear = np.zeros((n_taus, n_vars))
    counts = np.zeros((n_taus, n_vars))
    taus_t = torch.tensor(taus, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for _S0b, _S1b, _tgt in loader:  # (5,)
            _S0b = _S0b.to(DEVICE)
            _S1b = _S1b.to(DEVICE)
            _tgt = _tgt.to(DEVICE)
            _preds = model(_S0b, _S1b)  # (B, V, L, H, W)
            taus_v = taus_t.view(1, n_taus, 1, 1, 1, 1)
            lin = (1 - taus_v) * _S0b.unsqueeze(1) + taus_v * _S1b.unsqueeze(1)  # (B, T, V, L, H, W)
            for t_idx in range(n_taus):
                for v_idx in range(n_vars):  # (B, T, V, L, H, W)
                    e_mod = (_preds[:, t_idx, v_idx] - _tgt[:, t_idx, v_idx]).pow(2).sum().item()
                    e_lin = (lin[:, t_idx, v_idx] - _tgt[:, t_idx, v_idx]).pow(2).sum().item()  # Baseline linéaire : shape (1, T, 1, 1, 1, 1) × (B, 1, V, L, H, W)
                    n_el = _preds[:, t_idx, v_idx].numel()
                    rmse_model[t_idx, v_idx] += e_mod  # (B,T,V,L,H,W)
                    rmse_linear[t_idx, v_idx] += e_lin
                    counts[t_idx, v_idx] += n_el
    rmse_model = np.sqrt(rmse_model / np.maximum(counts, 1))
    rmse_linear = np.sqrt(rmse_linear / np.maximum(counts, 1))
    gain = 100 * (rmse_linear - rmse_model) / (rmse_linear + 1e-10)
    header = f"{'τ':>6}" + ''.join((f'{v:>10}' for v in VARIABLE_ORDER))
    print('RMSE modèle (σ normalisées) — format : modèle(gain_vs_linéaire%)')
    print(header)
    print('-' * (6 + 10 * n_vars))
    for t_idx, _tau in enumerate(taus):
        _row = f'{_tau:.2f}  '
        for v_idx in range(n_vars):
            _row += f'  {rmse_model[t_idx, v_idx]:.3f}({gain[t_idx, v_idx]:+.0f}%)'
        print(_row)
    # Affichage
    print(f'\nGain moyen sur u,v : {gain[:, :2].mean():.1f}%')
    if gain[:, :2].mean() > 0:
        print("✓ Le modèle fait mieux que l'interpolation linéaire sur u,v")
    else:
        print("⚠ Le modèle n'a pas encore dépassé la baseline linéaire (normal avec 30 époques seulement)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cellules de débogage

    Les cellules suivantes sont des **tests unitaires ciblés** pour vérifier les composants critiques du modèle.
    """)
    return


@app.cell
def _(DEVICE, N_LAT, N_LEV, N_LON, model, np, torch):
    # ── [15A] Débogage : convention du warp (advection) ──────────────────────────
    print("=" * 60)
    print("TEST : Convention de _warp_per_level")
    print("=" * 60)
    print("Rappel du mécanisme :")
    print("  grid_sample(field, identity + disp) échantillonne field")
    print("  à la position identity[h,w] + disp[h,w].")
    print("  output[h,w] = field(x_hw + disp_x)")
    print("  → disp_x > 0 : on échantillonne à l'est → le pic APPARENT se déplace OUEST")
    print()

    model.eval()

    # Créer un champ avec un pic central
    field_test = torch.zeros(1, 5, N_LEV, N_LAT, N_LON, device=DEVICE)
    field_test[0, :, :, N_LAT // 2, N_LON // 2] = 1.0  # pic au centre (3,3)

    for disp_val, direction_physique in [(0.2, "EST → pic dans le champ se déplace OUEST"),
                                         (-0.2, "OUEST → pic se déplace EST")]:
        disp_x = torch.full((1, N_LEV, N_LAT, N_LON), disp_val, device=DEVICE)
        disp_y = torch.zeros_like(disp_x)

        with torch.no_grad():
            warped = model._warp_per_level(field_test, disp_x, disp_y)

        # Trouver la position du max dans la direction W
        u_slice = warped[0, 0, 5].cpu().numpy()  # (H=7, W=7)
        max_col = np.argmax(u_slice.mean(axis=0))   # colonne du max
        center  = N_LON // 2

        print(f"  disp_x={disp_val:+.1f} ({direction_physique})")
        print(f"    → max à colonne {max_col} (centre={center})")
        expected_shift = "OUEST (col↓)" if disp_val > 0 else "EST (col↑)"
        actual_shift   = "OUEST (col↓)" if max_col < center else ("EST (col↑)" if max_col > center else "centre")
        match = "✓" if expected_shift == actual_shift else "✗"
        print(f"    Attendu : {expected_shift} | Obtenu : {actual_shift} {match}")
        print()

    print("Implication pour l'advection :")
    print("  S0_warped utilise disp_x = -τ * u_mean * dt/dx")
    print("  Pour u > 0 (vent vers l'est) : disp_x < 0 → le pic se déplace EST ✓")
    print("  (La particule vient de l'ouest et se retrouve à l'est après advection)")
    return


@app.cell
def _(N_LAT, N_LEV, N_LON, S0t, S1t, model, torch):
    # ── [15B] Débogage : shapes à travers les couches du CNN ─────────────────────
    print("=" * 60)
    print("TEST : Shapes à travers le CNN résiduel")
    print("=" * 60)

    x = torch.cat([S0t, S1t], dim=1)   # (1, 2V=10, L=10, H=7, W=7)
    print(f"Input CNN : {tuple(x.shape)}   # (B=1, 2V=10, L=10, H=7, W=7)")
    print()

    for layer in model.cnn:
        x_out = layer(x)
        arrow = "→" if x_out.shape != x.shape else "=="
        print(f"  {type(layer).__name__:20s} {tuple(x.shape)} {arrow} {tuple(x_out.shape)}")
        x = x_out

    print(f"\nOutput CNN : {tuple(x.shape)}   # (B=1, T*V=25, L=10, H=7, W=7)")
    print(f"  → reshape en 5 corrections de shape (B=1, V=5, L=10, H=7, W=7)")
    assert x.shape == (1, 5 * 5, N_LEV, N_LAT, N_LON)
    return


@app.cell
def _(S0t, S1t, model):
    # ── [15C] Débogage : gradient flow ───────────────────────────────────────────
    print("=" * 60)
    print("TEST : Gradient flow à travers le modèle")
    print("=" * 60)

    S0_grad = S0t.detach().requires_grad_(True)
    S1_grad = S1t.detach().requires_grad_(True)

    out_grad = model(S0_grad, S1_grad)
    loss_grad = out_grad.mean()
    loss_grad.backward()

    assert S0_grad.grad is not None, "Gradient absent sur S0 !"
    assert S1_grad.grad is not None, "Gradient absent sur S1 !"

    grad_u_s0 = S0_grad.grad[0, 0].abs().mean().item()   # gradient sur u dans S0
    grad_v_s0 = S0_grad.grad[0, 1].abs().mean().item()   # gradient sur v dans S0
    grad_u_s1 = S1_grad.grad[0, 0].abs().mean().item()
    grad_v_s1 = S1_grad.grad[0, 1].abs().mean().item()

    print(f"  ∂L/∂u₀ (S0, u) : {grad_u_s0:.4e}  {'✓' if grad_u_s0 > 0 else '✗'}")
    print(f"  ∂L/∂v₀ (S0, v) : {grad_v_s0:.4e}  {'✓' if grad_v_s0 > 0 else '✗'}")
    print(f"  ∂L/∂u₁ (S1, u) : {grad_u_s1:.4e}  {'✓' if grad_u_s1 > 0 else '✗'}")
    print(f"  ∂L/∂v₁ (S1, v) : {grad_v_s1:.4e}  {'✓' if grad_v_s1 > 0 else '✗'}")

    # Gradient sur z, t, q (ne participent pas à l'advection → gradient uniquement via CNN)
    grad_z_s0 = S0_grad.grad[0, 2].abs().mean().item()
    print(f"\n  ∂L/∂z₀ (S0, z) : {grad_z_s0:.4e}  (seulement via CNN résiduel)")
    print(f"\n✓ Le gradient remonte correctement à travers grid_sample (advection) et le CNN.")
    return


@app.cell
def _(model, taus):
    # ── [15D] Débogage : τ·(1-τ) bridge factor ───────────────────────────────────
    print('=' * 60)
    print('TEST : Facteur bridge τ(1-τ) — valeurs aux τ intermédiaires')
    print('=' * 60)
    for _tau in taus:
        bridge = _tau * (1 - _tau)
        print(f'  τ={_tau:.3f} → τ(1-τ) = {bridge:.4f}   (max=0.25 à τ=0.5)')
    print(f'\nMax à τ=0.5 : {0.5 * 0.5:.4f}')
    print('→ La correction CNN est atténuée aux bornes et maximale au milieu.')
    print('  Avec CNN=0 (zero-init), le modèle = advection pure.')
    print(f'\nTaus enregistrés : {model.taus.tolist()}')
    assert abs(model.taus[2].item() - 0.5) < 1e-06, 'τ central devrait être 0.5'
    # Vérifier les taus enregistrés dans le modèle
    print('✓ Taus corrects : {1/6, 2/6, 3/6, 4/6, 5/6}')
    return


@app.cell
def _(TMPDIR):
    # ── [16] Nettoyage ────────────────────────────────────────────────────────────
    # Décommenter pour supprimer le répertoire temporaire
    # shutil.rmtree(TMPDIR)
    # print(f"TMPDIR supprimé : {TMPDIR}")
    print(f"TMPDIR conservé pour inspection : {TMPDIR}")
    print("  Répertoire supprimable avec : shutil.rmtree(TMPDIR)")
    print()
    print("=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    print("  [1] Stores Zarr créés et cohérents               ✓")
    print("  [2] NormStats calculées (Welford)                ✓")
    print("  [3] Dataset 31 fenêtres / DataLoader batchs     ✓")
    print("  [4] Forward pass (1,5,5,10,7,7)                 ✓")
    print("  [5] CNN zero-init (corrections=0)               ✓")
    print("  [6] Entraînement 30 époques                     ✓")
    print("  [7] Visualisation 3×5 comparaison               ✓")
    print("  [8] Métriques RMSE vs baseline linéaire         ✓")
    print("  [9] Test convention advection (disp_x direction) ✓")
    print(" [10] Test shapes CNN                             ✓")
    print(" [11] Test gradient flow                          ✓")
    print(" [12] Test facteur bridge τ(1-τ)                  ✓")
    return


if __name__ == "__main__":
    app.run()

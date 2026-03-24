"""
calculate_dhdt.py
=================
Cálculo de dh/dt por tile — ajuste espaço-temporal.

Modelo:
  h(x,y,t) = p0 + p1·Δt + p2·½Δt²
            + ax·dx_n + ay·dy_n + axy·dx_n·dy_n
            + axx·dx_n² + ayy·dy_n²

  p0  = elevação de referência em T_REF  (m)
  p1  = dh/dt                            (m/ano)  ← resultado principal
  p2  = d²h/dt²  aceleração              (m/ano²)

Calibrado para:
  Região  : lon (-106.5, -103.5)  lat (-75.5, -74.5)
  Pontos  : ~1,17 M  em 7 tiles
  Período : 2019.4 – 2025.6  (JJA, ~6 ciclos anuais)
"""

import numpy as np
import h5py
from pathlib import Path
from pyproj import Transformer
from scipy import linalg
from scipy.spatial import cKDTree
import gc
from tqdm import tqdm

# ============================================================
# CAMINHOS
# ============================================================
BASE      = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")
TILES_DIR = BASE / "tiles"
DHDT_DIR  = BASE / "dhdt"

# ============================================================
# PARÂMETROS — calibrados para sua região e densidade de dados
# ============================================================

# Raio de busca: precisa cobrir pelo menos 1 track vizinho.
# Separação entre tracks ICESat-2 a ~75°S ≈ 20–28 km → 25 km cobre com folga.
SEARCH_RADIUS = 25_000.0   # metros

# Mínimo de pontos para aceitar um ajuste.
# Com 6 anos × ~3 passes JJA por track ≈ 18 obs por vizinhança bem coberta.
MIN_POINTS = 20

# Ordem do modelo espacial (2 = quadrático → captura curvatura topográfica)
POLY_ORDER = 2

# Ordem temporal (2 = quadrático → estima aceleração p2)
TEMP_ORDER = 2

# Iterações de rejeição de outliers
MAX_ITER = 5

# Limiar de rejeição (n-sigma)
N_SIGMA = 3.0

# Limite absoluto de resíduo
RESID_LIM = 5.0    # metros

# Limite físico de dh/dt — Thwaites pode chegar a ~5 m/ano
RATE_LIM = 10.0    # m/ano

# Limite de aceleração plausível
P2_LIM = 1.0       # m/ano²

# Span temporal mínimo — precisa de pelo menos 2 anos para separar p1 de p0
DT_MIN = 1.5       # anos

# Ano de referência — centro do período 2019–2025
# p0 será a elevação estimada em julho de 2022
T_REF = 2022.5

# Resolução da grade de nós dentro de cada tile.
# 2 km → ~625 nós num tile de 50×50 km (rápido e suficiente para sua área)
GRID_RES = 2_000   # metros

# Ponderar pelo erro formal s_elv do ATL06?
USE_WEIGHTS = True

# Pular tiles que já têm arquivo de saída? (útil para retomar após interrupção)
SKIP_DONE = True

# ============================================================
to_lonlat = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)


# ============================================================
# FUNÇÕES
# ============================================================

def mad_std(r):
    """Desvio padrão robusto via MAD."""
    return 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))


def build_A(dx, dy, dt, po=2, to=2):
    """
    Matriz de design com normalização de dx/dy.
    Colunas: [1, dt, ½dt², dx_n, dy_n, dx_n·dy_n, dx_n², dy_n²]
    """
    n  = len(dx)
    sx = np.std(dx) if np.std(dx) > 1.0 else 1.0
    sy = np.std(dy) if np.std(dy) > 1.0 else 1.0
    dxn = dx / sx
    dyn = dy / sy

    cols = [np.ones(n), dt]
    if to >= 2:
        cols.append(0.5 * dt**2)
    if po >= 1:
        cols += [dxn, dyn, dxn * dyn]
    if po >= 2:
        cols += [dxn**2, dyn**2]

    return np.column_stack(cols)


def lstsq_iter(A, z, w=None, n_iter=5, n_sigma=3.0, rlim=5.0):
    """
    Mínimos quadrados ponderados com rejeição iterativa de outliers.
    Retorna (xhat, ehat, mask, rmse).
    """
    n, m = A.shape
    mask = np.ones(n, dtype=bool)
    xhat = ehat = None
    rmse = np.nan

    for _ in range(n_iter):

        Af = A[mask]
        zf = z[mask]
        nf = mask.sum()

        if nf < m + 2:
            break

        if w is not None:
            wf  = w[mask]
            wf  = wf / (wf.mean() + 1e-30)   # normalizar para não distorcer escala
            W   = np.sqrt(wf)
            Aw  = Af * W[:, None]
            zw  = zf * W
        else:
            Aw = Af
            zw = zf

        try:
            xhat_new, _, rank, _ = linalg.lstsq(Aw, zw, check_finite=False)
        except (linalg.LinAlgError, ValueError):
            break

        if rank < m:
            break

        xhat = xhat_new
        res  = zf - Af @ xhat
        rmse = float(np.std(res))

        # Erro formal via covariância
        try:
            AtA  = Aw.T @ Aw
            sig2 = float(np.sum(res**2) / max(nf - m, 1))
            ehat = np.sqrt(np.abs(np.diag(linalg.inv(AtA))) * sig2)
        except linalg.LinAlgError:
            ehat = np.full(m, np.nan)

        # Rejeitar outliers
        mad  = mad_std(res)
        if mad < 1e-10:
            break
        bad  = (np.abs(res) > n_sigma * mad) | (np.abs(res) > rlim)
        if not bad.any():
            break
        mask[np.where(mask)[0][bad]] = False

    return xhat, ehat, mask, rmse


def fit_node(xo, yo, ho, to, so, x0, y0):
    """
    Ajusta o modelo espaço-temporal num nó (x0, y0).
    Retorna dict com resultados ou None se inválido.
    """
    n = len(ho)
    if n < MIN_POINTS:
        return None

    dt_span = float(to.max() - to.min())
    if dt_span < DT_MIN:
        return None

    dx = xo - x0
    dy = yo - y0
    dt = to - T_REF

    # Pesos: 1 / s_elv²
    if USE_WEIGHTS:
        sv = np.where(so <= 0, np.median(so[so > 0]) if (so > 0).any() else 0.01, so)
        wc = 1.0 / (sv**2 + 1e-12)
    else:
        wc = None

    A = build_A(dx, dy, dt, po=POLY_ORDER, to=TEMP_ORDER)
    xhat, ehat, mask, rmse = lstsq_iter(
        A, ho, w=wc,
        n_iter=MAX_ITER, n_sigma=N_SIGMA, rlim=RESID_LIM
    )

    if xhat is None or mask.sum() < MIN_POINTS:
        return None

    p0_val = float(xhat[0])
    p1_val = float(xhat[1])
    p0_err = float(ehat[0]) if ehat is not None else np.nan
    p1_err = float(ehat[1]) if ehat is not None else np.nan

    # Filtro físico de taxa
    if abs(p1_val) > RATE_LIM:
        return None

    # Aceleração
    p2_val = p2_err = np.nan
    if TEMP_ORDER >= 2 and len(xhat) > 2:
        p2_val = float(xhat[2])
        p2_err = float(ehat[2]) if ehat is not None else np.nan
        if abs(p2_val) > P2_LIM:
            p2_val = p2_err = np.nan

    return {
        "p0":     p0_val,
        "p1":     p1_val,
        "p2":     p2_val,
        "p0_err": p0_err,
        "p1_err": p1_err,
        "p2_err": p2_err,
        "rmse":   float(rmse),
        "nobs":   int(mask.sum()),
        "tspan":  dt_span,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("CÁLCULO DE dh/dt  (fitsec)")
print("=" * 60)
print(f"  Raio de busca  : {SEARCH_RADIUS/1000:.0f} km")
print(f"  Grade interna  : {GRID_RES} m")
print(f"  Mín. pontos    : {MIN_POINTS}")
print(f"  Span mín.      : {DT_MIN} anos")
print(f"  T_REF          : {T_REF}")
print(f"  Limite |dh/dt| : {RATE_LIM} m/ano")
print(f"  Limite |d²h/dt²|: {P2_LIM} m/ano²")
print(f"  Ponderação     : {USE_WEIGHTS}")
print(f"  Retomar (skip) : {SKIP_DONE}")
print("=" * 60)

DHDT_DIR.mkdir(exist_ok=True, parents=True)

tile_files = sorted(TILES_DIR.glob("tile_*.h5"))
print(f"\nTiles encontrados: {len(tile_files)}")

if not tile_files:
    raise SystemExit("✗ Nenhum tile encontrado. Execute crop_and_retile.py primeiro.")

summary = []

for tile_path in tile_files:
    stem     = tile_path.stem
    out_path = DHDT_DIR / f"{stem}_dhdt.h5"

    print(f"\n{'─'*55}")
    print(f"Tile: {stem}")

    # Retomar se já processado
    if SKIP_DONE and out_path.exists():
        print(f"  ↩ Já processado, pulando.")
        continue

    # ------ Ler tile ------
    with h5py.File(tile_path, "r") as f:
        x_all   = f["x"][:]
        y_all   = f["y"][:]
        h_all   = f["h_elv"][:]    # elevação
        t_all   = f["t_year"][:]   # tempo em anos decimais
        s_all   = f["s_elv"][:]    # incerteza formal
        lon_all = f["lon"][:]
        lat_all = f["lat"][:]

    # Remover NaN
    ok    = ~(np.isnan(x_all) | np.isnan(y_all) |
              np.isnan(h_all) | np.isnan(t_all))
    x_all = x_all[ok];  y_all = y_all[ok]
    h_all = h_all[ok];  t_all = t_all[ok]
    s_all = s_all[ok];  lon_all = lon_all[ok];  lat_all = lat_all[ok]

    n_pts = len(h_all)
    print(f"  Pontos válidos : {n_pts:,}")

    if n_pts < MIN_POINTS * 3:
        print("  ⚠ Pontos insuficientes, pulando.")
        continue

    # Corrigir s_elv zerado/negativo
    med_s = float(np.median(s_all[s_all > 0])) if (s_all > 0).any() else 0.05
    s_all = np.where(s_all <= 0, med_s, s_all)

    # Span temporal do tile
    t_span_tile = float(t_all.max() - t_all.min())
    n_years     = len(np.unique(np.floor(t_all).astype(int)))
    print(f"  Span temporal  : {t_span_tile:.2f} anos  ({n_years} anos distintos)")

    if t_span_tile < DT_MIN:
        print("  ⚠ Span temporal insuficiente, pulando.")
        continue

    # ------ Grade de nós ------
    gx = np.arange(
        np.floor(x_all.min() / GRID_RES) * GRID_RES + GRID_RES / 2,
        np.ceil (x_all.max() / GRID_RES) * GRID_RES,
        GRID_RES
    )
    gy = np.arange(
        np.floor(y_all.min() / GRID_RES) * GRID_RES + GRID_RES / 2,
        np.ceil (y_all.max() / GRID_RES) * GRID_RES,
        GRID_RES
    )
    GX, GY  = np.meshgrid(gx, gy)
    nodes_x = GX.ravel()
    nodes_y = GY.ravel()
    n_nodes = len(nodes_x)
    print(f"  Grade interna  : {len(gx)} × {len(gy)} = {n_nodes:,} nós")

    # ------ KD-Tree ------
    tree = cKDTree(np.c_[x_all, y_all])

    # ------ Pré-alocar saída ------
    out_p0  = np.full(n_nodes, np.nan)
    out_p1  = np.full(n_nodes, np.nan)
    out_p2  = np.full(n_nodes, np.nan)
    out_e0  = np.full(n_nodes, np.nan)
    out_e1  = np.full(n_nodes, np.nan)
    out_e2  = np.full(n_nodes, np.nan)
    out_rms = np.full(n_nodes, np.nan)
    out_nob = np.zeros(n_nodes, dtype=np.int32)
    out_tsp = np.full(n_nodes, np.nan)

    # ------ Loop sobre nós ------
    n_empty = 0
    for ni in tqdm(range(n_nodes), desc="  Nós", mininterval=10):
        idx = tree.query_ball_point([nodes_x[ni], nodes_y[ni]], r=SEARCH_RADIUS)

        if len(idx) < MIN_POINTS:
            n_empty += 1
            continue

        idx = np.asarray(idx)
        res = fit_node(
            x_all[idx], y_all[idx], h_all[idx],
            t_all[idx], s_all[idx],
            nodes_x[ni], nodes_y[ni]
        )
        if res is None:
            continue

        out_p0[ni]  = res["p0"]
        out_p1[ni]  = res["p1"]
        out_p2[ni]  = res["p2"]
        out_e0[ni]  = res["p0_err"]
        out_e1[ni]  = res["p1_err"]
        out_e2[ni]  = res["p2_err"]
        out_rms[ni] = res["rmse"]
        out_nob[ni] = res["nobs"]
        out_tsp[ni] = res["tspan"]

    n_valid = int(np.sum(~np.isnan(out_p1)))
    print(f"\n  Nós válidos    : {n_valid:,} / {n_nodes:,} "
          f"({100*n_valid/n_nodes:.1f}%)")
    print(f"  Nós sem dados  : {n_empty:,}")

    if n_valid == 0:
        print("  ⚠ Nenhum nó válido.")
        continue

    # Estatísticas rápidas
    p1v = out_p1[~np.isnan(out_p1)]
    print(f"  dh/dt          : {np.mean(p1v):+.4f} ± {np.std(p1v):.4f} m/ano"
          f"  [min {np.min(p1v):+.2f}, max {np.max(p1v):+.2f}]")

    # Lon/lat dos nós
    node_lon, node_lat = to_lonlat.transform(nodes_x, nodes_y)

    # ------ Salvar ------
    with h5py.File(out_path, "w") as fo:
        for name, arr in [
            ("x",        nodes_x),
            ("y",        nodes_y),
            ("lon",      node_lon),
            ("lat",      node_lat),
            ("p0",       out_p0),
            ("p1",       out_p1),          # dh/dt
            ("p2",       out_p2),          # d²h/dt²
            ("p0_error", out_e0),
            ("p1_error", out_e1),          # incerteza formal de dh/dt
            ("p2_error", out_e2),
            ("rmse",     out_rms),
            ("nobs",     out_nob.astype(np.float64)),
            ("tspan",    out_tsp),
        ]:
            fo.create_dataset(name, data=arr,
                              compression="gzip", compression_opts=4)

        fo.attrs["tile"]           = stem
        fo.attrs["t_ref"]          = T_REF
        fo.attrs["search_radius"]  = SEARCH_RADIUS
        fo.attrs["grid_res"]       = GRID_RES
        fo.attrs["n_nodes"]        = n_nodes
        fo.attrs["n_valid"]        = n_valid
        fo.attrs["epsg"]           = 3031
        fo.attrs["p1_mean"]        = float(np.mean(p1v))
        fo.attrs["p1_std"]         = float(np.std(p1v))

    sz = out_path.stat().st_size / 1024**2
    print(f"  ✓ Salvo: {out_path.name}  ({sz:.1f} MB)")

    summary.append({
        "tile":    stem,
        "n_pts":   n_pts,
        "n_valid": n_valid,
        "p1_mean": float(np.mean(p1v)),
        "p1_med":  float(np.median(p1v)),
        "p1_std":  float(np.std(p1v)),
        "p1_min":  float(np.min(p1v)),
        "p1_max":  float(np.max(p1v)),
    })

    del x_all, y_all, h_all, t_all, s_all
    gc.collect()

# ============================================================
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"{'Tile':<25}  {'n_pts':>8}  {'n_nós':>6}  "
      f"{'dh/dt médio':>12}  {'±std':>8}")
print("─" * 65)
for s in summary:
    print(f"  {s['tile']:<23}  {s['n_pts']:>8,}  {s['n_valid']:>6,}  "
          f"  {s['p1_mean']:>+10.4f}  {s['p1_std']:>8.4f}")

if summary:
    all_means = [s["p1_mean"] for s in summary]
    print("─" * 65)
    print(f"  {'MÉDIA GLOBAL':<23}  {'':>8}  {'':>6}  "
          f"  {np.mean(all_means):>+10.4f}")

print("=" * 60)
print(f"\nResultados em: {DHDT_DIR}")
print("Próximo passo: temporal_analysis.py")

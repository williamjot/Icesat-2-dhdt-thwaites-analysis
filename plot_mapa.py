"""
05_plot_polar_map.py
====================
Mapa polar de dh/dt

"""

import sys
import warnings
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import *

# ============================================================
# CONFIGURAÇÃO
# ============================================================
MAP_GRID_RES  = 5_000    # metros — grade de saída do mapa
SMOOTH_SIGMA  = 1.5      # suavização Gaussiana (em células da grade de saída)
DPI           = 300

MAP_LON_MIN = LON_MIN - 2
MAP_LON_MAX = LON_MAX  + 2
MAP_LAT_MIN = LAT_MIN  - 1
MAP_LAT_MAX = LAT_MAX  + 1

# ============================================================
# PALETA RdBu
# ============================================================
# Divergente com branco/cinza claro no zero.
# vmin/vmax assimétricos porque o sinal de perda domina.
VMIN, VCENTER, VMAX = -3.0, 0.0, 1.5

def make_rdbu_norm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX):
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

CMAP_DHDT = plt.cm.RdBu_r         # vermelho = negativo (perda), azul = positivo
NORM_DHDT = make_rdbu_norm()

CMAP_ACC  = plt.cm.RdBu_r
NORM_ACC  = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)

CMAP_ERR  = plt.cm.YlOrRd
NORM_ERR  = mcolors.Normalize(vmin=0, vmax=1.5)

# ============================================================
# CARTOPY
# ============================================================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠ Cartopy não instalado — usando projeção simples.")

to_stereo = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
to_ll     = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)

print("=" * 60)
print("05 — MAPA POLAR DE dh/dt  (v2: RdBu + interpolação suave)")
print("=" * 60)

# ============================================================
# 1. CARREGAR RESULTADOS
# ============================================================
print("\n1. Carregando resultados do fitsec...")

dhdt_files = sorted(DHDT_DIR.glob("tile_*_dhdt.h5"))
if not dhdt_files:
    raise FileNotFoundError("Execute 03_calculate_dhdt.py primeiro.")

xs, ys, p1s, p2s, errs = [], [], [], [], []

for df in dhdt_files:
    with h5py.File(df, "r") as f:
        p1 = f["p1"][:]
        ok = ~np.isnan(p1)
        if not ok.any():
            continue
        xs.extend(f["x"][:][ok])
        ys.extend(f["y"][:][ok])
        p1s.extend(p1[ok])
        p2s.extend(f["p2"][:][ok]       if "p2"       in f else np.full(ok.sum(), np.nan))
        errs.extend(f["p1_error"][:][ok] if "p1_error" in f else np.full(ok.sum(), np.nan))

x_pts   = np.array(xs)
y_pts   = np.array(ys)
p1_pts  = np.array(p1s)
p2_pts  = np.array(p2s)
err_pts = np.array(errs)

print(f"   Nós válidos : {len(p1_pts):,}")
print(f"   dh/dt       : {np.mean(p1_pts):+.4f} ± {np.std(p1_pts):.4f} m/ano")
print(f"   Min/Max     : {np.min(p1_pts):+.2f} / {np.max(p1_pts):+.2f} m/ano")

# ============================================================
# 2. GRADE REGULAR
# ============================================================
print("\n2. Criando grade regular...")

x_min_g = np.floor(x_pts.min() / MAP_GRID_RES) * MAP_GRID_RES
x_max_g = np.ceil (x_pts.max() / MAP_GRID_RES) * MAP_GRID_RES
y_min_g = np.floor(y_pts.min() / MAP_GRID_RES) * MAP_GRID_RES
y_max_g = np.ceil (y_pts.max() / MAP_GRID_RES) * MAP_GRID_RES

xi = np.arange(x_min_g, x_max_g + MAP_GRID_RES, MAP_GRID_RES)
yi = np.arange(y_min_g, y_max_g + MAP_GRID_RES, MAP_GRID_RES)
Xi, Yi = np.meshgrid(xi, yi)
pts = np.c_[x_pts, y_pts]

print(f"   Grade: {Xi.shape[0]} × {Xi.shape[1]} = {Xi.size:,} células  "
      f"({MAP_GRID_RES/1000:.0f} km)")

# ============================================================
# 3. INTERPOLAÇÃO LINEAR
# ============================================================
# griddata linear interpola triangulando os pontos irregulares
# e calcula valores contínuos na grade — sem efeito escada.
# Pontos fora do convex hull ficam NaN (mascarados naturalmente).
# ============================================================
print("\n3. Interpolando (griddata linear)...")

grid_pts = np.c_[Xi.ravel(), Yi.ravel()]

grid_p1  = griddata(pts, p1_pts,  grid_pts, method="linear").reshape(Xi.shape)
grid_p2  = griddata(pts, p2_pts,  grid_pts, method="linear").reshape(Xi.shape)
grid_err = griddata(pts, err_pts, grid_pts, method="linear").reshape(Xi.shape)

print("   ✓ Interpolação linear concluída")

# Mascarar células muito longe dos dados
# (preenche buracos do griddata linear fora do convex hull)
from scipy.spatial import cKDTree
tree = cKDTree(pts)
dist, _ = tree.query(grid_pts)
dist = dist.reshape(Xi.shape)
mask_far = dist > (GRID_RES * 1.5)   # 1.5× resolução do fitsec

grid_p1[mask_far]  = np.nan
grid_p2[mask_far]  = np.nan
grid_err[mask_far] = np.nan

# ============================================================
# 4. SUAVIZAÇÃO GAUSSIANA
# ============================================================
# Aplicada sobre a grade linear para eliminar ruído residual
# sem criar efeito escada. Preserva NaN via normalização.
# ============================================================
def smooth_nan(grid, sigma):
    """Suavização Gaussiana que preserva NaN."""
    valid  = ~np.isnan(grid)
    filled = np.where(valid, grid, 0.0)
    sm     = gaussian_filter(filled.astype(float), sigma=sigma)
    wt     = gaussian_filter(valid.astype(float),  sigma=sigma)
    with np.errstate(invalid="ignore"):
        out = np.where(wt > 0.05, sm / wt, np.nan)
    out[~valid & (wt <= 0.05)] = np.nan
    return out

if SMOOTH_SIGMA > 0:
    print(f"\n4. Suavização Gaussiana (σ = {SMOOTH_SIGMA} células = "
          f"{SMOOTH_SIGMA * MAP_GRID_RES/1000:.1f} km)...")
    grid_p1  = smooth_nan(grid_p1,  SMOOTH_SIGMA)
    grid_p2  = smooth_nan(grid_p2,  SMOOTH_SIGMA)
    grid_err = smooth_nan(grid_err, SMOOTH_SIGMA)
    print("   ✓ Suavização concluída")

n_valid = int(np.sum(~np.isnan(grid_p1)))
print(f"\n   Células preenchidas: {n_valid:,} ({100*n_valid/Xi.size:.1f}%)")

# Lon/lat da grade (para plotar)
grid_lon, grid_lat = to_ll.transform(Xi, Yi)

FIGURES.mkdir(exist_ok=True, parents=True)

# ============================================================
# FUNÇÃO DE MAPA
# ============================================================

def make_polar_map(data, cmap, norm, title, cbar_label, out_file,
                   cbar_ticks=None, extend="both"):

    if HAS_CARTOPY:
        proj = ccrs.SouthPolarStereo()
        fig  = plt.figure(figsize=(12, 10))
        ax   = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([MAP_LON_MIN, MAP_LON_MAX,
                       MAP_LAT_MIN, MAP_LAT_MAX],
                      crs=ccrs.PlateCarree())

        # Fundo oceano
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor="#D6EAF8", alpha=0.7, zorder=1)
        ax.add_feature(cfeature.LAND.with_scale("50m"),
                       facecolor="#F0EAD6", alpha=0.3, zorder=1)

        # Dados
        im = ax.pcolormesh(
            grid_lon, grid_lat, data,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            shading="gouraud",   # ← interpolação bilinear no rendering
            rasterized=True, zorder=2
        )

        # Costa
        ax.coastlines(resolution="10m", color="#333333",
                      linewidth=1.0, zorder=5)

        # Graticule
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.4, color="gray", alpha=0.5,
            linestyle="--", x_inline=False, y_inline=False,
            zorder=6
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8, "color": "#555555"}
        gl.ylabel_style = {"size": 8, "color": "#555555"}
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 10))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, -60, 2))

    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.pcolormesh(
            grid_lon, grid_lat, data,
            cmap=cmap, norm=norm,
            shading="gouraud", rasterized=True
        )
        ax.set_xlabel("Longitude (°)", fontsize=10)
        ax.set_ylabel("Latitude (°)",  fontsize=10)
        ax.set_aspect(1 / np.cos(np.radians(np.nanmean(grid_lat))))
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    # Colorbar
    cbar = plt.colorbar(
        im, ax=ax, orientation="vertical",
        shrink=0.72, pad=0.03, extend=extend,
        ticks=cbar_ticks, format="%.2g"
    )
    cbar.set_label(cbar_label, fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold",
                 pad=14, loc="left")

    fig.text(
        0.12, 0.015,
        (f"ICESat-2 ATL06  |  JJA 2019–2025  |  EPSG:3031  |  "
         f"Grade {MAP_GRID_RES/1000:.0f} km  |  "
         f"Interp. linear + Gauss σ={SMOOTH_SIGMA*MAP_GRID_RES/1000:.1f} km"),
        fontsize=7.5, color="#777777"
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✓ {out_file.name}")


# ============================================================
# 5. MAPAS
# ============================================================
print("\n5. Gerando mapas...")

# Ticks da colorbar dh/dt
ticks_dhdt = [-3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 1.5]

make_polar_map(
    data       = grid_p1,
    cmap       = CMAP_DHDT,
    norm       = NORM_DHDT,
    title      = ("Taxa de Mudança de Elevação  dh/dt\n"
                  "Amundsen Sea Embayment  —  Inverno Austral (JJA)"),
    cbar_label = "dh/dt (m/ano)",
    out_file   = FIGURES / "map_dhdt_polar.png",
    cbar_ticks = ticks_dhdt,
    extend     = "both"
)

has_p2 = np.sum(~np.isnan(p2_pts)) > 100

if has_p2:
    make_polar_map(
        data       = grid_p2,
        cmap       = CMAP_ACC,
        norm       = NORM_ACC,
        title      = ("Aceleração  d²h/dt²\n"
                      "Amundsen Sea Embayment  —  JJA"),
        cbar_label = "d²h/dt² (m/ano²)",
        out_file   = FIGURES / "map_accel_polar.png",
        cbar_ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
        extend     = "both"
    )

make_polar_map(
    data       = grid_err,
    cmap       = CMAP_ERR,
    norm       = NORM_ERR,
    title      = ("Incerteza Formal de dh/dt  (p1_error)\n"
                  "Amundsen Sea Embayment  —  JJA"),
    cbar_label = "Incerteza (m/ano)",
    out_file   = FIGURES / "map_error_polar.png",
    cbar_ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    extend     = "max"
)

# ============================================================
# 6. PAINEL 2×2 PARA PUBLICAÇÃO
# ============================================================
print("\n6. Gerando painel para publicação...")

if HAS_CARTOPY:
    proj = ccrs.SouthPolarStereo()
    fig  = plt.figure(figsize=(18, 14))

    def add_panel(pos, data, cm, nm, tks, ext, label):
        ax = fig.add_subplot(pos, projection=proj)
        ax.set_extent([MAP_LON_MIN, MAP_LON_MAX,
                       MAP_LAT_MIN, MAP_LAT_MAX],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                       facecolor="#D6EAF8", alpha=0.6, zorder=1)
        im = ax.pcolormesh(
            grid_lon, grid_lat, data,
            cmap=cm, norm=nm,
            transform=ccrs.PlateCarree(),
            shading="gouraud", rasterized=True, zorder=2
        )
        ax.coastlines(resolution="50m", color="#333333",
                      linewidth=0.7, zorder=5)
        gl = ax.gridlines(linewidth=0.3, color="gray",
                          alpha=0.4, linestyle="--")
        gl.top_labels = gl.right_labels = False
        cb = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03,
                          extend=ext, ticks=tks, format="%.2g")
        cb.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=10, fontweight="bold",
                     loc="left", pad=5)
        return ax

    add_panel(221, grid_p1, CMAP_DHDT, NORM_DHDT,
              ticks_dhdt, "both", "(a)  dh/dt (m/ano)")

    if has_p2:
        add_panel(222, grid_p2, CMAP_ACC, NORM_ACC,
                  [-0.3,-0.1,0,0.1,0.3], "both",
                  "(b)  d²h/dt² (m/ano²)")
    else:
        add_panel(222, grid_err, CMAP_ERR, NORM_ERR,
                  [0,0.5,1.0,1.5], "max", "(b)  Incerteza (m/ano)")

    add_panel(223, grid_err, CMAP_ERR, NORM_ERR,
              [0,0.25,0.5,0.75,1.0,1.5], "max",
              "(c)  Incerteza (m/ano)")

    # Painel (d) — histograma
    ax4  = fig.add_subplot(224)
    bins = np.arange(-3.5, 2.0, 0.1)
    ax4.hist(p1_pts, bins=bins, density=True,
             color="#CC2A2A", alpha=0.80, edgecolor="none")
    ax4.axvline(0, color="black", lw=1.5, ls="--", alpha=0.6)
    ax4.axvline(np.mean(p1_pts), color="#1565C0", lw=2,
                label=f"Média {np.mean(p1_pts):+.3f} m/ano")
    ax4.axvline(np.median(p1_pts), color="#1565C0", lw=1.5, ls=":",
                label=f"Mediana {np.median(p1_pts):+.3f} m/ano")
    ax4.set_xlabel("dh/dt (m/ano)", fontsize=10)
    ax4.set_ylabel("Densidade",     fontsize=10)
    ax4.set_title("(d)  Distribuição de dh/dt", fontsize=10,
                  fontweight="bold", loc="left")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-3.5, 2.0)

    thin  = 100 * np.sum(p1_pts < 0) / len(p1_pts)
    thick = 100 * np.sum(p1_pts > 0) / len(p1_pts)
    area  = n_valid * (MAP_GRID_RES / 1000)**2
    ax4.text(0.02, 0.97,
             f"n = {len(p1_pts):,} nós\n"
             f"Afinamento : {thin:.0f}%\n"
             f"Espessamento: {thick:.0f}%\n"
             f"Área: {area:,.0f} km²",
             transform=ax4.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#AAAAAA", alpha=0.9))

    fig.suptitle(
        "ICESat-2 ATL06  —  Mudança de Elevação do Gelo\n"
        "Amundsen Sea Embayment  |  Inverno Austral (JJA) 2019–2025",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_panel = FIGURES / "map_panel_publication.png"
    plt.savefig(out_panel, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"   ✓ {out_panel.name}")
else:
    print("   ⚠ Cartopy necessário para o painel. "
          "Instale: conda install -c conda-forge cartopy")

# ============================================================
print("\n" + "=" * 60)
print("FIGURAS GERADAS")
print("=" * 60)
for f in sorted(FIGURES.glob("map_*polar*.png")) + \
         sorted(FIGURES.glob("map_panel*.png")):
    sz = f.stat().st_size / 1024**2
    print(f"  {f.name:<40}  ({sz:.1f} MB)")
print(f"\n  Diretório: {FIGURES}")

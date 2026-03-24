"""
apply_mask.py  v2
Aplica máscara BedMachine Antarctica (.tif) ao arquivo merged.

Funciona com o .tif de toda a antártica, lê apenas a janela
correspondente à sua bbox

Remove oceano (mask == 0), mantém gelo (1, 2), rocha (3), lago (4).
"""

import numpy as np
import h5py
import rasterio
import rasterio.windows
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from pathlib import Path
from pyproj import Transformer
import gc

# ============================================================
# CAMINHOS — ajuste conforme sua instalação
# ============================================================
BASE     = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")
MERGED   = BASE / "data" / "atl06_merged.h5"
MASKED   = BASE / "data" / "atl06_masked.h5"
BED_TIF  = BASE / "data" / "bedmachine" / "bedmachine_mask.tif"

KEEP_MASK_VALUES = {1, 2, 3, 4}   # 0 = oceano → removido
CHUNK = 10_000_000

# ============================================================
print("=" * 60)
print("APLICAÇÃO DE MÁSCARA BEDMACHINE  (windowed reading)")
print("=" * 60)

if not MERGED.exists():
    raise FileNotFoundError(f"Merged não encontrado:\n  {MERGED}")
if not BED_TIF.exists():
    raise FileNotFoundError(
        f"BedMachine .tif não encontrado:\n  {BED_TIF}\n"
        f"Verifique o caminho e o nome exato do arquivo."
    )

# ============================================================
# 1. Bbox dos dados para recortar o raster
# ============================================================
print("\n1. Lendo bbox dos dados...")

with h5py.File(MERGED, "r") as f:
    n_total   = f["lat"].shape[0]
    variables = list(f.keys())
    lon_s = f["lon"][::1000]
    lat_s = f["lat"][::1000]

lon_min = float(np.nanmin(lon_s)) - 1.0
lon_max = float(np.nanmax(lon_s)) + 1.0
lat_min = float(np.nanmin(lat_s)) - 1.0
lat_max = float(np.nanmax(lat_s)) + 1.0

print(f"   Pontos totais : {n_total:,}")
print(f"   Variáveis     : {variables}")
print(f"   Bbox lon      : {lon_min:.2f}  a  {lon_max:.2f} °")
print(f"   Bbox lat      : {lat_min:.2f}  a  {lat_max:.2f} °")

# ============================================================
# 2. Recortar raster (janela da bbox)
# ============================================================
print(f"\n2. Abrindo raster (janela recortada): {BED_TIF.name}")

to_stereo = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

corners_lon = [lon_min, lon_max, lon_min, lon_max]
corners_lat = [lat_min, lat_min, lat_max, lat_max]
cx, cy = to_stereo.transform(corners_lon, corners_lat)

bbox_x_min = min(cx) - 10_000
bbox_x_max = max(cx) + 10_000
bbox_y_min = min(cy) - 10_000
bbox_y_max = max(cy) + 10_000

with rasterio.open(BED_TIF) as src:
    print(f"   CRS           : {src.crs}")
    print(f"   Shape total   : {src.height} x {src.width}")
    print(f"   Resolucao     : {src.res[0]:.0f} m")

    window = from_bounds(bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
                         transform=src.transform)
    window = window.intersection(
        rasterio.windows.Window(0, 0, src.width, src.height)
    )

    mask_data      = src.read(1, window=window)
    mask_transform = src.window_transform(window)

print(f"   Janela lida   : {mask_data.shape}  "
      f"({mask_data.nbytes / 1024**2:.0f} MB na RAM)")
print(f"   Valores unicos: {np.unique(mask_data)}")

# ============================================================
# 3. Passagem 1 — classificar pontos
# ============================================================
n_chunks      = (n_total + CHUNK - 1) // CHUNK
n_valid_total = 0
valid_chunks  = []

print(f"\n3. Passagem 1 — classificando {n_total/1e6:.0f}M pontos...")

with h5py.File(MERGED, "r") as f:
    for i in range(n_chunks):
        s = i * CHUNK
        e = min(s + CHUNK, n_total)

        x_c, y_c = to_stereo.transform(f["lon"][s:e], f["lat"][s:e])

        rows, cols = rowcol(mask_transform, x_c, y_c)
        rows = np.clip(rows, 0, mask_data.shape[0] - 1)
        cols = np.clip(cols, 0, mask_data.shape[1] - 1)

        valid = np.isin(mask_data[rows, cols], list(KEEP_MASK_VALUES))
        n_valid_total += int(valid.sum())
        valid_chunks.append(valid)

        print(f"   Chunk {i+1:2d}/{n_chunks}  ({100*(i+1)/n_chunks:.0f}%)  "
              f"validos: {n_valid_total:,}")

n_removed = n_total - n_valid_total
print(f"\n   Gelo/rocha : {n_valid_total:,}  ({100*n_valid_total/n_total:.1f}%)")
print(f"   Oceano     : {n_removed:,}  ({100*n_removed/n_total:.1f}%)")

del mask_data
gc.collect()

# ============================================================
# 4. Passagem 2 — escrever arquivo mascarado
# ============================================================
print(f"\n4. Passagem 2 — escrevendo {MASKED.name}...")

if MASKED.exists():
    MASKED.unlink()

datasets_created = False

with h5py.File(MERGED, "r") as f_in, h5py.File(MASKED, "w") as f_out:

    for i in range(n_chunks):
        s = i * CHUNK
        e = min(s + CHUNK, n_total)

        valid = valid_chunks[i]
        if not valid.any():
            continue

        chunk_data = {var: f_in[var][s:e][valid] for var in variables}

        if not datasets_created:
            for var, arr in chunk_data.items():
                f_out.create_dataset(var, data=arr,
                                     maxshape=(None,), chunks=True,
                                     compression="gzip", compression_opts=4)
            datasets_created = True
        else:
            for var, arr in chunk_data.items():
                cur = f_out[var].shape[0]
                f_out[var].resize(cur + len(arr), axis=0)
                f_out[var][cur:] = arr

        del chunk_data
        gc.collect()
        print(f"   Chunk {i+1:2d}/{n_chunks}  ({100*(i+1)/n_chunks:.0f}%)")

    f_out.attrs["n_points"]    = n_valid_total
    f_out.attrs["n_removed"]   = n_removed
    f_out.attrs["mask_source"] = "BedMachine Antarctica"
    f_out.attrs["keep_values"] = str(KEEP_MASK_VALUES)

sz = MASKED.stat().st_size / 1024**3
print(f"\n[OK] Salvo: {MASKED.name}  ({sz:.2f} GB)")
print("=" * 60)
print("Proximo passo: create_tiles.py")

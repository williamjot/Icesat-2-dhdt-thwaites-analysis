"""
create_tiles.py
Divide atl06_masked.h5 em tiles espaciais no sistema EPSG:3031 (metros).

BBOX da região: lon (-106.5, -103.5)  lat (-75.5, -74.5)
Área estimada : ~90 km × ~110 km  ≈ ~10 000 km²

Tamanho do tile: 50 km × 50 km  → grade 2×3 = ~6 tiles possíveis
  (ajuste TILE_KM se quiser mais ou menos subdivisão)

Cada tile salvo como tile_XXXX_YYYY.h5 com variáveis:
  x, y  (EPSG:3031, metros)
  lon, lat, h_elv, s_elv, t_year, beam, spot, orb

Leitura 100% sequencial via memmap — sem fancy-indexing.
"""

import numpy as np
import h5py
from pathlib import Path
from pyproj import Transformer
import gc
import shutil

# ============================================================
# CONFIGURAÇÃO
# ============================================================
BASE      = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")
MASKED    = BASE / "data"  / "atl06_masked.h5"
TILES_DIR = BASE / "tiles"

TILE_KM   = 50          # tamanho do tile em km  (50 × 50 km)
TILE_M    = TILE_KM * 1000

CHUNK     = 10_000_000  # pontos por chunk de leitura

# Projeção
to_stereo  = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# ============================================================
print("=" * 60)
print(f"CRIAÇÃO DE TILES  ({TILE_KM} km × {TILE_KM} km)")
print("=" * 60)

TILES_DIR.mkdir(exist_ok=True, parents=True)

# ------ Inspecionar arquivo ------
with h5py.File(MASKED, "r") as f:
    n_total   = f["lat"].shape[0]
    variables = list(f.keys())
    has_xy    = "x" in variables and "y" in variables

print(f"Arquivo  : {MASKED.name}")
print(f"Pontos   : {n_total:,}")
print(f"Variáveis: {variables}")
print(f"Tem x,y  : {has_xy}")

n_chunks = (n_total + CHUNK - 1) // CHUNK

# ============================================================
# PASSAGEM 1 — extensão espacial + tile_key por ponto
# ============================================================
print("\n1. Passagem 1 — extensão e tile keys (memmap)...")

temp_dir = TILES_DIR / "_temp"
temp_dir.mkdir(exist_ok=True, parents=True)
tkey_path = temp_dir / "tile_keys.dat"
xall_path = temp_dir / "x_all.dat"
yall_path = temp_dir / "y_all.dat"

# Arrays temporários em disco
x_mmap = np.memmap(xall_path, dtype=np.float64, mode="w+", shape=(n_total,))
y_mmap = np.memmap(yall_path, dtype=np.float64, mode="w+", shape=(n_total,))

x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf

with h5py.File(MASKED, "r") as f:
    for i in range(n_chunks):
        s = i * CHUNK
        e = min(s + CHUNK, n_total)

        if has_xy:
            xc = f["x"][s:e]
            yc = f["y"][s:e]
        else:
            lon_c = f["lon"][s:e]
            lat_c = f["lat"][s:e]
            xc, yc = to_stereo.transform(lon_c, lat_c)

        x_mmap[s:e] = xc
        y_mmap[s:e] = yc

        x_min = min(x_min, float(np.min(xc)))
        x_max = max(x_max, float(np.max(xc)))
        y_min = min(y_min, float(np.min(yc)))
        y_max = max(y_max, float(np.max(yc)))

        del xc, yc

x_mmap.flush()
y_mmap.flush()

print(f"   X: {x_min/1000:.1f} a {x_max/1000:.1f} km")
print(f"   Y: {y_min/1000:.1f} a {y_max/1000:.1f} km")

# Grade de tiles
x0 = np.floor(x_min / TILE_M) * TILE_M
y0 = np.floor(y_min / TILE_M) * TILE_M
nx = int(np.ceil((x_max - x0) / TILE_M))
ny = int(np.ceil((y_max - y0) / TILE_M))

print(f"   Grade: {nx} × {ny} = {nx*ny} tiles possíveis")

# Calcular tile_key por ponto
tkey_mmap = np.memmap(tkey_path, dtype=np.int32, mode="w+", shape=(n_total,))

for i in range(n_chunks):
    s = i * CHUNK
    e = min(s + CHUNK, n_total)

    xc = x_mmap[s:e]
    yc = y_mmap[s:e]

    ti = np.floor((xc - x0) / TILE_M).astype(np.int32)
    tj = np.floor((yc - y0) / TILE_M).astype(np.int32)
    np.clip(ti, 0, nx - 1, out=ti)
    np.clip(tj, 0, ny - 1, out=tj)

    tkey_mmap[s:e] = ti * ny + tj
    del xc, yc, ti, tj

tkey_mmap.flush()
print("   ✓ Tile keys calculados")

# ============================================================
# PASSAGEM 2 — contar pontos por tile
# ============================================================
print("\n2. Passagem 2 — contando pontos por tile...")

n_tiles_total = nx * ny
tile_counts = np.zeros(n_tiles_total, dtype=np.int64)

for i in range(n_chunks):
    s = i * CHUNK
    e = min(s + CHUNK, n_total)
    chunk = np.array(tkey_mmap[s:e])
    tile_counts += np.bincount(chunk, minlength=n_tiles_total)
    del chunk

active = np.where(tile_counts > 0)[0]
print(f"   Tiles com dados: {len(active)}/{n_tiles_total}")
for tk in active:
    ti, tj = divmod(int(tk), ny)
    print(f"   tile_{ti:04d}_{tj:04d}: {tile_counts[tk]:,} pontos")

# ============================================================
# PASSAGEM 3 — pré-criar arquivos temporários por tile
# ============================================================
print("\n3. Passagem 3 — pré-criando arquivos temporários...")

# dtypes das variáveis
with h5py.File(MASKED, "r") as f:
    var_dtypes = {v: f[v].dtype for v in variables}

# Adicionar x, y se não existirem
if not has_xy:
    var_dtypes["x"] = np.float64
    var_dtypes["y"] = np.float64

all_vars = list(var_dtypes.keys())
write_pos = {}

for tk in active:
    n = int(tile_counts[tk])
    path = temp_dir / f"temp_{tk:06d}.h5"
    with h5py.File(path, "w") as tf:
        for var in all_vars:
            tf.create_dataset(var, shape=(n,), dtype=var_dtypes[var])
    write_pos[tk] = 0

print("   ✓ Arquivos temporários criados")

# ============================================================
# PASSAGEM 4 — distribuir dados
# ============================================================
print("\n4. Passagem 4 — distribuindo pontos nos tiles...")

with h5py.File(MASKED, "r") as f:
    for i in range(n_chunks):
        s = i * CHUNK
        e = min(s + CHUNK, n_total)

        keys_c = np.array(tkey_mmap[s:e])

        # Ler variáveis do chunk
        data_c = {v: f[v][s:e] for v in variables}

        # Adicionar x, y se necessário
        if not has_xy:
            xc, yc = to_stereo.transform(data_c["lon"], data_c["lat"])
            data_c["x"] = xc
            data_c["y"] = yc

        for tk in np.unique(keys_c):
            mask_t = keys_c == tk
            n_t    = int(np.sum(mask_t))
            wp     = write_pos[tk]
            path   = temp_dir / f"temp_{tk:06d}.h5"

            with h5py.File(path, "a") as tf:
                for var in all_vars:
                    tf[var][wp:wp + n_t] = data_c[var][mask_t]

            write_pos[tk] = wp + n_t

        del data_c, keys_c
        gc.collect()

        pct = 100 * (i + 1) / n_chunks
        print(f"   Chunk {i+1}/{n_chunks} ({pct:.0f}%)")

print("   ✓ Dados distribuídos")

# ============================================================
# PASSAGEM 5 — converter temporários em tiles finais
# ============================================================
print("\n5. Passagem 5 — salvando tiles finais...")

n_created = 0

for tk in active:
    ti, tj = divmod(int(tk), ny)

    x_tile_min = x0 + ti * TILE_M
    x_tile_max = x_tile_min + TILE_M
    y_tile_min = y0 + tj * TILE_M
    y_tile_max = y_tile_min + TILE_M

    temp_path  = temp_dir / f"temp_{tk:06d}.h5"
    final_path = TILES_DIR / f"tile_{ti:04d}_{tj:04d}.h5"

    with h5py.File(temp_path, "r") as tf, h5py.File(final_path, "w") as fo:
        for var in all_vars:
            fo.create_dataset(
                var,
                data=tf[var][:],
                compression="gzip",
                compression_opts=4
            )
        # Metadados do tile
        fo.attrs["tile_i"]       = ti
        fo.attrs["tile_j"]       = tj
        fo.attrs["x_min"]        = x_tile_min
        fo.attrs["x_max"]        = x_tile_max
        fo.attrs["y_min"]        = y_tile_min
        fo.attrs["y_max"]        = y_tile_max
        fo.attrs["tile_size_m"]  = TILE_M
        fo.attrs["epsg"]         = 3031
        fo.attrs["n_points"]     = tile_counts[tk]

    n_created += 1
    print(f"   ✓ tile_{ti:04d}_{tj:04d}.h5  ({tile_counts[tk]:,} pontos)")

# Limpeza
print("\n6. Limpando temporários...")
shutil.rmtree(temp_dir)
print("   ✓ Temporários removidos")

# ============================================================
print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print(f"Tiles criados : {n_created}")
print(f"Diretório     : {TILES_DIR}")
print("=" * 60)
print("Próximo passo : calculate_dhdt.py")

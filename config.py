"""
config.py
=========
Configuração central do pipeline ICESat-2 ATL06.
Importada por todos os scripts.
"""

from pathlib import Path

# ============================================================
# CAMINHOS
# ============================================================
BASE       = Path(r"D:\WILLIAM\PIBIC_MARCO\thwaites_winter")

DATA_DIR   = BASE / "data"
TILES_DIR  = BASE / "tiles"
DHDT_DIR   = BASE / "dhdt"
RESULTS    = BASE / "results"
FIGURES    = BASE / "figures"
QGIS_DIR   = BASE / "qgis"
LOGS_DIR   = BASE / "logs"

# Arquivos principais
MERGED     = DATA_DIR / "atl06_merged.h5"       # entrada do pipeline
MASKED     = DATA_DIR / "atl06_masked.h5"        # pós máscara BedMachine
BED_TIF    = DATA_DIR / "bedmachine" / "bedmachine_mask.tif"

# Criar diretórios
for d in [DATA_DIR, TILES_DIR, DHDT_DIR, RESULTS, FIGURES, QGIS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ============================================================
# BOUNDING BOX
# ============================================================
LON_MIN = -114.4998
LON_MAX =  -69.0001
LAT_MIN =  -80.0000
LAT_MAX =  -68.5001

# ============================================================
# MÁSCARA BEDMACHINE
# ============================================================
# 0=oceano, 1=gelo terrestre, 2=gelo flutuante, 3=rocha, 4=lago subglacial
KEEP_MASK_VALUES = {1, 2, 3, 4}   # remove oceano

# ============================================================
# TILES
# ============================================================
TILE_KM = 100          # km  — tiles de 100×100 km para área grande
TILE_M  = TILE_KM * 1000

# ============================================================
# FITSEC (calculate_dhdt)
# ============================================================
SEARCH_RADIUS = 25_000.0   # metros — cobre separação entre tracks (~20-28 km a 75°S)
MIN_POINTS    = 20          # pontos mínimos para ajuste
POLY_ORDER    = 2           # ordem espacial
TEMP_ORDER    = 2           # ordem temporal (2 → estima aceleração)
MAX_ITER      = 5
N_SIGMA       = 3.0
RESID_LIM     = 5.0         # metros
RATE_LIM      = 10.0        # m/ano
P2_LIM        = 1.0         # m/ano²
DT_MIN        = 1.5         # anos — span mínimo
T_REF         = 2022.5      # centro do período 2019–2025
GRID_RES      = 5_000       # metros — resolução da grade interna dos tiles
                             # (use 2000 para mais detalhe, ~6× mais lento)
USE_WEIGHTS   = True
SKIP_DONE     = True        # pular tiles já processados (retomar)

# ============================================================
# PROCESSAMENTO
# ============================================================
CHUNK = 10_000_000   # pontos por chunk de leitura

# ============================================================
# QGIS — passo de amostragem para exportação
# ============================================================
# merged (~319M pts): 1/1000 = ~319K pts no QGIS
# masked (~278M pts): 1/500  = ~556K pts
# cropped (área da bbox): 1/100
SAMPLE_MASKED = 500
SAMPLE_CROPPED = 50

# ============================================================
# ANOS DE INTERESSE (JJA)
# ============================================================
JJA_YEARS = list(range(2019, 2026))   # 2019 a 2025

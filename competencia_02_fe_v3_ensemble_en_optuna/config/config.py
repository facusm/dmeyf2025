# config/config.py
import os
from datetime import datetime

# ==================================================================================
# PATHS BASE
# ==================================================================================

BUCKET_PATH_b1 = "/home/sanmartinofacundo/buckets/b1"

LOCAL_DATA_DIR = "/home/sanmartinofacundo/datasets"
DATASET_CRUDO_PATH = os.path.join(LOCAL_DATA_DIR, "competencia_02_crudo.csv.gz")
DATASET_TARGETS_CREADOS_PATH = os.path.join(BUCKET_PATH_b1, "competencia_02.csv.gz")

FILE_BASE = "competencia_02"
PROJECT_NAME = "competencia02"

# ==================================================================================
# FEATURE ENGINEERING
# ==================================================================================

# Describí acá la variante de FE (lags, ventanas, reglas, etc.)
SUFIJO_FE = "fe_v3"
VERSION = "v3"

FEATURES_ROOT = os.path.join(BUCKET_PATH_b1, "features")
FEATURES_DIR = os.path.join(FEATURES_ROOT, SUFIJO_FE)
os.makedirs(FEATURES_DIR, exist_ok=True)

FE_FILENAME = f"{FILE_BASE}_{SUFIJO_FE}.parquet"
FE_PATH = os.path.join(FEATURES_DIR, FE_FILENAME)

# ==================================================================================
# MESES (ACTUALIZADO CON LO QUE DEFINISTE)
# ==================================================================================

# Entrenamiento base: 201901–202103
MESES_TRAIN = [
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103, 
] # Evaluar sacar algunos meses según los experimentos

# Validación interna (Optuna)
MES_VAL_OPTUNA = [202104, 202105]

# Validación externa (ajuste de umbral / sanity check)
MES_VALID = [202106]

# Test final (podés correr ambos escenarios separados con el mismo experimento)
MES_TEST_FINAL = [202108]

# Semillas para ensemble
SEMILLAS = [
    306491, 336251, 900577, 182009, 182011,
    182027, 182029, 182041,
    101111, 103333, 105227, 107071, 109037,
    113483, 117109, 119617, 123457, 127043,
    130363, 137111, 139129, 149111, 151007,
    157337, 163811, 167009, 173807, 179989,
    191141, 197813
]


# ==================================================================================
# UNDERSAMPLING
# ==================================================================================

APLICAR_UNDERSAMPLING = True
RATIO_UNDERSAMPLING = 0.1 

def _tag_us():
    if (not APLICAR_UNDERSAMPLING) or (RATIO_UNDERSAMPLING >= 0.999):
        return "us100"
    return f"us{int(RATIO_UNDERSAMPLING * 100):03d}"

def _tag_train():
    return f"tr{min(MESES_TRAIN)}-{max(MESES_TRAIN)}"

def _tag_list(prefix: str, meses: list[int]) -> str:
    if not meses:
        return ""
    if len(meses) == 1:
        return f"{prefix}{meses[0]}"
    # No asumo continuidad; uso primero y último como resumen compacto
    return f"{prefix}{meses[0]}-{meses[-1]}"

def _tag_test():
    return _tag_list("test", MES_TEST_FINAL)

def build_experiment_name() -> str:
    """
    ID único y legible del experimento:
    lgbm_{FE}_{US}_{train}_{val}_{vext}_{test}_s{n_seeds}
    """
    n_seeds = len(SEMILLAS)

    parts = [
        "lgbm",
        SUFIJO_FE,
        _tag_us(),
        _tag_train(),
        _tag_list("val", MES_VAL_OPTUNA),
        _tag_list("vext", MES_VALID),
        _tag_test(),
        f"s{n_seeds}"  # cantidad de semillas usadas en ensemble
    ]

    return "_".join(p for p in parts if p)


NOMBRE_EXPERIMENTO = build_experiment_name()


# ==================================================================================
# ESTRUCTURA DE SALIDAS POR EXPERIMENTO
# ==================================================================================

EXPERIMENTS_ROOT = os.path.join(BUCKET_PATH_b1, "competencia_02")
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

EXPERIMENT_DIR = os.path.join(EXPERIMENTS_ROOT, NOMBRE_EXPERIMENTO)

DB_PATH = os.path.join(EXPERIMENT_DIR, "db")
MODELOS_PATH = os.path.join(EXPERIMENT_DIR, "modelos")
LOGS_PATH = os.path.join(EXPERIMENT_DIR, "logs")
RESULTADOS_PREDICCION_PATH = os.path.join(EXPERIMENT_DIR, "resultados_prediccion")

for path in [EXPERIMENT_DIR, DB_PATH, MODELOS_PATH, LOGS_PATH, RESULTADOS_PREDICCION_PATH]:
    os.makedirs(path, exist_ok=True)

# ==================================================================================
# OPTUNA
# ==================================================================================

NOMBRE_DE_ESTUDIO_OPTUNA = f"{NOMBRE_EXPERIMENTO}_optuna"
NOMBRE_BASE_DE_DATOS_OPTUNA = f"{NOMBRE_EXPERIMENTO}_optuna_study.db"

N_TRIALS = 50
N_STARTUP_TRIALS = 20

# ==================================================================================
# NEGOCIO / MÉTRICA
# ==================================================================================

GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = -20000

# ==================================================================================
# PARAMS BASE DEL MODELO
# ==================================================================================

PARAMS = {
    "ganancia_acierto": GANANCIA_ACIERTO,
    "costo_estimulo": COSTO_ESTIMULO,
    "n_folds": 5,
    "num_boost_round": 2000,
    "early_stopping_rounds": 50,
    "target": "clase_binaria2",
}

# ==================================================================================
# COLUMNAS
# ==================================================================================

COLS_ID = ["foto_mes", "numero_de_cliente"]
ELIMINAR_COLUMNAS_ID = False

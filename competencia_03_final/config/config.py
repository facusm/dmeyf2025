# config/config.py
import os
from datetime import datetime


# ==================================================================================
# PATHS BASE
# ==================================================================================

BUCKET_PATH_b1 = "/home/sanmartinofacundo/buckets/b1"

LOCAL_DATA_DIR = "/home/sanmartinofacundo/datasets"
DATASET_TARGETS_CREADOS_PATH = os.path.join(BUCKET_PATH_b1, "competencia_03.csv.gz")

FILE_BASE = "competencia_03"
PROJECT_NAME = "competencia03"

# ==================================================================================
# FEATURE ENGINEERING
# ==================================================================================

# Describí acá la variante de FE (lags, ventanas, reglas, etc.)
SUFIJO_FE = "fe_v10"
VERSION = "v10"

FEATURES_ROOT = os.path.join(BUCKET_PATH_b1, "features")
FEATURES_DIR = os.path.join(FEATURES_ROOT, SUFIJO_FE)
os.makedirs(FEATURES_DIR, exist_ok=True)

FE_FILENAME = f"{FILE_BASE}_{SUFIJO_FE}.parquet"
FE_PATH = os.path.join(FEATURES_DIR, FE_FILENAME)

# ==================================================================================
# MESES (ACTUALIZADO CON LO QUE DEFINISTE)
# ==================================================================================

# Entrenamiento base: 201901–202102
MESES_TRAIN_OPTUNA = [
    201901, 201902, 201903, 201904, 201905, 201906,
    201907, 201908, 201909, 201910, 201911, 201912,
    202001, 202002, 202003, 202004, 202005, 202006,
    202007, 202008, 202009, 202010, 202011, 202012,
    202101, 202102, 202103 
] # Evaluar sacar algunos meses según los experimentos

# Validación interna (Optuna)
MES_VAL_OPTUNA = [202105]

MESES_TRAIN_PARA_VAL_EXT = MESES_TRAIN_OPTUNA + [202104] + MES_VAL_OPTUNA

# Validación externa (ajuste de umbral / sanity check)
MES_VALID_EXT = [202107]

MESES_TRAIN_COMPLETO_PARA_TEST_FINAL = MESES_TRAIN_PARA_VAL_EXT + [202106] + MES_VALID_EXT

# Test final (podés correr ambos escenarios separados con el mismo experimento)
MES_TEST_FINAL = [202109]

SEMILLAS_TOTALES = 20  # Número total de semillas disponibles
# Semillas para Optuna 
SEMILLAS_OPTUNA = [
    306491
]

# Número de repeticiones (BO "repe" estilo APO)
N_REPE_OPTUNA = 1  

# Semillas para ensemble final:
SEMILLAS_ENSEMBLE = SEMILLAS_OPTUNA + [
    100003, 100019, 100043, 100049, 100057, 100069, 100103, 100109, 100129, 100151,
    100153, 100169, 100183, 100189, 100193, 100207, 100213, 100237, 100267, 100271,
    100279, 100291, 100297, 100313, 100333, 100343, 100357, 100361, 100363, 100379,
    100391, 100393, 100403, 100411, 100417, 100447, 100459, 100469, 100483, 100493,
    100501, 100511, 100517, 100519, 100523, 100537, 100547, 100549, 100559, 100591,
    100609, 100613, 100621, 100649, 100669, 100673, 100693, 100699, 100703, 100733,
    100741, 100747, 100769, 100787, 100799, 100801, 100811, 100823, 100829, 100847,
    100853, 100907, 100913, 100927, 100931, 100937, 100943, 100957, 100981, 100987,
    100999, 101009, 101021, 101027, 101051, 101063, 101081, 101089, 101107, 101111,
    101113, 101117, 101119, 336251, 900577, 182009, 182011, 182027, 800089
] # Poner 100 semillas en total para que coincida con APO_K_SEM * APO_N_APO si se usa APO

SEMILLAS_ENSEMBLE = SEMILLAS_ENSEMBLE[:SEMILLAS_TOTALES]  

# ==================================================================================
# UNDERSAMPLING
# ==================================================================================

APLICAR_UNDERSAMPLING = True
RATIO_UNDERSAMPLING = 0.03 
RATIO_UNDERSAMPLING_VAL_EXT = 0.075
RATIO_UNDERSAMPLING_TEST_FINAL = 0.2

def _tag_us():
    if (not APLICAR_UNDERSAMPLING) or (RATIO_UNDERSAMPLING >= 0.999):
        return "us100"
    return f"usopt{int(RATIO_UNDERSAMPLING * 100):03d}"

def _tag_us_val_ext():
    """Tag para undersampling de validación externa"""
    if (not APLICAR_UNDERSAMPLING) or (RATIO_UNDERSAMPLING_VAL_EXT >= 0.999):
        return "usv100"
    return f"usvalext{int(RATIO_UNDERSAMPLING_VAL_EXT * 100):03d}"

def _tag_us_test():
    """Tag para undersampling de test final"""
    if (not APLICAR_UNDERSAMPLING) or (RATIO_UNDERSAMPLING_TEST_FINAL >= 0.999):
        return "ust100"
    return f"ustestfinal{int(RATIO_UNDERSAMPLING_TEST_FINAL * 100):03d}"


def _tag_train():
    return f"tr{min(MESES_TRAIN_OPTUNA)}-{max(MESES_TRAIN_OPTUNA)}"

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
    n_seeds = len(SEMILLAS_ENSEMBLE)

    parts = [
        "lgbm",
        SUFIJO_FE,
        _tag_us(),           # undersampling train para optuna
        _tag_us_val_ext(),   # undersampling train para val externa
        _tag_us_test(),      # undersampling train para test final
        _tag_train(),
        _tag_list("val", MES_VAL_OPTUNA),
        _tag_list("vext", MES_VALID_EXT),
        _tag_test(),
        f"s{n_seeds}"  # cantidad de semillas usadas en ensemble
    ]

    return "_".join(p for p in parts if p)


NOMBRE_EXPERIMENTO = build_experiment_name()




# ==================================================================================
# ESTRUCTURA DE SALIDAS POR EXPERIMENTO
# ==================================================================================

EXPERIMENTS_ROOT = os.path.join(BUCKET_PATH_b1, FILE_BASE)
os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)

EXPERIMENT_DIR = os.path.join(EXPERIMENTS_ROOT, NOMBRE_EXPERIMENTO)

DB_PATH = os.path.join(EXPERIMENT_DIR, "db")
MODELOS_PATH = os.path.join(EXPERIMENT_DIR, "modelos")
LOGS_PATH = os.path.join(EXPERIMENT_DIR, "logs")
RESULTADOS_PREDICCION_PATH = os.path.join(EXPERIMENT_DIR, "resultados_prediccion")

for path in [EXPERIMENT_DIR, DB_PATH, MODELOS_PATH, LOGS_PATH, RESULTADOS_PREDICCION_PATH]:
    os.makedirs(path, exist_ok=True)

# ==================================================================================
# APO (A Prueba Overfiteros) sobre validación externa (p.ej. 202107)
# ==================================================================================

# Semillerio para APO (validación externa)
SEMILLAS_APO = SEMILLAS_ENSEMBLE   # por ej: 20 seeds para APO
APO_K_SEM = 10                          # 10 seeds por APO
APO_N_APO = 2                          # 2 repes

assert len(SEMILLAS_ENSEMBLE) >= APO_K_SEM * APO_N_APO, \
    f"ERROR: Se necesitan {APO_K_SEM * APO_N_APO} semillas para APO pero solo hay {len(SEMILLAS_ENSEMBLE)}."

# Lista de N candidatos (cantidad de envíos) a evaluar
APO_CORTES_ENVIO = [9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000]  

# Carpeta donde se guardan los modelos entrenados SOLO para validación externa
MODEL_DIR_VAL_EXT = os.path.join(EXPERIMENT_DIR, "modelos_val_ext")

# Directorio para modelos finales (train completo + test)
MODEL_DIR_TEST_FINAL = os.path.join(EXPERIMENT_DIR, "modelos_test_final")

# Directorio para resultados de predicción final (csv de envío)
RESULTADOS_PREDICCION_DIR = os.path.join(EXPERIMENT_DIR, "resultados_prediccion")

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
# PARAMS BASE DEL MODELO DE LIGHTGBM
# ==================================================================================


LGBM_PARAMS_BASE = {
    "objective": "binary",
    "metric": "None",                 
    "boosting_type": "gbdt",
    "first_metric_only": True,
    "boost_from_average": True,
    "feature_pre_filter": False,

    "verbosity": -1,                  
    "force_row_wise": True,           # evita warnings y suele ser más estable

    "max_depth": -1,
    "max_bin": 31,
    "lambda_l1": 0.0,
    "num_threads": -1,                 # usar todos los núcleos disponibles
}



# ==================================================================================
# COLUMNAS
# ==================================================================================

COLS_ID = ["foto_mes", "numero_de_cliente"]
ELIMINAR_COLUMNAS_ID = False

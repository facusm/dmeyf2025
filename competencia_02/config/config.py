# config/config.py
import os

# === CONFIGURACIÓN DE PATHS === #

# Ruta al bucket montado localmente 
BUCKET_PATH_b1 = "/home/sanmartinofacundo/buckets/b1/"

# Rutas derivadas dentro del bucket
DB_PATH = os.path.join(BUCKET_PATH_b1, "db")
MODELOS_PATH = os.path.join(BUCKET_PATH_b1, "modelos")
LOGS_PATH = os.path.join(BUCKET_PATH_b1, "logs")
RESULTADOS_PREDICCION_PATH = os.path.join(BUCKET_PATH_b1, "resultados_prediccion")

# Dataset crudo local
DATASET_CRUDO_PATH = "/home/sanmartinofacundo/datasets/competencia_02_crudo.csv.gz"

# Crear directorios dentro del bucket si no existen
for path in [DB_PATH, MODELOS_PATH, LOGS_PATH, RESULTADOS_PREDICCION_PATH]:
    os.makedirs(path, exist_ok=True)

# === CONFIGURACIÓN DE NOMBRES === #
VERSION = "v2"
BUCKET_NAME = "b1"
FILE_BASE = "competencia_02"
SUFIJO_FE = f"FE_{VERSION}"

# Archivos y nombres derivados
NOMBRE_BASE_DE_DATOS_OPTUNA = f"optimization_lgbm_{SUFIJO_FE}.db"
NOMBRE_DE_ESTUDIO_OPTUNA = f"lgbm_cv_{SUFIJO_FE}"
ARCHIVO_DATOS_CSV = f"{FILE_BASE}_{SUFIJO_FE}.csv"
NOMBRE_NOTEBOOK = f"CV_clasico_semillas_{SUFIJO_FE}"
NOMBRE_EXPERIMENTO = f"lgbm_cv_{SUFIJO_FE}"


# === CONFIGURACIÓN DE UNDERSAMPLING === #
APLICAR_UNDERSAMPLING = True  # Activar/desactivar undersampling
RATIO_UNDERSAMPLING = 0.4  

# === CONFIGURACIÓN DE MESES Y SEMILLAS === #
MESES_TRAIN = [202101, 202102, 202103]  # Entrenamiento Optuna con cv clásico
MES_VALID = [202104] # Validación
MES_TEST_FINAL = [202106]  # Predicción final

SEMILLAS = [181459, 306491, 336251, 900577, 901751, 182009, 182011, 182027, 182029, 182041]

# === CONFIGURACIÓN DE NEGOCIO === #
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = -20000  # Negativo porque es un costo


# === CONFIGURACIÓN DE MODELO === #
PARAMS = {
    "ganancia_acierto": GANANCIA_ACIERTO,
    "costo_estimulo": COSTO_ESTIMULO,
    "n_folds": 5,
    "num_boost_round": 2000,
    "early_stopping_rounds": 50,
    "target": "clase_binaria2",  
}

# === CONFIGURACIÓN DE COLUMNAS === #
COLS_ID = ['foto_mes', 'numero_de_cliente']
ELIMINAR_COLUMNAS_ID = False  # Cambiar a True si deseas eliminarlas

# === CONFIGURACIÓN DE OPTUNA === #
N_TRIALS = 5
N_STARTUP_TRIALS = 2  # Trials aleatorios iniciales



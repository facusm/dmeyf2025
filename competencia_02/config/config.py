# config/config.py
import os

# === CONFIGURACIÓN DE PATHS === #
BASE_PATH = os.getcwd()
DATASET_PATH = os.path.join(BASE_PATH, 'datos')
MODELOS_PATH = os.path.join(BASE_PATH, 'modelos')
DB_PATH = os.path.join(BASE_PATH, 'db')
SUBMISSIONS_PATH = os.path.join(BASE_PATH, 'submissions')

# Crear directorios si no existen
for path in [DATASET_PATH, MODELOS_PATH, DB_PATH, SUBMISSIONS_PATH]:
    os.makedirs(path, exist_ok=True)

# === CONFIGURACIÓN DE NOMBRES === #
SUFIJO_FE = "FE_v1"

NOMBRE_BASE_DE_DATOS_OPTUNA = f"optimization_lgbm_{SUFIJO_FE}.db"
NOMBRE_DE_ESTUDIO_OPTUNA = f"lgbm_cv_{SUFIJO_FE}"
ARCHIVO_DATOS_CSV = f"competencia_02_{SUFIJO_FE}.csv"
NOMBRE_NOTEBOOK = f"CV_clasico_semillas_{SUFIJO_FE}"
NOMBRE_EXPERIMENTO = f"lgbm_cv_{SUFIJO_FE}"

# === CONFIGURACIÓN DE DATASETS === #
DATASET_PROCESADO_PATH = os.path.join(DATASET_PATH, ARCHIVO_DATOS_CSV)

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
    "target": "clase_binaria2",  # o "clase_binaria1"
}

# === CONFIGURACIÓN DE COLUMNAS === #
COLS_ID = ['foto_mes', 'numero_de_cliente']
ELIMINAR_COLUMNAS_ID = False  # Cambiar a True si deseas eliminarlas

# === CONFIGURACIÓN DE OPTUNA === #
N_TRIALS = 5
N_STARTUP_TRIALS = 2  # Trials aleatorios iniciales



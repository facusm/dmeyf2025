import pandas as pd
import os
import datetime
import logging
import time
import duckdb

from src.data_load_preparation import cargar_datos
from .features import (
    feature_engineering_lag,
    feature_engineering_min_max,
    feature_engineering_deltas,
    feature_engineering_medias_moviles,
    feature_engineering_cum_sum,
    feature_engineering_ratios,
    feature_engineering_medias_moviles_lag,
    generar_shock_relativo_delta_lag,
    crear_indicador_aguinaldo,
)
from config.config import BUCKET_PATH_b1, FILE_BASE, VERSION


# ===========================
# CONFIGURACIÓN DE LOGGING
# ===========================
LOG_DIR = os.path.join(BUCKET_PATH_b1, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, nombre_log), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================
def main():
    inicio = time.time()
    logger.info("🚀 Inicio de ejecución de Feature Engineering")

    # 00. DEFINICIÓN DE PATHS
    path_input = os.path.join(BUCKET_PATH_b1, f"{FILE_BASE}.csv.gz")
    path_output = os.path.join(BUCKET_PATH_b1, f"{FILE_BASE}_FE_{VERSION}.csv.gz")

    logger.info(f"📥 Leyendo dataset desde: {path_input}")
    logger.info(f"📤 Guardando resultado en: {path_output}")

    # 01. CARGA DE DATOS
    df = pd.read_csv(path_input, compression="gzip")
    df.drop(columns=["mprestamos_personales", "cprestamos_personales"], inplace=True, errors="ignore")
    logger.info(f"✅ Dataset cargado correctamente con forma: {df.shape}")

    # 02. FEATURE ENGINEERING
    atributos = [
        "mrentabilidad", "mcomisiones", "mpasivos_margen", "mcaja_ahorro",
        "mcaja_ahorro_dolares", "mcuentas_saldo", "ctarjeta_debito_transacciones",
        "mautoservicio", "ctarjeta_visa_transacciones", "mtarjeta_visa_consumo",
        "ctarjeta_master_transacciones", "mtarjeta_master_consumo",
        "mprestamos_prendarios", "mprestamos_hipotecarios", "mplazo_fijo_dolares",
        "mplazo_fijo_pesos", "cpayroll_trx", "cpayroll2_trx", "mcomisiones_mantenimiento",
        "ctrx_quarter", "Master_mlimitecompra", "Master_mconsumototal",
        "Visa_mlimitecompra", "Visa_mconsumototal", "mactivos_margen",
        "mcuenta_corriente", "mcuenta_corriente_adicional", "mcaja_ahorro_adicional",
        "minversion1_pesos", "minversion1_dolares", "minversion2"
    ]

    ratio_pairs = [
        ("Master_mconsumototal", "Master_mlimitecompra"),
        ("Visa_mconsumototal", "Visa_mlimitecompra"),
        ("mtarjeta_master_consumo", "Master_mlimitecompra"),
        ("mtarjeta_visa_consumo", "Visa_mlimitecompra"),
        ("mprestamos_prendarios", "mcuentas_saldo"),
        ("mprestamos_hipotecarios", "mcuentas_saldo"),
        ("mcaja_ahorro", "mcuentas_saldo"),
        ("mcaja_ahorro_dolares", "mcuentas_saldo"),
        ("mcomisiones", "mrentabilidad"),
        ("mcuenta_corriente", "mcuentas_saldo"),
        ("mcuenta_corriente_adicional", "mcuentas_saldo"),
        ("minversion1_pesos", "mcuentas_saldo"),
        ("minversion1_dolares", "mcuentas_saldo"),
        ("minversion2", "mcuentas_saldo"),
        ("mactivos_margen", "mpasivos_margen"),
        ("mcomisiones_mantenimiento", "mcomisiones"),
        ("mtarjeta_master_consumo", "mcuentas_saldo"),
        ("mtarjeta_visa_consumo", "mcuentas_saldo")
    ]

    cant_lag = 2
    window_size = 2

    logger.info("🔧 Iniciando feature engineering...")

    df_fe = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_ratios(df_fe, ratio_pairs=ratio_pairs)
    df_fe = crear_indicador_aguinaldo(df_fe)

    logger.info(f"✅ Feature engineering finalizado. Forma resultante: {df_fe.shape}")

    # 03. GUARDAR DIRECTAMENTE EN EL BUCKET (optimizado con DuckDB)
    logger.info("💾 Guardando dataset final directamente en el bucket (modo rápido con DuckDB)...")

    con = duckdb.connect(database=':memory:')
    con.register("df_fe", df_fe)

    con.execute(f"""
        COPY df_fe 
        TO '{path_output}' 
        (FORMAT CSV, HEADER, COMPRESSION GZIP);
    """)

    file_size_mb = os.path.getsize(path_output) / (1024 * 1024)
    logger.info(f"✅ Archivo guardado eficientemente en el bucket: {path_output} ({file_size_mb:.2f} MB)")

    # 04. DURACIÓN TOTAL
    duracion_min = (time.time() - inicio) / 60
    logger.info(f"⏱️ Duración total del proceso: {duracion_min:.2f} minutos")

    logger.info("🎯 Ejecución finalizada correctamente.")


if __name__ == "__main__":
    main()

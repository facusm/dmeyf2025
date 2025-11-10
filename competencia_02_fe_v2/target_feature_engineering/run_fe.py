import pandas as pd
import os
import datetime
import logging
import time
import duckdb

from src.data_load_preparation import cargar_datos
from .features import (
    pisar_con_mes_anterior_duckdb,
    feature_engineering_lag,
    feature_engineering_min_max,
    feature_engineering_deltas,
    feature_engineering_medias_moviles,
    feature_engineering_cum_sum,
    feature_engineering_ratios,
    feature_engineering_medias_moviles_lag,
    generar_shock_relativo_delta_lag,
    crear_indicador_aguinaldo
)
from config.config import (
    DATASET_TARGETS_CREADOS_PATH,
    FE_PATH,
    LOGS_PATH,
    SUFIJO_FE,
    NOMBRE_EXPERIMENTO,
)


# ===========================
# CONFIGURACIÓN DE LOGGING
# ===========================
# Logs específicos del proceso de Feature Engineering dentro del experimento
LOG_DIR = os.path.join(LOGS_PATH, "fe")
os.makedirs(LOG_DIR, exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"fe_{SUFIJO_FE}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, nombre_log), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================
def main():
    inicio = time.time()
    logger.info("🚀 Inicio de ejecución de Feature Engineering")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"🏷️ Versión FE: {SUFIJO_FE}")

    # 00. DEFINICIÓN DE PATHS
    path_input = DATASET_TARGETS_CREADOS_PATH
    path_output = FE_PATH

    logger.info(f"📥 Leyendo dataset crudo desde: {path_input}")
    logger.info(f"📤 Guardando dataset FE en formato Parquet: {path_output}")

    # 01. CARGA DE DATOS
    # Podés cambiar a `cargar_datos(path_input)` si lo preferís,
    # pero acá mantenemos lectura directa del crudo comprimido.
    df = pd.read_csv(path_input, compression="gzip")
    logger.info(f"✅ Dataset cargado correctamente con forma: {df.shape}")

    # 01.1 Eliminación de columnas con posible data drift / poco confiables
    logger.info("🧹 Eliminando columnas con posible data drift / poco confiables...")
    df.drop(
        columns=[
            "mprestamos_personales",
            "cprestamos_personales",
            "internet",
            "cpagodeservicios",
            "mpagodeservicios",
            "tmobile_app",
            "cmobile_app_trx",
        ],
        inplace=True,
        errors="ignore",
    )
    logger.info("✅ Columnas conflictivas eliminadas (si existían)")

    # 01.2 Corrección de meses anómalos usando valores del mes anterior
    logger.info("🩺 Corrigiendo meses anómalos usando valores del mes anterior...")

    df = pisar_con_mes_anterior_duckdb(df, variable="active_quarter", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mrentabilidad", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mrentabilidad_annual", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcomisiones", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mpasivos_margen", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcuentas_saldo", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ctarjeta_debito_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mautoservicio", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ctarjeta_visa_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mtarjeta_visa_consumo", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ctarjeta_master_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mtarjeta_master_consumo", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ctarjeta_visa_debitos_automaticos", meses_anomalos=[201904])
    df = pisar_con_mes_anterior_duckdb(df, variable="mttarjeta_visa_debitos_automaticos", meses_anomalos=[201904])
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="ccajeros_propios_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="mcajeros_propios_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="ctarjeta_visa_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="mtarjeta_visa_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="ctarjeta_master_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="mtarjeta_master_descuentos",
        meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102],
    )
    df = pisar_con_mes_anterior_duckdb(df, variable="ccomisiones_otras", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcomisiones_otras", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="cextraccion_autoservicio", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mextraccion_autoservicio", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccheques_depositados", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcheques_depositados", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccheques_emitidos", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcheques_emitidos", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="ccheques_depositados_rechazados",
        meses_anomalos=[202006],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="mcheques_depositados_rechazados",
        meses_anomalos=[202006],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="ccheques_emitidos_rechazados",
        meses_anomalos=[202006],
    )
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="mcheques_emitidos_rechazados",
        meses_anomalos=[202006],
    )
    df = pisar_con_mes_anterior_duckdb(df, variable="tcallcenter", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccallcenter_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="thomebanking", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(
        df,
        variable="chomebanking_transacciones",
        meses_anomalos=[201910, 202006],
    )
    df = pisar_con_mes_anterior_duckdb(df, variable="ccajas_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccajas_consultas", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccajas_depositos", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccajas_extracciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccajas_otras", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="catm_trx", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="matm", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="catm_trx_other", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="matm_other", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ctrx_quarter", meses_anomalos=[202006])

    logger.info("✅ Corrección de meses anómalos finalizada")

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
        "minversion1_pesos", "minversion1_dolares", "minversion2",
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
        ("mtarjeta_visa_consumo", "mcuentas_saldo"),
    ]

    cant_lag = 4
    window_size = 4

    logger.info("🔧 Iniciando feature engineering...")

    df_fe = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_ratios(df_fe, ratio_pairs=ratio_pairs)
    df_fe = crear_indicador_aguinaldo(df_fe)

    logger.info(f"✅ Feature engineering finalizado. Forma resultante: {df_fe.shape}")

    # 03. GUARDAR EN PARQUET (DUCKDB) EN LA RUTA DEL EXPERIMENTO
    logger.info("💾 Guardando dataset FE en formato Parquet con compresión ZSTD...")

    os.makedirs(os.path.dirname(path_output), exist_ok=True)

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df_fe", df_fe)
        con.execute(
            f"""
            COPY df_fe
            TO '{path_output}'
            (FORMAT PARQUET, COMPRESSION 'ZSTD');
            """
        )
    finally:
        con.close()

    try:
        file_size_mb = os.path.getsize(path_output) / (1024 * 1024)
        logger.info(f"✅ Archivo guardado: {path_output} ({file_size_mb:.2f} MB)")
    except OSError:
        logger.info(f"✅ Archivo guardado en: {path_output} (no se pudo leer tamaño local)")

    # 04. DURACIÓN TOTAL
    duracion_min = (time.time() - inicio) / 60
    logger.info(f"⏱️ Duración total del proceso: {duracion_min:.2f} minutos")
    logger.info("🎯 Ejecución de Feature Engineering finalizada correctamente.")


if __name__ == "__main__":
    main()

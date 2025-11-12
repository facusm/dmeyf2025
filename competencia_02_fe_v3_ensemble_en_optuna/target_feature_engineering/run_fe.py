# run_fe.py  (versión conservadora)
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
# LOGGING
# ===========================
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
# HELPERS LOCF / FLAGS
# ===========================
def aplicar_locf_con_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica LOCF (pisado hacia atrás) sobre las variables indicadas y
    deja en el dataframe los flags <variable>_locf creados por la función.
    Devuelve el df ya corregido.
    """
    logger.info("🩺 Corrigiendo meses anómalos usando valores del mes anterior (LOCF con flags)...")

    # >>>> LISTA DE CORRECCIONES (con tus mismos meses) <<<<
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

    for v in ["ccajeros_propios_descuentos", "mcajeros_propios_descuentos",
              "ctarjeta_visa_descuentos", "mtarjeta_visa_descuentos",
              "ctarjeta_master_descuentos", "mtarjeta_master_descuentos"]:
        df = pisar_con_mes_anterior_duckdb(
            df, variable=v,
            meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102]
        )

    df = pisar_con_mes_anterior_duckdb(df, variable="ccomisiones_otras", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcomisiones_otras", meses_anomalos=[201905, 201910, 202006])

    for v in ["cextraccion_autoservicio", "mextraccion_autoservicio",
              "ccheques_depositados", "mcheques_depositados",
              "ccheques_emitidos", "mcheques_emitidos",
              "ccheques_depositados_rechazados", "mcheques_depositados_rechazados",
              "ccheques_emitidos_rechazados", "mcheques_emitidos_rechazados"]:
        df = pisar_con_mes_anterior_duckdb(df, variable=v, meses_anomalos=[202006])

    df = pisar_con_mes_anterior_duckdb(df, variable="tcallcenter", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccallcenter_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="thomebanking", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="chomebanking_transacciones", meses_anomalos=[201910, 202006])

    for v in ["ccajas_transacciones", "ccajas_consultas", "ccajas_depositos",
              "ccajas_extracciones", "ccajas_otras", "catm_trx", "matm",
              "catm_trx_other", "matm_other", "ctrx_quarter"]:
        df = pisar_con_mes_anterior_duckdb(df, variable=v, meses_anomalos=[202006])

    logger.info("✅ Corrección LOCF finalizada (flags *_locf generados)")

    return df


def agregar_contadores_locf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea contadores por familia y un total de flags LOCF aplicados.
    NO genera lags/deltas sobre estos contadores (se usan como features estáticas).
    """
    # Detectar todos los flags generados por LOCF
    locf_flags = [c for c in df.columns if c.endswith("_locf")]

    # Total de variables corregidas ese mes/cliente
    df["n_locf_vars"] = df[locf_flags].sum(axis=1) if locf_flags else 0

    # Grupos (ajustá listas si cambias variables corregidas)
    grupos = {
        "tarjetas_locf": [
            "ctarjeta_visa_transacciones_locf", "mtarjeta_visa_consumo_locf",
            "ctarjeta_master_transacciones_locf", "mtarjeta_master_consumo_locf",
            "ctarjeta_visa_debitos_automaticos_locf", "mttarjeta_visa_debitos_automaticos_locf",
            "ctarjeta_visa_descuentos_locf", "mtarjeta_visa_descuentos_locf",
            "ctarjeta_master_descuentos_locf", "mtarjeta_master_descuentos_locf",
        ],
        "cajas_locf": [
            "ccajas_transacciones_locf", "ccajas_consultas_locf",
            "ccajas_depositos_locf", "ccajas_extracciones_locf", "ccajas_otras_locf",
        ],
        "canales_locf": [
            "thomebanking_locf", "chomebanking_transacciones_locf",
            "tcallcenter_locf", "ccallcenter_transacciones_locf",
        ],
        "atm_locf": [
            "catm_trx_locf", "matm_locf", "catm_trx_other_locf", "matm_other_locf",
        ],
        "cheques_locf": [
            "ccheques_depositados_locf", "mcheques_depositados_locf",
            "ccheques_emitidos_locf", "mcheques_emitidos_locf",
            "ccheques_depositados_rechazados_locf", "mcheques_depositados_rechazados_locf",
            "ccheques_emitidos_rechazados_locf", "mcheques_emitidos_rechazados_locf",
        ],
        "comisiones_locf": [
            "mcomisiones_locf", "ccomisiones_otras_locf", "mcomisiones_otras_locf",
            "mcomisiones_mantenimiento_locf"  # si en algún momento la agregás al LOCF
        ],
        "desc_propios_locf": [
            "ccajeros_propios_descuentos_locf", "mcajeros_propios_descuentos_locf",
        ],
        "saldos_margenes_locf": [
            "mcuentas_saldo_locf", "mpasivos_margen_locf", "mactivos_margen_locf",
        ],
        "rentabilidad_locf": [
            "mrentabilidad_locf", "mrentabilidad_annual_locf",
        ],
        "sucursales_locf": [
            "mautoservicio_locf",
        ],
        "otros_locf": [
            "active_quarter_locf", "ctrx_quarter_locf",
        ],
    }

    for nombre, cols in grupos.items():
        cols_validas = [c for c in cols if c in df.columns]
        if cols_validas:
            df[nombre] = df[cols_validas].sum(axis=1)
        else:
            df[nombre] = 0

    logger.info("🧮 Contadores LOCF agregados (n_locf_vars y familias)")
    return df


# ===========================
# MAIN
# ===========================
def main():
    inicio = time.time()
    logger.info("🚀 Inicio de ejecución de Feature Engineering")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"🏷️ Versión FE: {SUFIJO_FE}")

    # Paths
    path_input = DATASET_TARGETS_CREADOS_PATH
    path_output = FE_PATH

    logger.info(f"📥 Leyendo dataset crudo desde: {path_input}")
    logger.info(f"📤 Guardando dataset FE en formato Parquet: {path_output}")

    # 01. CARGA
    df = pd.read_csv(path_input, compression="gzip")
    logger.info(f"✅ Dataset cargado correctamente con forma: {df.shape}")

    # 01.1 DROPS por posible data drift / poca confianza
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

    # 01.2 LOCF + FLAGS
    df = aplicar_locf_con_flags(df)

    # 01.3 Contadores/agrupadores de flags (features estáticas)
    df = agregar_contadores_locf(df)

    # 02. FEATURE ENGINEERING (versión conservadora)
    # Importante: **NO** incluir *_locf ni los contadores en listas de atributos/ratios/etc.
    atributos = [
        # Rentabilidad / comisiones / márgenes
        "mrentabilidad", "mcomisiones", "mcomisiones_mantenimiento", "mpasivos_margen", "mactivos_margen",

        # Saldos y cajas
        "mcuentas_saldo", "mcaja_ahorro", "mcaja_ahorro_dolares",
        "mcuenta_corriente", "mcuenta_corriente_adicional", "mcaja_ahorro_adicional",

        # Débito / cajeros / sucursales
        "ctarjeta_debito_transacciones", "mautoservicio",
        "catm_trx", "matm", "catm_trx_other", "matm_other",
        "ccajas_transacciones",

        # Canales digitales / call center
        "thomebanking", "chomebanking_transacciones", "tcallcenter", "ccallcenter_transacciones",

        # Tarjetas (resúmenes)
        "ctarjeta_visa_transacciones", "mtarjeta_visa_consumo",
        "ctarjeta_master_transacciones", "mtarjeta_master_consumo",
        "Master_msaldototal", "Master_mconsumototal", "Master_mpagado",
        "Master_mlimitecompra", "Master_madelantopesos",
        "Visa_msaldototal", "Visa_mconsumototal", "Visa_mpagado",
        "Visa_mlimitecompra", "Visa_madelantopesos",

        # Préstamos (se mantienen prend/hipo)
        "mprestamos_prendarios", "mprestamos_hipotecarios",

        # Inversiones / ahorro estructural
        "mplazo_fijo_pesos", "mplazo_fijo_dolares",
        "minversion1_pesos", "minversion1_dolares", "minversion2",

        # Payroll
        "cpayroll_trx", "cpayroll2_trx",

        # Actividad global
        "ctrx_quarter",
    ]

    # Ratios (sin *_locf ni contadores)
    ratio_pairs = [
        # Utilización / presión sobre límite de tarjetas
        ("Master_msaldototal", "Master_mlimitecompra"),
        ("Visa_msaldototal", "Visa_mlimitecompra"),
        ("Master_mconsumototal", "Master_mlimitecompra"),
        ("Visa_mconsumototal", "Visa_mlimitecompra"),

        # Adelantos vs consumo (estrés)
        ("Master_madelantopesos", "Master_mconsumototal"),
        ("Visa_madelantopesos", "Visa_mconsumototal"),

        # Préstamos vs saldo total
        ("mprestamos_prendarios", "mcuentas_saldo"),
        ("mprestamos_hipotecarios", "mcuentas_saldo"),

        # Inversiones vs saldo
        ("minversion1_pesos", "mcuentas_saldo"),
        ("minversion1_dolares", "mcuentas_saldo"),
        ("minversion2", "mcuentas_saldo"),
        ("mplazo_fijo_pesos", "mcuentas_saldo"),
        ("mplazo_fijo_dolares", "mcuentas_saldo"),

        # Liquidez vs saldo
        ("mcaja_ahorro", "mcuentas_saldo"),
        ("mcuenta_corriente", "mcuentas_saldo"),

        # Comisiones vs rentabilidad
        ("mcomisiones", "mrentabilidad"),

        # Penetración digital sobre actividad total
        ("chomebanking_transacciones", "ctrx_quarter"),
    ]

    # Cumsum de “engagement” (sin *_locf)
    cumsum_cols = [
        "ctrx_quarter", "cpayroll_trx", "cpayroll2_trx",
        "chomebanking_transacciones", "ccallcenter_transacciones",
        "ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones", "ccajas_transacciones",
    ]

    # Min/max histórico por cliente (sin *_locf)
    minmax_cols = [
        "mcuentas_saldo", "mcaja_ahorro", "mcaja_ahorro_dolares",
        "mcuenta_corriente", "mplazo_fijo_pesos", "mplazo_fijo_dolares",
        "minversion1_pesos", "minversion1_dolares", "minversion2",
        "Master_mlimitecompra", "Visa_mlimitecompra",
        "Master_msaldototal", "Visa_msaldototal",
    ]

    # Parámetros de ventanas (conservador)
    cant_lag = 3
    window_size = 3

    logger.info("🔧 Iniciando feature engineering...")

    # IMPORTANTE: los contadores *_locf y n_locf_vars NO pasan por estas funciones
    df_fe = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributos, cant_lag=cant_lag)
    df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos, window_size=window_size)
    df_fe = feature_engineering_ratios(df_fe, ratio_pairs=ratio_pairs)
    df_fe = feature_engineering_cum_sum(df_fe, columnas=cumsum_cols)
    df_fe = feature_engineering_min_max(df_fe, columnas=minmax_cols)
    df_fe = crear_indicador_aguinaldo(df_fe)

    logger.info(f"✅ Feature engineering finalizado. Forma resultante: {df_fe.shape}")

    # 03. GUARDAR PARQUET
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

    # 04. DURACIÓN
    duracion_min = (time.time() - inicio) / 60
    logger.info(f"⏱️ Duración total del proceso: {duracion_min:.2f} minutos")
    logger.info("🎯 Ejecución de Feature Engineering finalizada correctamente.")


if __name__ == "__main__":
    main()

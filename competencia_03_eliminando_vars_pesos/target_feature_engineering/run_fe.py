# run_fe.py  (conservadora + drift correction para pesos + cumsum/minmax ventana 6)
import pandas as pd
import os
import datetime
import logging
import time
import duckdb

from .features import (
    pisar_con_mes_anterior_duckdb,
    agregar_drift_features_monetarias,
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
# VARIABLES EN PESOS
# ===========================
VARIABLES_PESOS = [
    "mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen",
    "mcuenta_corriente_adicional", "mcuenta_corriente", "mcaja_ahorro", "mcaja_ahorro_adicional",
    "mcaja_ahorro_dolares", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo",
    "mtarjeta_master_consumo", "mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios",
    "mplazo_fijo_dolares", "mplazo_fijo_pesos", "minversion1_pesos", "minversion1_dolares", "minversion2",
    "mpayroll", "mpayroll2", "mcuenta_debitos_automaticos", "mttarjeta_visa_debitos_automaticos",
    "mttarjeta_master_debitos_automaticos", "mpagodeservicios", "mpagomiscuentas",
    "mcajeros_propios_descuentos", "mtarjeta_visa_descuentos", "mtarjeta_master_descuentos",
    "mcomisiones_mantenimiento", "mcomisiones_otras", "mforex_buy", "mforex_sell",
    "mtransferencias_recibidas", "mtransferencias_emitidas", "mextraccion_autoservicio",
    "mcheques_depositados", "mcheques_emitidos", "mcheques_depositados_rechazados",
    "mcheques_emitidos_rechazados", "matm", "matm_other",
    "Master_mfinanciacion_limite", "Master_msaldototal", "Master_msaldopesos", "Master_msaldodolares",
    "Master_mconsumospesos", "Master_mconsumosdolares", "Master_mlimitecompra", "Master_madelantopesos",
    "Master_madelantodolares", "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares",
    "Master_mconsumototal", "Master_mpagominimo",
    "Visa_mfinanciacion_limite", "Visa_msaldototal", "Visa_msaldopesos", "Visa_msaldodolares",
    "Visa_mconsumospesos", "Visa_mconsumosdolares", "Visa_mlimitecompra", "Visa_madelantopesos",
    "Visa_madelantodolares", "Visa_mpagado", "Visa_mpagospesos", "Visa_mpagosdolares",
    "Visa_mconsumototal", "Visa_mpagominimo",
]


# ===========================
# HELPERS LOCF / FLAGS
# ===========================
def aplicar_locf_con_flags(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("🩺 Corrigiendo meses anómalos usando valores del mes anterior (LOCF con flags)...")

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

    for v in [
        "ccajeros_propios_descuentos", "mcajeros_propios_descuentos",
        "ctarjeta_visa_descuentos", "mtarjeta_visa_descuentos",
        "ctarjeta_master_descuentos", "mtarjeta_master_descuentos",
    ]:
        df = pisar_con_mes_anterior_duckdb(
            df, variable=v,
            meses_anomalos=[201910, 202002, 202006, 202009, 202010, 202102]
        )

    df = pisar_con_mes_anterior_duckdb(df, variable="ccomisiones_otras", meses_anomalos=[201905, 201910, 202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="mcomisiones_otras", meses_anomalos=[201905, 201910, 202006])

    for v in [
        "cextraccion_autoservicio", "mextraccion_autoservicio",
        "ccheques_depositados", "mcheques_depositados",
        "ccheques_emitidos", "mcheques_emitidos",
        "ccheques_depositados_rechazados", "mcheques_depositados_rechazados",
        "ccheques_emitidos_rechazados", "mcheques_emitidos_rechazados",
    ]:
        df = pisar_con_mes_anterior_duckdb(df, variable=v, meses_anomalos=[202006])

    df = pisar_con_mes_anterior_duckdb(df, variable="tcallcenter", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="ccallcenter_transacciones", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="thomebanking", meses_anomalos=[202006])
    df = pisar_con_mes_anterior_duckdb(df, variable="chomebanking_transacciones", meses_anomalos=[201910, 202006])

    for v in [
        "ccajas_transacciones", "ccajas_consultas", "ccajas_depositos",
        "ccajas_extracciones", "ccajas_otras", "catm_trx", "matm",
        "catm_trx_other", "matm_other", "ctrx_quarter",
    ]:
        df = pisar_con_mes_anterior_duckdb(df, variable=v, meses_anomalos=[202006])

    logger.info("✅ Corrección LOCF finalizada (flags *_locf generados)")
    return df


def agregar_contadores_locf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suma total de flags LOCF y algunos contadores por familia (si existen).
    """
    locf_flags = [c for c in df.columns if c.endswith("_locf")]
    df["n_locf_vars"] = df[locf_flags].sum(axis=1) if locf_flags else 0

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
            "mcomisiones_mantenimiento_locf",
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
        df[nombre] = df[cols_validas].sum(axis=1) if cols_validas else 0

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

    path_input = DATASET_TARGETS_CREADOS_PATH
    path_output = FE_PATH

    logger.info(f"📥 Leyendo dataset crudo desde: {path_input}")
    logger.info(f"📤 Guardando dataset FE en formato Parquet: {path_output}")

    # 01. CARGA
    df = pd.read_csv(path_input, compression="gzip")
    logger.info(f"✅ Dataset cargado correctamente con forma: {df.shape}")

    # 01.1 DROPS por poca confianza
    logger.info("🧹 Eliminando columnas con posible data drift / poco confiables...")
    df.drop(
        columns=["internet", "cpagodeservicios", "mpagodeservicios", "tmobile_app", "cmobile_app_trx"],
        inplace=True,
        errors="ignore",
    )
    logger.info("✅ Columnas conflictivas eliminadas (si existían)")

    # 01.2 LOCF + FLAGS
    df = aplicar_locf_con_flags(df)

    # 01.3 Contadores LOCF (features estáticas)
    df = agregar_contadores_locf(df)

    # ===========================
    # 02. ATRIBUTOS (no-pesos vs pesos)
    # ===========================
    atributos_no_pesos = [
        "ctarjeta_debito_transacciones",
        "catm_trx", "catm_trx_other",
        "ccajas_transacciones",
        "thomebanking", "chomebanking_transacciones",
        "tcallcenter", "ccallcenter_transacciones",
        "ctarjeta_visa_transacciones", "ctarjeta_master_transacciones",
        "cpayroll_trx", "cpayroll2_trx",
        "ctrx_quarter",
    ]
    atributos_no_pesos = [c for c in atributos_no_pesos if c in df.columns]
    atributos_pesos = [c for c in VARIABLES_PESOS if c in df.columns]
    logger.info(f"📌 atributos_no_pesos: {len(atributos_no_pesos)} | atributos_pesos: {len(atributos_pesos)}")

    # ===========================
    # 03. Drift features SOLO pesos => genera *_rz_mes
    # ===========================
    # 03. Drift features SOLO pesos => genera *_rz_mes
    if atributos_pesos:
        logger.info("💸 Generando drift-features monetarias: creando *_rz_mes")
        df = agregar_drift_features_monetarias(df, atributos_pesos)

        # ✅ Drop de montos nominales para no reintroducir drift + bajar columnas/RAM
        cols_drop = [c for c in atributos_pesos if c in df.columns]
        logger.info(f"🧹 Dropeando montos nominales en pesos: {len(cols_drop)} columnas (se conserva *_rz_mes)")
        df.drop(columns=cols_drop, inplace=True, errors="ignore")
    else:
        logger.info("ℹ️ No hay atributos_pesos presentes; no se crea *_rz_mes ni se dropean nominales.")

    logger.info(f"📐 Dataset luego de drift+drop nominales: shape={df.shape}")


    # ===========================
    # 04. FE TEMPORAL (lags/deltas/MA/shock)
    # ===========================
    atributos_temporales = atributos_no_pesos + [
        f"{c}_rz_mes" for c in atributos_pesos if f"{c}_rz_mes" in df.columns
    ]

    cant_lag = 3
    window_size_ma = 3

    logger.info("🔧 Iniciando feature engineering temporal...")
    df_fe = feature_engineering_lag(df, columnas=atributos_temporales, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributos_temporales, cant_lag=cant_lag)
    df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos_temporales, window_size=window_size_ma)
    df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos_temporales, window_size=window_size_ma)
    df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos_temporales, window_size=window_size_ma)

    # ===========================
    # 05. FE NO TEMPORAL (ratios / cumsum / minmax / aguinaldo)
    # ===========================
    ratio_pairs = [
        ("Master_msaldototal", "Master_mlimitecompra"),
        ("Visa_msaldototal", "Visa_mlimitecompra"),
        ("Master_mconsumototal", "Master_mlimitecompra"),
        ("Visa_mconsumototal", "Visa_mlimitecompra"),
        ("Master_madelantopesos", "Master_mconsumototal"),
        ("Visa_madelantopesos", "Visa_mconsumototal"),
        ("mprestamos_prendarios", "mcuentas_saldo"),
        ("mprestamos_hipotecarios", "mcuentas_saldo"),
        ("minversion1_pesos", "mcuentas_saldo"),
        ("minversion1_dolares", "mcuentas_saldo"),
        ("minversion2", "mcuentas_saldo"),
        ("mplazo_fijo_pesos", "mcuentas_saldo"),
        ("mplazo_fijo_dolares", "mcuentas_saldo"),
        ("mcaja_ahorro", "mcuentas_saldo"),
        ("mcuenta_corriente", "mcuentas_saldo"),
        ("mcomisiones", "mrentabilidad"),
        ("chomebanking_transacciones", "ctrx_quarter"),
    ]
    ratio_pairs = [(a, b) for (a, b) in ratio_pairs if a in df_fe.columns and b in df_fe.columns]

    cumsum_cols = [
        "ctrx_quarter", "cpayroll_trx", "cpayroll2_trx",
        "chomebanking_transacciones", "ccallcenter_transacciones",
        "ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones", "ccajas_transacciones",
    ]
    cumsum_cols = [c for c in cumsum_cols if c in df_fe.columns]

    # min/max sobre rz_mes (desinflado por mes)
    minmax_cols_corr = [f"{c}_rz_mes" for c in atributos_pesos if f"{c}_rz_mes" in df_fe.columns]

    WINDOW_STATS = 6
    STRICT = False  # si quisieras exigir ventana completa, ponelo en True

    logger.info("🔧 Iniciando feature engineering no-temporal (ratios/cumsum/minmax)...")
    df_fe = feature_engineering_ratios(df_fe, ratio_pairs=ratio_pairs)
    df_fe = feature_engineering_cum_sum(df_fe, columnas=cumsum_cols, window_size=WINDOW_STATS, strict=STRICT)
    df_fe = feature_engineering_min_max(df_fe, columnas=minmax_cols_corr, window_size=WINDOW_STATS, strict=STRICT)
    df_fe = crear_indicador_aguinaldo(df_fe)

    logger.info(f"✅ Feature engineering finalizado. Forma resultante: {df_fe.shape}")

    # ===========================
    # 06. GUARDAR PARQUET
    # ===========================
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

    duracion_min = (time.time() - inicio) / 60
    logger.info(f"⏱️ Duración total del proceso: {duracion_min:.2f} minutos")
    logger.info("🎯 Ejecución de Feature Engineering finalizada correctamente.")


if __name__ == "__main__":
    main()

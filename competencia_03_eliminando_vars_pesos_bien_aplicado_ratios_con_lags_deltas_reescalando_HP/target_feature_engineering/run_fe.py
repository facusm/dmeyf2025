# run_fe.py  

import numpy as np
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
    feature_engineering_tendencia,
    detectar_variables_rotas_por_mes,
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

    # 01. CARGA + CORRECCIÓN DE DATOS ROTOS
    df = pd.read_csv(path_input, compression="gzip")
    logger.info(f"✅ Dataset cargado correctamente con forma: {df.shape}")

    logger.info("🔍 Iniciando corrección de datos rotos por mes...")

    excluir = {"foto_mes", "numero_de_cliente", "clase_ternaria"}
    columnas_a_chequear = [c for c in df.columns if c not in excluir]
    
    rotos = detectar_variables_rotas_por_mes(df, columnas=columnas_a_chequear, strict=True)
    
    for mes, columnas in rotos.items():
        mask = df["foto_mes"] == mes
        df.loc[mask, columnas] = np.nan

    logger.info({m: len(cols) for m, cols in rotos.items()})
    logger.info("✅ Corrección de datos rotos completada.")
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
    # 03. RATIOS (ANTES de dropear nominales)
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
    ratio_pairs = [(a, b) for (a, b) in ratio_pairs if a in df.columns and b in df.columns]

    logger.info(f"➗ Generando ratios NOMINALES antes de drift+drop: {len(ratio_pairs)} pares válidos")
    df = feature_engineering_ratios(df, ratio_pairs=ratio_pairs)

    ratio_cols = [f"{a}_over_{b}" for (a, b) in ratio_pairs]
    ratio_cols = [c for c in ratio_cols if c in df.columns]
    logger.info(f"✅ Ratios nominales generados: {len(ratio_cols)}")

    # ===========================
    # 04. Drift features SOLO pesos => genera *_rz_mes
    # ===========================
    if atributos_pesos:
        logger.info("💸 Generando drift-features monetarias: creando *_rz_mes")
        df = agregar_drift_features_monetarias(df, atributos_pesos)

        # ✅ Drop de montos nominales (pero NO afecta ratios ya creados)
        cols_drop = [c for c in atributos_pesos if c in df.columns]
        logger.info(f"🧹 Dropeando montos nominales en pesos: {len(cols_drop)} columnas (se conserva *_rz_mes y ratios)")
        df.drop(columns=cols_drop, inplace=True, errors="ignore")
    else:
        logger.info("ℹ️ No hay atributos_pesos presentes; no se crea *_rz_mes ni se dropean nominales.")

    logger.info(f"📐 Dataset luego de drift+drop nominales: shape={df.shape}")

    # ===========================
    # 05. FE TEMPORAL (lags/deltas/MA/shock) INCLUYE RATIOS
    # ===========================
    rz_cols = [f"{c}_rz_mes" for c in atributos_pesos if f"{c}_rz_mes" in df.columns]

    atributos_temporales = atributos_no_pesos + rz_cols + ratio_cols
    atributos_temporales = [c for c in dict.fromkeys(atributos_temporales) if c in df.columns]  # unique + existe

    cant_lag = 3
    window_size_ma = 3

    logger.info(
        f"🔧 FE temporal sobre {len(atributos_temporales)} columnas "
        f"(no_pesos={len(atributos_no_pesos)}, rz={len(rz_cols)}, ratios={len(ratio_cols)})"
    )

    df_fe = feature_engineering_tendencia(df, columnas=atributos_temporales, window_size=6)
    df_fe = feature_engineering_lag(df_fe, columnas=atributos_temporales, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributos_temporales, cant_lag=cant_lag)
    # df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos_temporales, window_size=window_size_ma)
    # df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos_temporales, window_size=window_size_ma)
    # df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos_temporales, window_size=window_size_ma)

    # # ===========================
    # # 06. FE NO TEMPORAL (cumsum / minmax / aguinaldo)
    # # ===========================
    # cumsum_cols = [
    #     "ctrx_quarter", "cpayroll_trx", "cpayroll2_trx",
    #     "chomebanking_transacciones", "ccallcenter_transacciones",
    #     "ctarjeta_debito_transacciones", "ctarjeta_visa_transacciones",
    #     "ctarjeta_master_transacciones", "ccajas_transacciones",
    # ]
    # cumsum_cols = [c for c in cumsum_cols if c in df_fe.columns]

    # minmax_cols_corr = [c for c in rz_cols if c in df_fe.columns]

    # WINDOW_STATS = 6
    # STRICT = False

    # logger.info("🔧 Iniciando feature engineering no-temporal (cumsum/minmax/aguinaldo)...")
    # df_fe = feature_engineering_cum_sum(df_fe, columnas=cumsum_cols, window_size=WINDOW_STATS, strict=STRICT)
    # df_fe = feature_engineering_min_max(df_fe, columnas=minmax_cols_corr, window_size=WINDOW_STATS, strict=STRICT)
    # df_fe = crear_indicador_aguinaldo(df_fe)

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

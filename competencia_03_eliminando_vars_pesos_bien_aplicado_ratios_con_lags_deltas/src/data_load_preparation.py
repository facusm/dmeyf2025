# src/data_load_preparation.py

import pandas as pd
import numpy as np
import duckdb

from config.config import (
    MESES_TRAIN_OPTUNA,
    MES_VAL_OPTUNA,
    COLS_ID,
    ELIMINAR_COLUMNAS_ID,
    APLICAR_UNDERSAMPLING,
    RATIO_UNDERSAMPLING,
)
from src.utils import logger, aplicar_undersampling


def cargar_datos(path: str) -> pd.DataFrame:
    """
    Carga datasets en formato .parquet o .csv(.gz) de forma eficiente usando DuckDB.
    Devuelve un DataFrame de pandas listo para usar.
    """
    logger.info(f"📥 Cargando dataset desde: {path}")

    try:
        con = duckdb.connect(database=":memory:")
        if path.endswith(".parquet"):
            query = f"SELECT * FROM read_parquet('{path}')"
        elif path.endswith(".csv.gz") or path.endswith(".csv"):
            query = f"SELECT * FROM read_csv_auto('{path}', header=True)"
        else:
            raise ValueError("❌ Formato no soportado (solo .parquet o .csv(.gz))")

        df = con.execute(query).fetchdf()

        logger.info(f"✅ Dataset cargado con {df.shape[0]:,} filas y {df.shape[1]:,} columnas")
        return df

    except Exception as e:
        logger.error(f"⚠️ Error al cargar el dataset: {e}")
        raise


def preparar_clases_y_pesos(data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas de clase y peso.
    """
    data = data.copy()

    data["clase_peso"] = 1.0
    data.loc[data["clase_ternaria"] == "BAJA+2", "clase_peso"] = 1.00002
    data.loc[data["clase_ternaria"] == "BAJA+1", "clase_peso"] = 1.00001

    data["clase_binaria1"] = np.where(data["clase_ternaria"] == "BAJA+2", 1, 0)
    data["clase_binaria2"] = np.where(data["clase_ternaria"] == "CONTINUA", 0, 1)

    logger.info("✅ Clases y pesos creados")
    logger.info(f"   clase_binaria1: {data['clase_binaria1'].value_counts().to_dict()}")
    logger.info(f"   clase_binaria2: {data['clase_binaria2'].value_counts().to_dict()}")

    return data

def preparar_train_meses(
    data: pd.DataFrame,
    meses: list[int],
    target: str = "clase_binaria2",
    apply_undersampling: bool | None = None,
    ratio: float | None = None,
    seed: int = 42,
    nombre_split: str = "train",
):
    """
    Prepara datos de entrenamiento a partir de una lista de meses.
    Aplica undersampling SOLO sobre esos meses, si está activado.

    Sirve tanto para:
    - Train de Optuna (MESES_TRAIN_OPTUNA)
    - Train inicial post-Optuna (p.ej. MESES_TRAIN_OPTUNA + 202104 + MES_VAL_OPTUNA)
    - Train final para test, etc.
    """
    apply_undersampling = (
        apply_undersampling
        if apply_undersampling is not None
        else APLICAR_UNDERSAMPLING
    )
    ratio = ratio or RATIO_UNDERSAMPLING

    train_data = data[data["foto_mes"].isin(meses)].copy()
    logger.info(
        f"📊 {nombre_split}: {len(train_data):,} registros de meses {sorted(set(meses))}"
    )

    if apply_undersampling and ratio < 1.0:
        logger.info(f"🔧 Undersampling activo en {nombre_split} con ratio={ratio:.2f}")
        train_data = aplicar_undersampling(
            train_data,
            target_col="clase_ternaria",
            rate=ratio,
            seed=seed,
        )
    else:
        logger.info(f"🔧 Sin undersampling en {nombre_split}")

    cols_to_drop = ["clase_ternaria", "clase_peso", "clase_binaria1", "clase_binaria2"]
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_train = train_data.drop(columns=cols_to_drop, errors="ignore")
    y_train = train_data[target]
    w_train = train_data["clase_peso"]

    logger.info(f"✅ {nombre_split}: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train, w_train


def preparar_validacion_meses(
    data: pd.DataFrame,
    meses: list[int],
    target: str = "clase_binaria2",
    nombre_split: str = "Validación",
):
    """
    Prepara datos de validación a partir de una lista de meses.
    SIN undersampling: refleja distribución real.

    Se puede usar para:
    - Validación interna (Optuna) -> MES_VAL_OPTUNA
    - Validación externa -> MES_VALID_EXT
    - O cualquier otro conjunto de meses que quieras evaluar.
    """
    valid_data = data[data["foto_mes"].isin(meses)].copy()

    logger.info(
        f"📊 {nombre_split}: {len(valid_data):,} registros de meses {sorted(set(meses))}"
    )

    cols_to_drop = ["clase_ternaria", "clase_peso", "clase_binaria1", "clase_binaria2"]
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_valid = valid_data.drop(columns=cols_to_drop, errors="ignore")
    y_valid = valid_data[target]
    w_valid = valid_data["clase_peso"]

    logger.info(f"✅ {nombre_split}: X={X_valid.shape}")
    return X_valid, y_valid, w_valid



def preparar_test_final_meses(
    data: pd.DataFrame,
    meses: list[int],
    nombre_split: str = "Test final",
):
    """
    Prepara datos de test (sin target) a partir de una lista de meses.

    - NO hace undersampling.
    - Devuelve:
        X_test: matriz de features
        numero_de_cliente: np.array con los IDs de cliente
    """
    test_data = data[data["foto_mes"].isin(meses)].copy()

    logger.info(
        f"📊 {nombre_split}: {len(test_data):,} registros de meses {sorted(set(meses))}"
    )

    cols_to_drop = ["clase_ternaria", "clase_peso", "clase_binaria1", "clase_binaria2"]
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    # Ojo: drop se hace sobre una copia, no toca test_data
    X_test = test_data.drop(columns=cols_to_drop, errors="ignore")
    numero_de_cliente = test_data["numero_de_cliente"].values

    logger.info(f"✅ {nombre_split}: X={X_test.shape}")
    return X_test, numero_de_cliente




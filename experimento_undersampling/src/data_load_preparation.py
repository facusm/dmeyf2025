# src/data_load_preparation.py

import pandas as pd
import numpy as np
import duckdb
import os
from config.config import (
    MESES_TRAIN,
    MES_VAL_OPTUNA,
    MES_VALID,
    MES_TEST_FINAL,
    COLS_ID,
    ELIMINAR_COLUMNAS_ID,
    APLICAR_UNDERSAMPLING,
    RATIO_UNDERSAMPLING
)
from src.utils import logger, aplicar_undersampling


def cargar_datos(path: str) -> pd.DataFrame:
    """
    Carga datasets en formato .parquet o .csv(.gz) de forma eficiente usando DuckDB.
    Devuelve un DataFrame de pandas listo para usar.
    """
    logger.info(f"📥 Cargando dataset desde {path}")

    try:
        con = duckdb.connect(database=':memory:')
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


def preparar_clases_y_pesos(data):
    """
    Crea columnas de clase y peso.
    """
    data = data.copy()

    data['clase_peso'] = 1.0
    data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
    data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)

    logger.info("✅ Clases y pesos creados")
    logger.info(f"   clase_binaria1: {data['clase_binaria1'].value_counts().to_dict()}")
    logger.info(f"   clase_binaria2: {data['clase_binaria2'].value_counts().to_dict()}")

    return data


def preparar_train_optuna(data, target='clase_binaria2', apply_undersampling=None, ratio=None, seed=42):
    """
    Prepara datos de entrenamiento para Optuna usando MESES_TRAIN.
    Aplica undersampling SOLO sobre MESES_TRAIN (train puro).
    """
    apply_undersampling = apply_undersampling if apply_undersampling is not None else APLICAR_UNDERSAMPLING
    ratio = ratio or RATIO_UNDERSAMPLING

    train_data = data[data['foto_mes'].isin(MESES_TRAIN)].copy()
    logger.info(f"📊 Train Optuna: {len(train_data)} registros de meses {MESES_TRAIN}")

    if apply_undersampling:
        logger.info(f"🔧 Aplicando undersampling con ratio={ratio}")
        train_data = aplicar_undersampling(
            train_data,
            target_col='clase_ternaria',
            rate=ratio,
            seed=seed
        )

    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_train = train_data.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_data[target]
    w_train = train_data['clase_peso']

    logger.info(f"✅ X_train_optuna: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train, w_train


def preparar_validacion_optuna(data, target='clase_binaria2'):
    """
    Prepara datos de validación interna para Optuna (MES_VAL_OPTUNA).
    SIN undersampling: refleja distribución real.
    """
    valid_data = data[data['foto_mes'].isin(MES_VAL_OPTUNA)].copy()
    logger.info(f"📊 Validación Optuna: {len(valid_data)} registros del mes {MES_VAL_OPTUNA}")

    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_valid = valid_data.drop(columns=cols_to_drop, errors='ignore')
    y_valid = valid_data[target]
    w_valid = valid_data['clase_peso']

    logger.info(f"✅ X_valid_optuna: {X_valid.shape}")
    return X_valid, y_valid, w_valid


def preparar_validacion(data, target='clase_binaria2'):
    """
    Prepara datos de validación externa (MES_VALID).
    Se usa para ajustar umbral / evaluar generalización.
    """
    valid_data = data[data['foto_mes'].isin(MES_VALID)].copy()
    logger.info(f"📊 Validación externa: {len(valid_data)} registros del mes {MES_VALID}")

    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_valid = valid_data.drop(columns=cols_to_drop, errors='ignore')
    y_valid = valid_data[target]
    w_valid = valid_data['clase_peso']

    logger.info(f"✅ X_valid_ext: {X_valid.shape}")
    return X_valid, y_valid, w_valid


def preparar_test_final(data):
    """
    Prepara datos de test final (MES_TEST_FINAL).
    """
    test_data = data[data['foto_mes'].isin(MES_TEST_FINAL)].copy()
    logger.info(f"📊 Test final: {len(test_data)} registros de meses {MES_TEST_FINAL}")

    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID

    X_test = test_data.drop(columns=cols_to_drop, errors='ignore')
    numero_de_cliente = test_data['numero_de_cliente'].values

    logger.info(f"✅ X_test: {X_test.shape}")
    return X_test, numero_de_cliente


def preparar_train_completo(train_optuna, valid_optuna=None, valid_externa=None):
    """
    Arma un train completo concatenando:
    - train_optuna (ya undersampleado, MESES_TRAIN)
    - valid_optuna (MES_VAL_OPTUNA, sin undersampling)
    - valid_externa (MES_VALID, sin undersampling)
    """
    X_parts = [train_optuna[0]]
    y_parts = [train_optuna[1]]
    w_parts = [train_optuna[2]]

    if valid_optuna is not None:
        X_parts.append(valid_optuna[0])
        y_parts.append(valid_optuna[1])
        w_parts.append(valid_optuna[2])

    if valid_externa is not None:
        X_parts.append(valid_externa[0])
        y_parts.append(valid_externa[1])
        w_parts.append(valid_externa[2])

    X_train_completo = pd.concat(X_parts, ignore_index=True)
    y_train_completo = pd.concat(y_parts, ignore_index=True)
    w_train_completo = pd.concat(w_parts, ignore_index=True)

    logger.info(f"✅ X_train_completo: {X_train_completo.shape}")
    return X_train_completo, y_train_completo, w_train_completo

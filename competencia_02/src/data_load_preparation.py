# src/data_preparation.py
import pandas as pd
import numpy as np
import duckdb
import os
from config.config import (
    MESES_TRAIN, MES_VALID, MES_TEST_FINAL,
    COLS_ID, ELIMINAR_COLUMNAS_ID, APLICAR_UNDERSAMPLING, RATIO_UNDERSAMPLING
)
from src.utils import logger, aplicar_undersampling


## Funcion para cargar datos
def cargar_datos(path: str) -> pd.DataFrame:
    """
    Carga datasets en formato .parquet o .csv(.gz) de forma eficiente usando DuckDB.
    Devuelve un DataFrame de pandas listo para usar.
    """
    logger.info(f"📥 Cargando dataset desde {path}")

    try:
        con = duckdb.connect(database=':memory:')
        # Detectar formato automáticamente
        if path.endswith(".parquet"):
            query = f"SELECT * FROM read_parquet('{path}')"
        elif path.endswith(".csv.gz") or path.endswith(".csv"):
            query = f"SELECT * FROM read_csv_auto('{path}', header=True)"
        else:
            raise ValueError("❌ Formato de archivo no soportado (solo .parquet o .csv(.gz))")

        # Leer y convertir a pandas
        df = con.execute(query).fetchdf()

        logger.info(f"✅ Dataset cargado con {df.shape[0]:,} filas y {df.shape[1]:,} columnas")
        return df

    except Exception as e:
        logger.error(f"⚠️ Error al cargar el dataset: {e}")
        raise


def preparar_clases_y_pesos(data):
    """
    Crea las columnas de clase y peso.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset original.
    
    Returns
    -------
    pd.DataFrame
        Dataset con columnas adicionales.
    """
    data = data.copy()
    
    # Crear pesos
    data['clase_peso'] = 1.0
    data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001
    
    # Crear targets binarios
    data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
    data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)
    
    logger.info("✅ Clases y pesos creados")
    logger.info(f"   clase_binaria1: {data['clase_binaria1'].value_counts().to_dict()}")
    logger.info(f"   clase_binaria2: {data['clase_binaria2'].value_counts().to_dict()}")
    
    return data


def preparar_train_optuna(data, target='clase_binaria2', apply_undersampling=None, ratio=None, seed=42):
    """
    Prepara datos de entrenamiento para Optuna (meses de train).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset completo.
    target : str
        Columna target a usar ('clase_binaria1' o 'clase_binaria2').
    apply_undersampling : bool, optional
        Si aplicar undersampling. Si no se provee, usa config.
    ratio : float, optional
        Ratio de undersampling. Si no se provee, usa config.
    seed : int
        Semilla para undersampling.
    
    Returns
    -------
    tuple
        (X_train, y_train, w_train)
    """
    apply_undersampling = apply_undersampling if apply_undersampling is not None else APLICAR_UNDERSAMPLING
    ratio = ratio or RATIO_UNDERSAMPLING
    
    train_data = data[data['foto_mes'].isin(MESES_TRAIN)].copy()
    logger.info(f"📊 Datos de entrenamiento: {len(train_data)} registros de meses {MESES_TRAIN}")
    
    # Aplicar undersampling si está habilitado
    if apply_undersampling:
        logger.info(f"🔧 Aplicando undersampling con ratio={ratio}")
        train_data = aplicar_undersampling(train_data, target_col='clase_ternaria', rate=ratio, seed=seed)
    
    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID
    
    X_train = train_data.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_data[target]
    w_train = train_data['clase_peso']
    
    logger.info(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"🔧 Columnas eliminadas: {cols_to_drop}")
    
    return X_train, y_train, w_train


def preparar_validacion(data, target='clase_binaria2'):
    """
    Prepara datos de validación (abril).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset completo.
    target : str
        Columna target a usar.
    
    Returns
    -------
    tuple
        (X_valid, y_valid, w_valid)
    """
    valid_data = data[data['foto_mes'].isin(MES_VALID)].copy() # Si hay mas de un mes en MES_VALID
    logger.info(f"📊 Datos de validación: {len(valid_data)} registros del mes {MES_VALID}")
    
    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID
    
    X_valid = valid_data.drop(columns=cols_to_drop, errors='ignore')
    y_valid = valid_data[target] if target in valid_data.columns else None
    w_valid = valid_data['clase_peso']
    
    logger.info(f"✅ X_valid: {X_valid.shape}")
    
    return X_valid, y_valid, w_valid


def preparar_test_final(data):
    """
    Prepara datos de test final (junio).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset completo.
    
    Returns
    -------
    tuple
        (X_test, numero_de_cliente)
    """
    test_data = data[data['foto_mes'].isin(MES_TEST_FINAL)].copy() # Si hay mas de un mes en MES_TEST_FINAL
    logger.info(f"📊 Datos de test: {len(test_data)} registros del mes {MES_TEST_FINAL}")
    
    cols_to_drop = ['clase_ternaria', 'clase_peso', 'clase_binaria1', 'clase_binaria2']
    if ELIMINAR_COLUMNAS_ID:
        cols_to_drop += COLS_ID
    
    X_test = test_data.drop(columns=cols_to_drop, errors='ignore')
    numero_de_cliente = test_data['numero_de_cliente'].values
    
    logger.info(f"✅ X_test: {X_test.shape}")
    
    return X_test, numero_de_cliente


def preparar_train_completo(train_optuna, X_valid, y_valid, w_valid):
    """
    Prepara datos de entrenamiento completo (train + validación)
    a partir del set de train de Optuna ya undersampleado.
    """
    # Concatenar features
    X_train_completo = pd.concat([train_optuna[0], X_valid], ignore_index=True)
    y_train_completo = pd.concat([train_optuna[1], y_valid], ignore_index=True)
    w_train_completo = pd.concat([train_optuna[2], w_valid], ignore_index=True)
    
    logger.info(f"✅ X_train_completo: {X_train_completo.shape}")
    return X_train_completo, y_train_completo, w_train_completo


    """
EN main.py

# --- Preparar train de Optuna ---
X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_optuna(data)

# --- Preparar validación ---
X_valid, y_valid, w_valid = preparar_validacion(data)

# --- Concatenar para train completo ---
X_train_completo, y_train_completo, w_train_completo = preparar_train_completo(
    train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
    X_valid=X_valid,
    y_valid=y_valid,
    w_valid=w_valid
)
    """
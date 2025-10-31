# src/optuna_optimization.py
import lightgbm as lgb
import optuna
import numpy as np
import os
from config.config import (
    PARAMS, NOMBRE_DE_ESTUDIO_OPTUNA, NOMBRE_BASE_DE_DATOS_OPTUNA,
    DB_PATH, N_TRIALS, N_STARTUP_TRIALS, GANANCIA_ACIERTO, COSTO_ESTIMULO
)
from src.utils import logger


def lgb_gan_eval(y_pred, data):
    """
    Función de evaluación personalizada para LightGBM.
    
    Calcula ganancia acumulada usando los pesos para identificar BAJA+2.
    """
    weight = data.get_weight()
    ganancia = np.where(
        weight == 1.00002,
        GANANCIA_ACIERTO,
        0
    ) - np.where(
        weight < 1.00002,
        abs(COSTO_ESTIMULO),
        0
    )
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return 'gan_eval', np.max(ganancia), True


def crear_estudio_optuna(seed, load_if_exists=True):
    """
    Crea o carga un estudio de Optuna.
    
    Parameters
    ----------
    seed : int
        Semilla para reproducibilidad.
    load_if_exists : bool
        Si True, carga el estudio existente. Si False, lo sobrescribe.
    
    Returns
    -------
    optuna.Study
        Objeto de estudio de Optuna.
    """
    storage_name = f"sqlite:///{os.path.join(DB_PATH, NOMBRE_BASE_DE_DATOS_OPTUNA)}"
    
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=seed
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name=NOMBRE_DE_ESTUDIO_OPTUNA,
        storage=storage_name,
        load_if_exists=load_if_exists,
        sampler=sampler
    )
    
    logger.info(f"📊 Estudio de Optuna {'cargado' if load_if_exists else 'creado'}: {NOMBRE_DE_ESTUDIO_OPTUNA}")
    logger.info(f"💾 Base de datos: {storage_name}")
    
    return study


def objective(trial, X_train, y_train, w_train, seed):
    """
    Función objetivo para la optimización bayesiana con Optuna.
    
    Parameters
    ----------
    trial : optuna.Trial
        Objeto trial de Optuna.
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    w_train : pd.Series
        Pesos de entrenamiento.
    seed : int
        Semilla para reproducibilidad.
    
    Returns
    -------
    float
        Ganancia promedio multiplicada por número de folds.
    """
    # Espacio de búsqueda de hiperparámetros
    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'num_leaves': trial.suggest_int('num_leaves', 8, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 30),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 5.0),
        'max_depth': trial.suggest_int('max_depth', 3, 40),
        'seed': seed,
        'verbose': -1
    }

    num_boost_round = trial.suggest_int('num_boost_round', 100, 2000)

    train_dataset = lgb.Dataset(X_train, label=y_train, weight=w_train)
    nfold = PARAMS.get('n_folds', 5)

    cv_results = lgb.cv(
        params,
        train_dataset,
        num_boost_round=num_boost_round,
        feval=lgb_gan_eval,
        stratified=True,
        nfold=nfold,
        seed=seed,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(50 + 5 / params['learning_rate']),
                verbose=False
            ),
            lgb.log_evaluation(period=200),
        ]
    )

    max_gan = max(cv_results['valid gan_eval-mean'])
    best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1
    trial.set_user_attr("best_iter", best_iter)

    return max_gan * nfold


def ejecutar_optimizacion(X_train, y_train, w_train, seed, n_trials=None):
    """
    Ejecuta la optimización de hiperparámetros.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    w_train : pd.Series
        Pesos de entrenamiento.
    seed : int
        Semilla para reproducibilidad.
    n_trials : int, optional
        Número de trials a ejecutar. Si no se provee, usa N_TRIALS de config.
    
    Returns
    -------
    optuna.Study
        Estudio completado.
    """
    study = crear_estudio_optuna(seed, load_if_exists=True)
    
    n_trials = n_trials or N_TRIALS
    logger.info(f"🚀 Iniciando optimización con {n_trials} trials...")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, w_train, seed),
        n_trials=n_trials
    )
    
    logger.info(f"✅ Optimización completada. Mejor ganancia: {study.best_value:.2f}")
    
    return study
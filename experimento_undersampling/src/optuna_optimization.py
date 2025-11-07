# src/optuna_optimization.py

import os
import numpy as np
import lightgbm as lgb
import optuna

from config.config import (
    PARAMS,
    NOMBRE_DE_ESTUDIO_OPTUNA,
    NOMBRE_BASE_DE_DATOS_OPTUNA,
    N_TRIALS,
    N_STARTUP_TRIALS,
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
    DB_PATH,
    NOMBRE_EXPERIMENTO,
)
from src.utils import logger, mejor_umbral_probabilidad


def lgb_gan_eval(y_pred, data):
    """
    Métrica personalizada de ganancia para LightGBM.
    Usa los pesos para identificar BAJA+2 y el costo del estímulo.
    """
    weight = data.get_weight()

    # Ganancia por registro ordenado por probabilidad descendente
    ganancia = (
        np.where(weight == 1.00002, GANANCIA_ACIERTO, 0)
        - np.where(weight < 1.00002, abs(COSTO_ESTIMULO), 0)
    )
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return "gan_eval", float(np.max(ganancia)), True


def crear_estudio_optuna(seed, load_if_exists=True):
    """
    Crea o carga un estudio de Optuna con storage SQLite,
    guardado en la carpeta específica del experimento actual.
    """
    os.makedirs(DB_PATH, exist_ok=True)

    storage_path = os.path.join(DB_PATH, NOMBRE_BASE_DE_DATOS_OPTUNA)
    storage_name = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        seed=seed,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=NOMBRE_DE_ESTUDIO_OPTUNA,
        storage=storage_name,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )

    logger.info(
        f"📊 Estudio Optuna "
        f"{'cargado' if load_if_exists else 'creado'}: {NOMBRE_DE_ESTUDIO_OPTUNA}"
    )
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"💾 Storage Optuna: {storage_name}")

    return study


def objective(trial, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros):
    """
    Función objetivo sin CV explícito:
    - Entrena en MESES_TRAIN (potencialmente undersampleado).
    - Valida en MES_VAL_OPTUNA (sin undersampling).
    - Promedia la ganancia sobre múltiples semillas.
    """
    params = {
        "objective": "binary",
        "metric": "custom",
        "boosting_type": "gbdt",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "max_bin": 31,
        "num_leaves": trial.suggest_int("num_leaves", 8, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 30),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        "max_depth": trial.suggest_int("max_depth", 3, 40),
        "verbose": -1,
    }

    num_boost_round = trial.suggest_int("num_boost_round", 100, 2000)

    ganancias_semillas = []
    best_iters = []

    for seed in semilleros:
        params["seed"] = seed

        dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            feval=lgb_gan_eval,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(50 + 5 / params["learning_rate"]),
                    verbose=False,
                )
            ],
        )

        best_iters.append(model.best_iteration)

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        _, _, ganancia, _ = mejor_umbral_probabilidad(y_pred, w_valid)
        ganancias_semillas.append(ganancia)

    gan_prom = float(np.mean(ganancias_semillas))
    best_iter_prom = int(np.mean(best_iters))

    trial.set_user_attr("ganancias_semillas", ganancias_semillas)
    trial.set_user_attr("best_iter", best_iter_prom)

    return gan_prom


def ejecutar_optimizacion(
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    semilleros,
    seed,
    n_trials=None,
):
    """
    Ejecuta Optuna con validación temporal (MES_VAL_OPTUNA) y multisemilla.
    Usa el estudio y storage definidos por el experimento actual.
    """
    study = crear_estudio_optuna(seed, load_if_exists=True)
    n_trials = n_trials or N_TRIALS

    completados = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    logger.info(f"📊 Trials completados: {completados}/{n_trials}")

    restantes = n_trials - completados
    if restantes <= 0:
        logger.info("✅ Estudio ya alcanzó el número de trials requerido.")
        logger.info(f"🏅 Mejor ganancia promedio: {study.best_value:,.0f}")
        logger.info(f"🏅 Mejores parámetros: {study.best_params}")
        logger.info(
            f"🔁 Mejor iter (promedio): {study.best_trial.user_attrs.get('best_iter')}"
        )
        return study

    logger.info(f"🚀 Ejecutando {restantes} trials adicionales...")

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            w_train,
            X_valid,
            y_valid,
            w_valid,
            semilleros,
        ),
        n_trials=restantes,
    )

    logger.info(f"✅ Mejor ganancia promedio: {study.best_value:,.0f}")
    logger.info(f"🏅 Mejores parámetros: {study.best_params}")
    logger.info(
        f"🔁 Mejor iter (promedio): {study.best_trial.user_attrs.get('best_iter')}"
    )

    return study

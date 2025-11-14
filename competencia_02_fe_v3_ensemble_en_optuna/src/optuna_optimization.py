# src/optuna_optimization.py

import os
import numpy as np
import lightgbm as lgb
import optuna

from config.config import (
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


# ==============================================================================
# 📌 MÉTRICA CUSTOM DE GANANCIA
# ==============================================================================
def lgb_gan_eval(y_pred, data):
    """
    Métrica personalizada de ganancia para LightGBM.
    Usa los pesos para identificar BAJA+2 y el costo del estímulo.
    """
    weight = data.get_weight()

    ganancia = (
        np.where(weight == 1.00002, GANANCIA_ACIERTO, 0)
        - np.where(weight < 1.00002, abs(COSTO_ESTIMULO), 0)
    )

    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    return "gan_eval", float(np.max(ganancia)), True


# ==============================================================================
# 📌 CREAR O CARGAR ESTUDIO OPTUNA
# ==============================================================================
def crear_estudio_optuna(seed, load_if_exists=True):

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

    logger.info(f"📊 Estudio Optuna: {NOMBRE_DE_ESTUDIO_OPTUNA}")
    logger.info(f"🏷️ Storage: {storage_name}")
    return study


# ==============================================================================
# 📌 HP dinámicos según tamaño del dataset (igual que tu versión original)
# ==============================================================================
def get_boosting_schedule(n_train: int):
    """
    Define num_boost_round y early_stopping_rounds según tamaño del train.
    """
    if n_train >= 4_000_000:
        return 4000, 200
    elif n_train >= 1_000_000:
        return 3000, 150
    elif n_train >= 300_000:
        return 2000, 100
    else:
        return 1000, 80


def suggest_params_dynamic(trial, n_train: int):
    """
    Search space dinámico según tamaño del dataset.
    COMPLETAMENTE compatible con lo que venías haciendo.
    """

    if n_train >= 4_000_000:
        nl_lo, nl_hi = 128, 512
        ml_lo, ml_hi = 400, 4000
        l2_hi = 50.0

    elif n_train >= 1_000_000:
        nl_lo, nl_hi = 64, 384
        ml_lo, ml_hi = 200, 3000
        l2_hi = 30.0

    elif n_train >= 300_000:
        nl_lo, nl_hi = 32, 256
        ml_lo, ml_hi = 50, 1500
        l2_hi = 20.0

    else:
        nl_lo, nl_hi = 16, 128
        ml_lo, ml_hi = 20, 500
        l2_hi = 10.0

    ml_lo = max(10, ml_lo)
    ml_hi = max(ml_lo + 10, ml_hi)

    params = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,

        "num_leaves": trial.suggest_int("num_leaves", nl_lo, nl_hi),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", ml_lo, ml_hi),
        "max_depth": -1,

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),

        "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127, 255]),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),

        "lambda_l1": 0.0,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, l2_hi, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),

        "num_threads": -1,
        "verbose": -1,
    }

    return params


# ==============================================================================
# 📌 FUNCIÓN OBJETIVO (ENSEMBLE DENTRO DEL TRIAL)
# ==============================================================================
def objective(trial, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros):

    n_train = X_train.shape[0]

    # HP dinámicos
    params = suggest_params_dynamic(trial, n_train)
    num_boost_round, early_stopping = get_boosting_schedule(n_train)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)

    sum_preds_valid = None
    ganancias_seeds = []
    best_iters = []

    for i, seed in enumerate(semilleros):
        params["seed"] = int(seed)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            feval=lgb_gan_eval,
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iters.append(int(model.best_iteration))

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        if sum_preds_valid is None:
            sum_preds_valid = y_pred
        else:
            sum_preds_valid += y_pred

        _, _, gan_seed, _ = mejor_umbral_probabilidad(y_pred, w_valid)
        ganancias_seeds.append(float(gan_seed))

    # Ensemble final en validación
    y_pred_ensemble = sum_preds_valid / len(semilleros)

    umbral_ens, N_opt, gan_ens, _ = mejor_umbral_probabilidad(
        y_pred_ensemble, w_valid
    )

    trial.set_user_attr("ganancias_semillas", ganancias_seeds)
    trial.set_user_attr("ganancia_ensemble", float(gan_ens))
    trial.set_user_attr("best_iter", int(np.mean(best_iters)))
    trial.set_user_attr("umbral_ensemble", float(umbral_ens))
    trial.set_user_attr("N_opt_ensemble", int(N_opt))

    logger.info(
        f"🧪 Trial {trial.number} → Ganancia ensemble = ${gan_ens:,.0f}"
    )

    return float(gan_ens)


# ==============================================================================
# 📌 FUNCIÓN DE EJECUCIÓN DE OPTUNA
# ==============================================================================
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

    study = crear_estudio_optuna(seed)
    n_trials = n_trials or N_TRIALS

    completados = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    restantes = n_trials - completados

    logger.info(f"📊 Trials completados: {completados}/{n_trials}")

    if restantes > 0:
        logger.info(f"🚀 Ejecutando {restantes} trials…")
        study.optimize(
            lambda t: objective(
                t, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros
            ),
            n_trials=restantes,
        )
    else:
        logger.info("⚠️ El estudio ya tiene todos los trials completos.")

    best_trial = study.best_trial

    logger.info("\n==============================")
    logger.info("🏆 OPTIMIZACIÓN COMPLETADA")
    logger.info("==============================")
    logger.info(f"⭐ Mejor trial = #{best_trial.number}")
    logger.info(f"💰 Ganancia ensemble = ${study.best_value:,.0f}")
    logger.info(f"🧩 Best params = {best_trial.params}")
    logger.info(f"🔁 Best iter promedio = {best_trial.user_attrs['best_iter']}")

    return study

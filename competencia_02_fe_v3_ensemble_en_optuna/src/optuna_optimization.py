#src/optuna_optimization.py

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


def get_boosting_schedule(n_train: int) -> tuple[int, int]:
    """
    Define num_boost_round y early_stopping_rounds según el tamaño efectivo del train.
    Mantiene suficiente techo para learning rates chicos.
    Ajustá cortes si hace falta, pero la lógica es esta.
    """
    if n_train >= 4_000_000:
        max_rounds = 4000
        es_rounds = 200
    elif n_train >= 1_000_000:
        max_rounds = 3000
        es_rounds = 150
    elif n_train >= 300_000:
        max_rounds = 2000
        es_rounds = 100
    else:
        max_rounds = 1000
        es_rounds = 80

    return max_rounds, es_rounds


def suggest_params_dynamic(trial, n_train: int) -> dict:
    """
    Rangos de hiperparámetros ajustados al tamaño del dataset (post-undersampling).

    Idea:
    - Dataset grande -> más capacidad pero min_data_in_leaf alto.
    - Dataset chico (por undersampling) -> menos leaves y min_data_in_leaf más chico.
    """

    if n_train >= 4_000_000:          # muy grande
        nl_lo, nl_hi = 128, 512
        ml_lo, ml_hi = 400, 4000
        l2_hi = 50.0
    elif n_train >= 1_000_000:       # grande
        nl_lo, nl_hi = 64, 384
        ml_lo, ml_hi = 200, 3000
        l2_hi = 30.0
    elif n_train >= 300_000:         # medio
        nl_lo, nl_hi = 32, 256
        ml_lo, ml_hi = 50, 1500
        l2_hi = 20.0
    else:                            # chico (undersampling fuerte)
        nl_lo, nl_hi = 16, 128
        ml_lo, ml_hi = 20, 500
        l2_hi = 10.0

    # sanity
    ml_lo = max(10, ml_lo)
    ml_hi = max(ml_lo + 10, ml_hi)

    params = {
        "objective": "binary",
        "metric": "None",              # usamos solo lgb_gan_eval (custom)
        "boosting_type": "gbdt",
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,

        # Complejidad del árbol escalada a n_train
        "num_leaves": trial.suggest_int("num_leaves", nl_lo, nl_hi),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", ml_lo, ml_hi),
        "max_depth": -1,

        # LR tunable (clave, no fijo)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),

        # Resolución de bins:
        # Incluimos 31 (receta cátedra) pero dejamos que explore más fino.
        "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127, 255]),

        # Subsampling:
        # Si ya aplicaste undersampling fuerte, Optuna tenderá a valores altos.
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),  # 0 = sin bagging permitido

        # Regularización: simple y robusta
        "lambda_l1": 0.0,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, l2_hi, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),

        "num_threads": -1,
        "verbose": -1,
    }

    return params


def objective(trial, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros):
    """
    Función objetivo sin CV explícito.

    - Entrena en MESES_TRAIN (potencialmente undersampleado).
    - Valida en MESES_VAL_OPTUNA (sin undersampling, distribución real).
    - Entrena un modelo por semilla.
    - Promedia las predicciones de todas las semillas (ensemble) en validación.
    - Calcula la ganancia óptima del ENSEMBLE en validación.
    - Usa ESA ganancia del ensemble como métrica objetivo del trial.
    - Ajusta rangos de HP según len(X_train).
    """

    n_train = X_train.shape[0]

    # HP dinámicos según tamaño del dataset
    params = suggest_params_dynamic(trial, n_train)

    # num_boost_round y early_stopping también en función de n_train
    num_boost_round, early_stopping_rounds = get_boosting_schedule(n_train)

    # Datasets (una sola vez)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)

    # Para ensemble y trazabilidad
    sum_preds_valid = None
    ganancias_semillas = []
    best_iters = []

    # ========= Loop por seeds (ensemble) =========
    for i, seed in enumerate(semilleros):
        params["seed"] = int(seed)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            feval=lgb_gan_eval,  # tu métrica custom de ganancia
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        best_iters.append(int(model.best_iteration))

        # Predicciones en valid para esta seed
        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

        # Acumular para ensemble
        if sum_preds_valid is None:
            sum_preds_valid = y_pred
        else:
            sum_preds_valid += y_pred

        # Ganancia individual (tracking)
        _, _, gan_seed, _ = mejor_umbral_probabilidad(y_pred, w_valid)
        ganancias_semillas.append(float(gan_seed))

        # ---------- Logging informativo entre seeds ----------
        y_pred_ens_parcial = sum_preds_valid / (i + 1)
        _, _, gan_ens_parcial, _ = mejor_umbral_probabilidad(
            y_pred_ens_parcial,
            w_valid,
        )

        logger.info(
            f"🧩 Trial {trial.number} | Seed {i+1}/{len(semilleros)} ({seed}) | "
            f"Ganancia parcial ensemble: ${gan_ens_parcial:,.0f}"
        )

    # ========= Ensemble final (todas las seeds) =========
    y_pred_ensemble = sum_preds_valid / len(semilleros)

    umbral_ens, N_opt_ens, ganancia_ens, _ = mejor_umbral_probabilidad(
        y_pred_ensemble,
        w_valid,
    )

    best_iter_prom = int(np.mean(best_iters))

    # ========= Metadata del trial =========
    trial.set_user_attr("ganancias_semillas", ganancias_semillas)
    trial.set_user_attr("ganancia_ensemble", float(ganancia_ens))
    trial.set_user_attr("best_iter", best_iter_prom)
    trial.set_user_attr("umbral_ensemble", float(umbral_ens))
    trial.set_user_attr("N_opt_ensemble", int(N_opt_ens))
    trial.set_user_attr("n_train", int(n_train))

    logger.info(
        f"✅ Trial {trial.number} COMPLETADO | Ganancia ensemble final: ${ganancia_ens:,.0f}"
    )

    # Objetivo: ganancia del ensemble en validación
    return float(ganancia_ens)



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
    Usa como métrica objetivo la ganancia del ENSEMBLE de semillas en el set de validación.
    """
    study = crear_estudio_optuna(seed, load_if_exists=True)
    n_trials = n_trials or N_TRIALS

    # Trials ya completos
    completados = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    logger.info(f"📊 Trials completados: {completados}/{n_trials}")

    restantes = n_trials - completados
    if restantes > 0:
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
    else:
        logger.info("✅ Estudio ya alcanzó el número de trials requerido.")

    # =========================
    # 🔍 RESUMEN DEL MEJOR TRIAL
    # =========================
    best_trial = study.best_trial
    best_value = float(study.best_value)  # debería ser igual a ganancia_ensemble

    ganancia_ensemble = float(
        best_trial.user_attrs.get("ganancia_ensemble", best_value)
    )
    best_iter = (
        int(best_trial.user_attrs["best_iter"])
        if "best_iter" in best_trial.user_attrs
        else None
    )
    umbral_ensemble = (
        float(best_trial.user_attrs["umbral_ensemble"])
        if "umbral_ensemble" in best_trial.user_attrs
        else None
    )
    N_opt_ensemble = (
        int(best_trial.user_attrs["N_opt_ensemble"])
        if "N_opt_ensemble" in best_trial.user_attrs
        else None
    )
    ganancias_semillas = best_trial.user_attrs.get("ganancias_semillas", None)

    logger.info("✅ OPTIMIZACIÓN COMPLETADA")
    logger.info(f"🏅 Mejor trial: #{best_trial.number}")
    logger.info(f"💰 Mejor ganancia (objective / ensemble): ${best_value:,.0f}")

    if umbral_ensemble is not None and N_opt_ensemble is not None:
        logger.info(
            f"🎯 Umbral ensemble (valid): {umbral_ensemble:.6f} | "
            f"N óptimo ensemble: {N_opt_ensemble:,}"
        )

    if best_iter is not None:
        logger.info(f"🔁 Iteraciones promedio (best_iter): {best_iter}")

    logger.info(f"⚙️ Mejores hiperparámetros: {best_trial.params}")

    if ganancias_semillas is not None:
        try:
            gs = [float(g) for g in ganancias_semillas]
            logger.info(
                "📈 Ganancias individuales por semilla (valid): "
                + ", ".join(f"{g:,.0f}" for g in gs)
            )
        except Exception:
            logger.info("📈 Ganancias por semilla disponibles en user_attrs.")

    return study

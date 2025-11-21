#src/optuna_optimization.py

import os
import numpy as np
import lightgbm as lgb
import optuna
from src.utils import logger


from config.config import (
    NOMBRE_DE_ESTUDIO_OPTUNA,
    NOMBRE_BASE_DE_DATOS_OPTUNA,
    N_TRIALS,
    N_STARTUP_TRIALS,
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
    DB_PATH,
    NOMBRE_EXPERIMENTO,
    N_REPE_OPTUNA,
    LGBM_PARAMS_BASE
)


def calcular_umbral_y_ganancia_meseta(
    y_pred: np.ndarray,
    weight: np.ndarray,
    window: int = 2001,
):
    """
    Calcula:
    - gan_acum = cumsum(gan) ordenada por probabilidad descendente
    - gan_meseta = promedio móvil centrado de gan_acum (ventana 'window')
    - Devuelve:
        umbral_opt (probabilidad en el punto de máxima meseta),
        N_opt (cantidad de clientes hasta ese punto),
        gan_max_meseta (valor máximo de la meseta).
    Es el análogo en Python de la lógica de 'meseta' de la cátedra.
    """

    y_pred = np.asarray(y_pred)
    weight = np.asarray(weight)

    n = y_pred.shape[0]
    if n == 0:
        return 0.0, 0, 0.0

    # Misma definición de "gan" que tu lgb_gan_eval actual
    gan = (
        np.where(weight == 1.00002, GANANCIA_ACIERTO, 0)
        - np.where(weight < 1.00002, abs(COSTO_ESTIMULO), 0)
    )

    # Ordenar por probabilidad descendente
    orden = np.argsort(y_pred)[::-1]
    gan_ordenada = gan[orden]
    y_pred_ordenada = y_pred[orden]

    # Ganancia acumulada (gan_acum)
    gan_acum = np.cumsum(gan_ordenada)

    # Ventana de la meseta (tipo frollmean align="center")
    win = min(window, n)
    if win <= 1:
        gan_meseta = gan_acum.copy().astype(float)
    else:
        kernel = np.ones(win, dtype=float) / float(win)
        # 'same' ≈ promedio centrado (bordes con ventanas parciales)
        gan_meseta = np.convolve(gan_acum, kernel, mode="same")

    # Índice del máximo de la meseta
    idx_max = int(np.nanargmax(gan_meseta))
    gan_max_meseta = float(gan_meseta[idx_max])
    N_opt = idx_max + 1  # cantidad de clientes hasta ese punto
    umbral_opt = float(y_pred_ordenada[idx_max])

    return umbral_opt, N_opt, gan_max_meseta


def lgb_gan_eval(y_pred, data):
    """
    Métrica personalizada de ganancia para LightGBM basada en MESETA.

    - Usa los pesos para identificar BAJA+2 y el costo del estímulo.
    - Ordena por probabilidad descendente.
    - Calcula gan_acum, luego una "meseta" (promedio móvil) y toma su máximo.
    - Eso es lo que optimizan LightGBM y Optuna.
    """
    weight = data.get_weight()

    _, _, gan_max_meseta = calcular_umbral_y_ganancia_meseta(
        y_pred=y_pred,
        weight=weight,
        window=2001,  # podés parametrizarlo si querés
    )

    # nombre de la métrica (solo texto), valor, higher_is_better=True
    return "gan_meseta", float(gan_max_meseta), True



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

    params = LGBM_PARAMS_BASE.copy()
    params.update({
        "num_leaves": trial.suggest_int("num_leaves", nl_lo, nl_hi),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", ml_lo, ml_hi),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127, 255]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, l2_hi, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    })
    return params


def objective(trial, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros):
    """
    Función objetivo sin CV explícito.

    - Entrena en MESES_TRAIN (potencialmente undersampleado).
    - Valida en MESES_VAL_OPTUNA (sin undersampling, distribución real).
    - Para cada repetición:
        * Entrena un modelo por semilla (subconjunto de `semilleros`).
        * Hace ensemble de probabilidades en validación.
        * Calcula la ganancia en la MESETA del ENSEMBLE.
    - Devuelve a Optuna el PROMEDIO de ganancias de meseta sobre las repeticiones.
    """

    n_train = X_train.shape[0]

    # HP dinámicos según tamaño del dataset
    params_base = suggest_params_dynamic(trial, n_train)

    # num_boost_round y early_stopping también en función de n_train
    num_boost_round, early_stopping_rounds = get_boosting_schedule(n_train)

    # Datasets (una sola vez)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid)

    # ================================
    # 🔁 Repeticiones estilo APO (repe)
    # ================================

    total_seeds = len(semilleros)
    n_repe = max(1, int(N_REPE_OPTUNA))

    # Cantidad de seeds por repetición (como ksemillerio * repe en R)
    ksem = total_seeds // n_repe
    if ksem == 0:
        # Demasiadas repes para tan pocas seeds → caemos a 1 repe
        n_repe = 1
        ksem = total_seeds

    ganancias_repes = []   # ganancia de meseta por repetición
    umbrales_repes = []    # umbral de meseta por repetición
    N_opts_repes = []      # N óptimo por repetición
    best_iters_repes = []  # best_iteration promedio por repetición

    logger.info(
        f"🧪 Trial {trial.number} | total_seeds={total_seeds}, "
        f"repe={n_repe}, ksem={ksem}"
    )

    for r in range(n_repe):
        inicio = r * ksem
        fin = inicio + ksem
        semillas_repe = semilleros[inicio:fin]

        if len(semillas_repe) == 0:
            continue

        sum_preds_valid = None
        best_iters_this_repe = []

        logger.info(
            f"   🔁 Repe {r+1}/{n_repe} | seeds usadas: {len(semillas_repe)}"
        )

        # ==============
        # Semillerio BO
        # ==============
        for seed in semillas_repe:
            params = params_base.copy()
            params["seed"] = int(seed)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                feval=lgb_gan_eval,  # métrica de MESETA
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            best_iters_this_repe.append(int(model.best_iteration))

            # Predicciones en valid para esta seed
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

            # Acumular para ensemble de esta repetición
            if sum_preds_valid is None:
                sum_preds_valid = y_pred
            else:
                sum_preds_valid += y_pred

        # ================================
        # Ensemble final de ESTA repetición
        # ================================
        y_pred_ensemble = sum_preds_valid / len(semillas_repe)

        umbral_r, N_opt_r, ganancia_r = calcular_umbral_y_ganancia_meseta(
            y_pred=y_pred_ensemble,
            weight=w_valid,
        )

        ganancias_repes.append(float(ganancia_r))
        umbrales_repes.append(float(umbral_r))
        N_opts_repes.append(int(N_opt_r))
        best_iters_repes.append(float(np.mean(best_iters_this_repe)))

        logger.info(
            f"   ✅ Repe {r+1}/{n_repe} | "
            f"Ganancia meseta: ${ganancia_r:,.0f} | "
            f"N_opt: {N_opt_r:,} | umbral: {umbral_r:.6f}"
        )

    # Seguridad
    if not ganancias_repes:
        logger.warning("⚠️ No se pudo calcular ganancia en ninguna repetición.")
        return 0.0

    # =================================
    # Promedio estilo APO sobre repes
    # =================================
    ganancia_prom = float(np.mean(ganancias_repes))
    best_iter_prom = int(np.round(np.mean(best_iters_repes)))

    # Para almacenar un N_opt y umbral representativo,
    # tomamos el de la repetición con mejor ganancia
    idx_best_repe = int(np.argmax(ganancias_repes))
    umbral_ens = float(umbrales_repes[idx_best_repe])
    N_opt_ens = int(N_opts_repes[idx_best_repe])

    # ========= Metadata del trial =========
    trial.set_user_attr("ganancia_ensemble", ganancia_prom)           # promedio de mesetas
    trial.set_user_attr("ganancias_repes", ganancias_repes)           # detalle por repe
    trial.set_user_attr("umbral_ensemble", umbral_ens)                # de la mejor repe
    trial.set_user_attr("N_opt_ensemble", N_opt_ens)                  # de la mejor repe
    trial.set_user_attr("best_iter", best_iter_prom)
    trial.set_user_attr("n_train", int(n_train))

    logger.info(
        f"✅ Trial {trial.number} COMPLETADO | "
        f"Ganancia promedio meseta (sobre {n_repe} repes): ${ganancia_prom:,.0f}"
    )

    # Objetivo de Optuna: promedio de ganancias de meseta (como gan_mesetas_prom en APO)
    return ganancia_prom


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
    best_value = float(study.best_value)  # == ganancia_ensemble (promedio)

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
    ganancias_repes = best_trial.user_attrs.get("ganancias_repes", None)

    logger.info("✅ OPTIMIZACIÓN COMPLETADA")
    logger.info(f"🏅 Mejor trial: #{best_trial.number}")
    logger.info(f"💰 Mejor ganancia (objective / promedio meseta): ${best_value:,.0f}")

    if umbral_ensemble is not None and N_opt_ensemble is not None:
        logger.info(
            f"🎯 Umbral ensemble (valid): {umbral_ensemble:.6f} | "
            f"N óptimo ensemble: {N_opt_ensemble:,}"
        )

    if best_iter is not None:
        logger.info(f"🔁 Iteraciones promedio (best_iter): {best_iter}")

    logger.info(f"⚙️ Mejores hiperparámetros: {best_trial.params}")

    if ganancias_repes is not None:
        try:
            gr = [float(g) for g in ganancias_repes]
            logger.info(
                "📈 Ganancias de meseta por repetición: "
                + ", ".join(f"{g:,.0f}" for g in gr)
            )
        except Exception:
            logger.info("📈 Ganancias por repetición disponibles en user_attrs.")

    return study


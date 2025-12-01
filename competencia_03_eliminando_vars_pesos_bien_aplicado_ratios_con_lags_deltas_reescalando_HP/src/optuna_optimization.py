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
    gan = np.where(
    weight == 1.00002,
    GANANCIA_ACIERTO + COSTO_ESTIMULO,  # 780k - 20k = 760k
    COSTO_ESTIMULO                      # -20k para incentivado no-BAJA+2
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



import numpy as np
import optuna

def suggest_params(trial, n_train: int, base: dict) -> tuple[dict, int]:
    params = base.copy()

    # num_iterations ~ 2^U(0, 11.1)  (≈ 1 .. 2200)
    it_exp = trial.suggest_float("num_iterations_exp", 0.0, 11.1)
    num_boost_round = int(np.round(2.0 ** it_exp))

    lr_exp = trial.suggest_float("lr_exp", -8.0, -2.0)
    params["learning_rate"] = float(2.0 ** lr_exp)

    leaves_pow = trial.suggest_int("num_leaves_pow", 4, 10)
    num_leaves = int(2 ** leaves_pow)
    params["num_leaves"] = num_leaves

    max_leaf = max(2, n_train // max(2, num_leaves))
    max_pow = int(np.floor(np.log2(max_leaf)))
    leaf_pow = trial.suggest_int("min_data_in_leaf_pow", 0, max_pow)
    params["min_data_in_leaf"] = int(2 ** leaf_pow)

    params.update({
        "feature_fraction": float(trial.suggest_float("feature_fraction", 0.05, 1.0)),
        "bagging_fraction": float(trial.suggest_float("bagging_fraction", 0.5, 1.0)),
        "bagging_freq": int(trial.suggest_int("bagging_freq", 1, 10)),
        "min_gain_to_split": float(trial.suggest_float("min_gain_to_split", 0.0, 1.0)),
        "lambda_l2": float(trial.suggest_float("lambda_l2", 1e-4, 50.0, log=True)),
    })

    return params, num_boost_round



def objective(trial, X_train, y_train, w_train, X_valid, y_valid, w_valid, semilleros):
    n_train = X_train.shape[0]
    params_base, num_boost_round = suggest_params(trial, n_train, LGBM_PARAMS_BASE)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)

    semilleros = np.asarray(list(semilleros), dtype=int)
    total_seeds = len(semilleros)
    if total_seeds == 0:
        logger.warning("⚠️ semilleros vacío.")
        return 0.0

    n_repe = max(1, int(N_REPE_OPTUNA))
    n_repe = min(n_repe, total_seeds)

    rng = np.random.default_rng(10_000 + int(trial.number))
    semilleros_perm = rng.permutation(semilleros)
    bloques = np.array_split(semilleros_perm, n_repe)

    ganancias_repes, umbrales_repes, N_opts_repes = [], [], []

    logger.info(f"🧪 Trial {trial.number} | total_seeds={total_seeds}, repe={n_repe}")

    for r, semillas_repe in enumerate(bloques, start=1):
        if len(semillas_repe) == 0:
            continue

        sum_preds_valid = np.zeros(X_valid.shape[0], dtype=float)
        logger.info(f"   🔁 Repe {r}/{n_repe} | seeds: {len(semillas_repe)}")

        for seed in semillas_repe:
            params = params_base.copy()
            params["seed"] = int(seed)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(period=0)],
            )

            sum_preds_valid += model.predict(X_valid, num_iteration=num_boost_round)

        y_pred_ensemble = sum_preds_valid / float(len(semillas_repe))
        umbral_r, N_opt_r, ganancia_r = calcular_umbral_y_ganancia_meseta(y_pred_ensemble, w_valid)

        ganancias_repes.append(float(ganancia_r))
        umbrales_repes.append(float(umbral_r))
        N_opts_repes.append(int(N_opt_r))

    if not ganancias_repes:
        logger.warning("⚠️ No se pudo calcular ganancia en ninguna repetición.")
        return 0.0

    ganancia_prom = float(np.mean(ganancias_repes))
    idx_best_repe = int(np.argmax(ganancias_repes))

    trial.set_user_attr("lgb_params", params_base)
    trial.set_user_attr("best_iter", int(num_boost_round))
    trial.set_user_attr("ganancia_ensemble", ganancia_prom)
    trial.set_user_attr("ganancias_repes", ganancias_repes)
    trial.set_user_attr("umbral_ensemble", float(umbrales_repes[idx_best_repe]))
    trial.set_user_attr("N_opt_ensemble", int(N_opts_repes[idx_best_repe]))
    trial.set_user_attr("n_train", int(n_train))

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
    Objetivo: ganancia (meseta) del ENSEMBLE (promedio sobre repes).
    """
    study = crear_estudio_optuna(seed, load_if_exists=True)
    n_trials = n_trials or N_TRIALS

    completados = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"📊 Trials completados: {completados}/{n_trials}")

    restantes = n_trials - completados
    if restantes > 0:
        logger.info(f"🚀 Ejecutando {restantes} trials adicionales...")

        # 🔥 HABILITAMOS EL LOGGING AUTOMÁTICO DE OPTUNA
        optuna.logging.enable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.INFO)

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

    best_trial = study.best_trial
    best_value = float(study.best_value)

    logger.info("✅ OPTIMIZACIÓN COMPLETADA")
    logger.info(f"🏅 Mejor trial: #{best_trial.number}")
    logger.info(f"💰 Mejor ganancia (objective / promedio meseta): ${best_value:,.0f}")

    if "umbral_ensemble" in best_trial.user_attrs and "N_opt_ensemble" in best_trial.user_attrs:
        logger.info(
            f"🎯 Umbral ensemble (valid): {best_trial.user_attrs['umbral_ensemble']:.6f} | "
            f"N óptimo ensemble: {best_trial.user_attrs['N_opt_ensemble']:,}"
        )

    if "best_iter" in best_trial.user_attrs:
        logger.info(f"🔁 Iteraciones promedio (best_iter): {int(best_trial.user_attrs['best_iter'])}")

    if "lgb_params" in best_trial.user_attrs:
        logger.info(f"⚙️ Mejores hiperparámetros (LGBM): {best_trial.user_attrs['lgb_params']}")

    return study


# src/training_predict.py

import os
import numpy as np
import lightgbm as lgb

from config.config import MODEL_DIR_TEST_FINAL
from src.utils import logger


def entrenar_ensemble_test_final(
    X_train,
    y_train,
    w_train,
    X_test,
    params: dict,
    num_boost_round: int,
    semillas,
    N_envios: int,
    guardar_modelos: bool = True,
    model_dir: str | None = None,
):
    """
    Entrena/carga un modelo LightGBM por cada seed en `semillas` usando X_train completo,
    predice sobre X_test, arma el ensemble por promedio y aplica un corte
    por N_envios (por ejemplo N_opt_APO).

    Comportamiento sobre modelos:
      - Si el archivo del modelo para una seed EXISTE -> se carga (♻️ reutiliza).
      - Si NO existe -> se entrena y (si guardar_modelos=True) se guarda en disco.

    Parámetros
    ----------
    X_train, y_train, w_train : train completo hasta el mes previo al test (MESES_TRAIN_COMPLETO_PARA_TEST_FINAL)
    X_test                   : features de MES_TEST_FINAL (sin target)
    params                   : dict con hiperparámetros finales (LGBM_PARAMS_BASE + best_params)
    num_boost_round          : best_iter (promedio de Optuna)
    semillas                 : lista de seeds para el ensemble final (SEMILLAS_ENSEMBLE)
    N_envios                 : N óptimo obtenido por APO (N_opt_APO)
    guardar_modelos          : si True, guarda los modelos en disco
    model_dir                : carpeta donde se guardan los modelos finales

    Devuelve
    --------
    dict con:
      - prob_test_ensemble : np.array de probabilidades promedio en test
      - pred_test_binaria  : np.array binario (0/1) usando top-N_envios
      - N_envios_usado     : N efectivo (min(N_envios, len(test)))
      - semillas           : lista de semillas usadas
    """
    model_dir = model_dir or MODEL_DIR_TEST_FINAL
    if guardar_modelos:
        os.makedirs(model_dir, exist_ok=True)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)

    preds_test = []

    for seed in semillas:
        seed = int(seed)
        params_seed = params.copy()
        params_seed["seed"] = seed

        model_path = os.path.join(model_dir, f"lgbm_testfinal_seed_{seed}_it{num_boost_round}.txt")

        if os.path.exists(model_path):
            # ♻️ Reutilizar modelo ya entrenado
            logger.info(
                f"♻️ Modelo FINAL test seed={seed} ya existe. Se carga desde: {model_path}"
            )
            model = lgb.Booster(model_file=model_path)
        else:
            # 🌱 Entrenar modelo nuevo
            logger.info(f"🌳 Entrenando modelo FINAL test seed={seed}...")
            model = lgb.train(
                params_seed,
                dtrain,
                num_boost_round=num_boost_round,
            )

            if guardar_modelos:
                model.save_model(model_path)
                logger.info(f"💾 Modelo FINAL test guardado en: {model_path}")

        # Predicciones en test para esta seed
        preds_test.append(model.predict(X_test, num_iteration=num_boost_round))

    # ============================
    # 🌐 Ensemble (promedio) en test
    # ============================
    prob_test_ensemble = np.mean(np.vstack(preds_test), axis=0)

    # ============================
    # 🎯 Top-N según N_envios (N_opt_APO)
    # ============================
    n_test = prob_test_ensemble.shape[0]
    N_eff = min(int(N_envios), n_test)

    orden_test = np.argsort(prob_test_ensemble)[::-1]
    pred_test_binaria = np.zeros(n_test, dtype=int)
    pred_test_binaria[orden_test[:N_eff]] = 1

    logger.info(
        f"📦 Ensemble final test: N_envios={N_envios} "
        f"(efectivo {N_eff}), se etiquetan {N_eff} clientes con Predicted=1."
    )

    return {
        "prob_test_ensemble": prob_test_ensemble,
        "pred_test_binaria": pred_test_binaria,
        "N_envios_usado": N_eff,
        "semillas": list(semillas),
    }

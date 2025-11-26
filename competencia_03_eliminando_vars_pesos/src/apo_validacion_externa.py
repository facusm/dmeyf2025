# src/apo_validacion_externa.py

import os
import numpy as np
import lightgbm as lgb

from config.config import (
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
    MODEL_DIR_VAL_EXT,
    APO_K_SEM,
    APO_N_APO,
    APO_CORTES_ENVIO,
)
from src.utils import logger


def _calcular_ganancias_por_cortes(y_pred, weight, cortes):
    """
    Calcula la ganancia para cada N de `cortes`, siguiendo la lógica de la cátedra.

    - Ordena por probabilidad descendente.
    - Construye el vector de "ganancia por cliente".
    - Hace cumsum y toma el valor en la posición N para cada corte.
    """
    y_pred = np.asarray(y_pred)
    weight = np.asarray(weight)

    n = y_pred.shape[0]
    if n == 0:
        return np.zeros(len(cortes), dtype=float)

    gan = np.where(
        weight == 1.00002,
        GANANCIA_ACIERTO + COSTO_ESTIMULO,  # BAJA+2: 780k + (-20k) = 760k
        COSTO_ESTIMULO  # CONTINUA o BAJA +1: -20k
    )

    orden = np.argsort(y_pred)[::-1]
    gan_ordenada = gan[orden]
    gan_acum = np.cumsum(gan_ordenada)

    ganancias_cortes = []
    for N in cortes:
        if N <= 0:
            ganancias_cortes.append(0.0)
        else:
            N_eff = min(N, n)
            ganancias_cortes.append(float(gan_acum[N_eff - 1]))

    return np.array(ganancias_cortes, dtype=float)


def entrenar_modelos_val_externa(
    X_train,
    y_train,
    w_train,
    params,
    num_boost_round,
    semillas,
    model_dir: str | None = None,
):
    """
    Entrena (si hace falta) un modelo LightGBM por cada seed en `semillas` y lo guarda.

    - Usa SIEMPRE el mismo train: X_train_inicial (2019..202105 en tu caso).
    - Reutiliza modelos si el archivo ya existe.
    """
    model_dir = model_dir or MODEL_DIR_VAL_EXT
    os.makedirs(model_dir, exist_ok=True)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)

    for seed in semillas:
        seed = int(seed)
        model_path = os.path.join(model_dir, f"lgbm_valext_seed_{seed}_it{num_boost_round}.txt")

        if os.path.exists(model_path):
            logger.info(f"♻️ Modelo val_ext para seed={seed} ya existe → se reutiliza.")
            continue

        logger.info(f"🌱 Entrenando modelo val_ext para seed={seed}...")
        params_seed = params.copy()
        params_seed["seed"] = seed

        model = lgb.train(
            params_seed,
            dtrain,
            num_boost_round=num_boost_round,
        )

        model.save_model(model_path)
        logger.info(f"💾 Modelo val_ext guardado en: {model_path}")


def seleccionar_N_optimo_APO(
    X_valid,
    w_valid,
    semillas,
    cortes=None,
    model_dir: str | None = None,
    ksem: int | None = None,
    n_apo: int | None = None,
    num_boost_round: int | None = None,
):
    """
    Replica la lógica APO sobre la validación externa (p.ej. mes 202107).

    - Toma todos los modelos entrenados en `model_dir` (uno por seed).
    - Parte `semillas` en bloques de tamaño ksem (ksemillerio).
    - Cada bloque define una repetición APO (n_apo).
    - Para cada repetición:
        * carga los modelos de esas seeds,
        * predice sobre X_valid,
        * hace ensemble promediando las probabilidades,
        * calcula la ganancia en cada N de `cortes`.
    - Devuelve:
        N_opt (corte con mayor ganancia promedio),
        ganancias_prom (ganancia promedio por corte),
        mganancias (matriz APO_N_APO x len(cortes)).
    """
    model_dir = model_dir or MODEL_DIR_VAL_EXT
    cortes = cortes or APO_CORTES_ENVIO
    ksem = ksem or APO_K_SEM
    n_apo = n_apo or APO_N_APO

    semillas = list(semillas)
    total_seeds = len(semillas)

    if total_seeds < ksem:
        raise ValueError(
            f"SEMILLAS_APO insuficientes: total_seeds={total_seeds} < ksem={ksem}. "
            "No se puede correr APO (no alcanza para 1 repetición)."
        )
    
    if num_boost_round is None:
        raise ValueError("num_boost_round es None. Pasá best_iter desde main.py.")

    esperado = ksem * n_apo
    if total_seeds < esperado:
        logger.warning(
            f"⚠️ SEMILLAS_APO tiene {total_seeds} seeds pero se esperaban "
            f"{esperado} (ksem={ksem}, APO={n_apo}). Se usará lo que haya."
        )
        n_apo = total_seeds // ksem

    mganancias = np.zeros((n_apo, len(cortes)), dtype=float)

    logger.info(
        f"📊 APO sobre validación externa: ksem={ksem}, APO={n_apo}, "
        f"total_seeds={total_seeds}, cortes={cortes}"
    )

    for i_apo in range(n_apo):
        inicio = i_apo * ksem
        fin = inicio + ksem
        semillas_repe = semillas[inicio:fin]

        if not semillas_repe:
            continue

        logger.info(
            f"   🔁 Repe APO {i_apo+1}/{n_apo} | seeds usadas: {semillas_repe}"
        )

        sum_preds = None

        for seed in semillas_repe:
            seed = int(seed)
            model_path = os.path.join(model_dir, f"lgbm_valext_seed_{seed}_it{num_boost_round}.txt")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No se encontró el modelo para seed={seed} en {model_path}. "
                    "Ejecutá primero entrenar_modelos_val_externa()."
                )

            model = lgb.Booster(model_file=model_path)
            y_pred = model.predict(X_valid, num_iteration=num_boost_round)

            if sum_preds is None:
                sum_preds = y_pred
            else:
                sum_preds += y_pred

        y_pred_ensemble = sum_preds / len(semillas_repe)

        ganancias_cortes = _calcular_ganancias_por_cortes(
            y_pred=y_pred_ensemble,
            weight=w_valid,
            cortes=cortes,
        )

        mganancias[i_apo, :] = ganancias_cortes

        logger.info(
            "   ✅ Ganancias por corte en esta repe: "
            + ", ".join(
                f"N={cortes[j]} → ${ganancias_cortes[j]:,.0f}"
                for j in range(len(cortes))
            )
        )

    ganancias_prom = mganancias.mean(axis=0)
    idx_best = int(np.argmax(ganancias_prom))
    N_opt = int(cortes[idx_best])
    gan_best = float(ganancias_prom[idx_best])

    logger.info(
        "📈 Ganancia promedio por corte (APO): "
        + ", ".join(
            f"N={cortes[j]} → ${ganancias_prom[j]:,.0f}"
            for j in range(len(cortes))
        )
    )
    logger.info(
        f"🏅 N óptimo APO sobre validación externa: N={N_opt} "
        f"con ganancia promedio=${gan_best:,.0f}"
    )

    return N_opt, ganancias_prom, mganancias

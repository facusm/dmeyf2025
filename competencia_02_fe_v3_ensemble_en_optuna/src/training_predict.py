# src/training_predict.py

import os
import json
import numpy as np
import lightgbm as lgb
from datetime import datetime

from config.config import MODELOS_PATH, SEMILLAS_ENSEMBLE, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


# ==============================================================================
# 📌 Entrenar un solo modelo para una semilla
# ==============================================================================
def entrenar_modelo_single_seed(X_train, y_train, w_train, params, num_boost_round, seed):
    params_seed = params.copy()
    params_seed["seed"] = seed

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)

    model = lgb.train(
        params_seed,
        dtrain,
        num_boost_round=num_boost_round,
    )
    return model


# ==============================================================================
# 📌 Guardar metadatos de umbrales "limpios" (fase 1)
# ==============================================================================
def guardar_metadatos_umbral(nombre_experimento, semillas, umbrales, ganancias):
    metadatos = {
        "timestamp": datetime.now().isoformat(),
        "experimento": nombre_experimento,
        "semillas": semillas,
        "umbrales_individuales": umbrales,
        "ganancias_individuales": ganancias,
        "umbral_promedio": float(np.mean(umbrales)),
        "nota": "Umbrales calculados con modelos entrenados hasta 202104 (FASE 1)"
    }

    os.makedirs(MODELOS_PATH, exist_ok=True)
    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")

    with open(path, "w") as f:
        json.dump(metadatos, f, indent=2)

    logger.info(f"💾 Metadatos FASE 1 guardados en: {path}")
    return path


# ==============================================================================
# 📌 Cargar metadatos si ya existen
# ==============================================================================
def cargar_metadatos_umbral(nombre_experimento):
    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        metadatos = json.load(f)

    logger.info(f"📥 Metadatos cargados desde {path}")
    return metadatos


# ==============================================================================
# 📌 ENTRENAMIENTO MULTI-SEED COMPLETO CON DOS FASES + CHECKPOINTS
# ==============================================================================
def entrenar_ensemble_multisemilla(
    X_train_inicial,
    y_train_inicial,
    w_train_inicial,
    X_train_completo,
    y_train_completo,
    w_train_completo,
    X_valid,
    w_valid,
    X_test,
    params,
    num_boost_round,
    semillas=None,
    nombre_experimento=NOMBRE_EXPERIMENTO,
):

    semillas = semillas or SEMILLAS_ENSEMBLE
    os.makedirs(MODELOS_PATH, exist_ok=True)

    # Paths por semilla
    def path_fase1(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_fase1.txt")
    def path_fase2(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_final.txt")

    # Intentar cargar metadatos
    metadatos = cargar_metadatos_umbral(nombre_experimento)
    tiene_metadatos = metadatos is not None

    probabilidades_valid = []
    probabilidades_test = []
    umbrales_individuales = []
    ganancias_individuales = []

    # ==========================================================================
    # 📌 CASO 1 — Ya existen FASE 2 + metadatos → SOLO predecimos Agosto
    # ==========================================================================
    if tiene_metadatos and all(os.path.exists(path_fase2(s)) for s in semillas):
        logger.info("\n🔄 Cargando modelos FASE 2 + aplicando umbrales guardados (FASE 1)")

        umbrales_individuales = metadatos["umbrales_individuales"]

        for seed in semillas:
            logger.info(f"📂 Semilla {seed}: cargando modelo final (FASE 2) ...")
            model = lgb.Booster(model_file=path_fase2(seed))

            y_pred_test = model.predict(X_test)
            probabilidades_test.append(y_pred_test)

        return {
            "probabilidades_valid": [],
            "probabilidades_test": probabilidades_test,
            "umbrales_individuales": umbrales_individuales,
        }

    # ==========================================================================
    # 📌 CASO 2 — Entrenar lo que falte (FASE 1 y FASE 2)
    # ==========================================================================
    logger.info("\n🚀 Entrenando ensemble multisemilla con checkpoints...")

    for seed in semillas:

        # ------------------ FASE 1 ------------------
        if os.path.exists(path_fase1(seed)):
            logger.info(f"🔁 Cargando modelo FASE 1 existente para seed={seed}")
            model_f1 = lgb.Booster(model_file=path_fase1(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 1 para seed={seed}")
            model_f1 = entrenar_modelo_single_seed(
                X_train_inicial, y_train_inicial, w_train_inicial,
                params, num_boost_round, seed
            )
            model_f1.save_model(path_fase1(seed))
            logger.info(f"💾 Guardado modelo FASE 1 en {path_fase1(seed)}")

        # Predicción sobre junio → cálculo umbral individual
        y_pred_valid = model_f1.predict(X_valid)
        umbral, _, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)

        umbrales_individuales.append(float(umbral))
        ganancias_individuales.append(float(ganancia))
        probabilidades_valid.append(y_pred_valid)

        # ------------------ FASE 2 ------------------
        if os.path.exists(path_fase2(seed)):
            logger.info(f"🔁 Cargando modelo FASE 2 existente para seed={seed}")
            model_f2 = lgb.Booster(model_file=path_fase2(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 2 para seed={seed}")
            model_f2 = entrenar_modelo_single_seed(
                X_train_completo, y_train_completo, w_train_completo,
                params, num_boost_round, seed
            )
            model_f2.save_model(path_fase2(seed))
            logger.info(f"💾 Guardado modelo FASE 2 en {path_fase2(seed)}")

        # Predicción final (agosto)
        y_pred_test = model_f2.predict(X_test)
        probabilidades_test.append(y_pred_test)

    # Guardar metadatos limpios de umbrales (solo una vez)
    guardar_metadatos_umbral(nombre_experimento, semillas, umbrales_individuales, ganancias_individuales)

    return {
        "probabilidades_valid": probabilidades_valid,
        "probabilidades_test": probabilidades_test,
        "umbrales_individuales": umbrales_individuales,
    }


# ==============================================================================
# 📌 EVALUACIÓN FINAL DEL ENSEMBLE (EN VALIDACIÓN + TEST)
# ==============================================================================
def evaluar_ensemble_y_umbral(
    probabilidades_valid,
    probabilidades_test,
    w_valid,
    umbrales_individuales,
):
    """
    - Hace promedio de predicciones en validación
    - Obtiene umbral óptimo del ensemble
    - Aplica ese umbral al promedio de predicciones del test
    """

    # Ensemble validación
    if len(probabilidades_valid) > 0:
        prob_valid_ens = np.mean(probabilidades_valid, axis=0)
        umbral_opt, N_opt, ganancia_opt, _ = mejor_umbral_probabilidad(
            prob_valid_ens,
            w_valid,
        )
    else:
        # Ya estaba todo cargado desde checkpoint
        umbral_opt = float(np.mean(umbrales_individuales))
        N_opt = None
        ganancia_opt = None

    # Ensemble test
    prob_test_ens = np.mean(probabilidades_test, axis=0)
    pred_bin = (prob_test_ens >= umbral_opt).astype(int)

    return {
        "probabilidades_test_ensemble": prob_test_ens,
        "prediccion_binaria": pred_bin,
        "umbral_optimo_ensemble": umbral_opt,
        "ganancia_maxima_valid": ganancia_opt,
        "N_en_umbral": N_opt,
        "umbral_promedio_individual": float(np.mean(umbrales_individuales)),
    }

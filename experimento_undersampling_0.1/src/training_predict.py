# ===============================================
# training_predict.py — versión completa y corregida
# ===============================================

import os
import json
import numpy as np
import lightgbm as lgb
from datetime import datetime

from config.config import MODELOS_PATH, SEMILLAS_ENSEMBLE, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


# ===========================================================
# 📌 Entrenar modelo simple con una semilla
# ===========================================================
def entrenar_modelo_single_seed(X_train, y_train, w_train, params, num_boost_round, seed):
    params_seed = params.copy()
    params_seed["seed"] = seed

    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train)
    model = lgb.train(params_seed, dtrain, num_boost_round=num_boost_round)

    return model


# ===========================================================
# 📌 Guardar metadatos de umbrales (FASE 1)
# ===========================================================
def guardar_metadatos_umbral(nombre_experimento, semillas, umbrales, ganancias):
    data = {
        "timestamp": datetime.now().isoformat(),
        "experimento": nombre_experimento,
        "semillas": semillas,
        "umbrales_individuales": umbrales,
        "ganancias_individuales": ganancias,
        "umbral_promedio": float(np.mean(umbrales)),
        "nota": "Umbrales obtenidos en validación (202106) con modelos FASE 1"
    }

    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"💾 Metadatos guardados en: {path}")
    return path


# ===========================================================
# 📌 Cargar metadatos si existen
# ===========================================================
def cargar_metadatos_umbral(nombre_experimento):
    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


# ===========================================================
# 📌 ENTRENAR ENSEMBLE MULTISEMILLA (FASE 1 + FASE 2)
# ===========================================================
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
    guardar_modelos=True,
    nombre_experimento=NOMBRE_EXPERIMENTO,
):

    semillas = semillas or SEMILLAS_ENSEMBLE
    os.makedirs(MODELOS_PATH, exist_ok=True)

    # Helpers de paths
    def path_f1(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_fase1.txt")
    def path_f2(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_final.txt")

    # Metadatos limpios ya existentes?
    metadatos = cargar_metadatos_umbral(nombre_experimento)
    tiene_metadatos = metadatos is not None

    probabilidades_valid = []
    probabilidades_test = []
    umbrales_individuales = []
    ganancias_individuales = []

    # ===========================================================
    # 🔄 CASO 1 — Todos los modelos FASE 2 + metadatos → SOLO PREDICT
    # ===========================================================
    if tiene_metadatos and all(os.path.exists(path_f2(s)) for s in semillas):
        logger.info("\n🔁 Cargando modelos FASE 2 + metadatos existentes (no se recalcula nada)")

        umbrales_individuales = metadatos["umbrales_individuales"]

        for seed in semillas:
            logger.info(f"📂 Cargando modelo final seed={seed}")
            model = lgb.Booster(model_file=path_f2(seed))
            probabilidades_test.append(model.predict(X_test))

        return {
            "probabilidades_valid": [],
            "probabilidades_test": probabilidades_test,
            "umbrales_individuales": umbrales_individuales,
        }

    # ===========================================================
    # 🚀 CASO 2 — Entrenar desde cero o completar seeds faltantes
    # ===========================================================
    logger.info("\n🚀 Entrenando ensemble multisemilla con checkpoints...")

    for seed in semillas:

        # ---------------------------
        # ✔ FASE 1
        # ---------------------------
        if os.path.exists(path_f1(seed)):
            logger.info(f"🔁 Cargando FASE 1 seed={seed}")
            model_f1 = lgb.Booster(model_file=path_f1(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 1 seed={seed}")
            model_f1 = entrenar_modelo_single_seed(
                X_train_inicial, y_train_inicial, w_train_inicial,
                params, num_boost_round, seed
            )
            if guardar_modelos:
                model_f1.save_model(path_f1(seed))
                logger.info(f"💾 Guardado FASE 1 → {path_f1(seed)}")

        # Predicción validación
        y_pred_valid = model_f1.predict(X_valid)
        umbral, _, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)

        probabilidades_valid.append(y_pred_valid)
        umbrales_individuales.append(float(umbral))
        ganancias_individuales.append(float(ganancia))

        # ---------------------------
        # ✔ FASE 2
        # ---------------------------
        if os.path.exists(path_f2(seed)):
            logger.info(f"🔁 Cargando FASE 2 seed={seed}")
            model_f2 = lgb.Booster(model_file=path_f2(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 2 seed={seed}")
            model_f2 = entrenar_modelo_single_seed(
                X_train_completo, y_train_completo, w_train_completo,
                params, num_boost_round, seed
            )
            if guardar_modelos:
                model_f2.save_model(path_f2(seed))
                logger.info(f"💾 Guardado FASE 2 → {path_f2(seed)}")

        # Pred test
        probabilidades_test.append(model_f2.predict(X_test))

    # Guardamos umbrales LIMPIOS de FASE 1
    guardar_metadatos_umbral(nombre_experimento, semillas, umbrales_individuales, ganancias_individuales)

    return {
        "probabilidades_valid": probabilidades_valid,
        "probabilidades_test": probabilidades_test,
        "umbrales_individuales": umbrales_individuales,
    }


# ===========================================================
# 📌 EVALUAR ENSEMBLE Y CALCULAR UMBRAL
# ===========================================================
def evaluar_ensemble_y_umbral(probabilidades_valid, probabilidades_test, w_valid, umbrales_individuales):


    tiene_valid = len(probabilidades_valid) > 0

    # ===========================================================
    # 🎯 Caso 1 — Hay validación (primera corrida)
    # ===========================================================
    if tiene_valid:
        matriz_valid = np.array(probabilidades_valid)
        proba_valid_ensemble = matriz_valid.mean(axis=0)

        umbral_opt, N_opt, ganancia_opt, curva = mejor_umbral_probabilidad(
            proba_valid_ensemble,
            w_valid
        )

        logger.info(f"\n🎯 Umbral óptimo calculado en validación = {umbral_opt:.6f}")

    # ===========================================================
    # 🎯 Caso 2 — NO hay validación (correr desde checkpoints)
    # ===========================================================
    else:
        umbral_opt = float(np.mean(umbrales_individuales))
        N_opt = 0
        ganancia_opt = 0
        curva = None
        proba_valid_ensemble = np.array([])

        logger.info(f"\n🔁 No hay validación. Usando umbral promedio = {umbral_opt:.6f}")

    # ===========================================================
    # 🚀 Ensemble en test final
    # ===========================================================
    matriz_test = np.array(probabilidades_test)
    proba_test_ensemble = matriz_test.mean(axis=0)

    pred_binaria = (proba_test_ensemble >= umbral_opt).astype(int)

    return {
        "umbral_optimo_ensemble": umbral_opt,
        "N_en_umbral": N_opt,
        "ganancia_maxima_valid": ganancia_opt,
        "umbral_promedio_individual": float(np.mean(umbrales_individuales)),
        "probabilidades_valid_ensemble": proba_valid_ensemble,
        "probabilidades_test_ensemble": proba_test_ensemble,
        "prediccion_binaria": pred_binaria,
        "N_enviados": int(pred_binaria.sum()),
        "curva_ganancia": curva,
    }

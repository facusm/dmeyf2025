import os
import json
import numpy as np
import lightgbm as lgb
from datetime import datetime

from config.config import MODELOS_PATH, SEMILLAS_ENSEMBLE, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


# ===========================================================
# 📌 Entrenar un modelo simple con una seed
# ===========================================================
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


# ===========================================================
# 📌 Guardar metadatos limpios (umbrales de FASE 1)
# ===========================================================
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

    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")
    with open(path, "w") as f:
        json.dump(metadatos, f, indent=2)
    return path


# ===========================================================
# 📌 Cargar metadatos limpios si existen
# ===========================================================
def cargar_metadatos_umbral(nombre_experimento):
    path = os.path.join(MODELOS_PATH, f"{nombre_experimento}_umbrales.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================
# 📌 Función PRINCIPAL: entrenar ensemble multisemilla
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
    nombre_experimento=NOMBRE_EXPERIMENTO,
):

    semillas = semillas or SEMILLAS_ENSEMBLE
    os.makedirs(MODELOS_PATH, exist_ok=True)

    # Paths por semilla
    def path_fase1(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_fase1.txt")
    def path_fase2(seed): return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_final.txt")

    # ¿Ya hay metadatos limpios?
    metadatos = cargar_metadatos_umbral(nombre_experimento)
    tiene_metadatos = metadatos is not None

    probabilidades_valid = []
    probabilidades_test = []
    umbrales_individuales = []
    ganancias_individuales = []

    # ===========================================================
    # 🔄 CASO 1: Modelos FASE 2 + metadatos → SOLO predecimos
    # ===========================================================
    if tiene_metadatos and all(os.path.exists(path_fase2(s)) for s in semillas):
        logger.info("\n🔄 Cargando modelos FASE 2 + usando umbrales limpios guardados")

        umbrales_individuales = metadatos["umbrales_individuales"]

        for idx, seed in enumerate(semillas):
            logger.info(f"📂 Semilla {seed}: cargando modelo final...")
            model = lgb.Booster(model_file=path_fase2(seed))

            # Solo predecimos test (agosto)
            y_pred_test = model.predict(X_test)
            probabilidades_test.append(y_pred_test)

        return {
            "probabilidades_valid": [],  # no se usan
            "probabilidades_test": probabilidades_test,
            "umbrales_individuales": umbrales_individuales,
        }

    # ===========================================================
    # 🚀 CASO 2: Entrenar todo desde cero o completar seeds faltantes
    # ===========================================================
    logger.info("\n🚀 Entrenando ensemble multisemilla con checkpoints")

    for seed in semillas:

        # -----------------------
        # ✔ FASE 1 CHECKPOINT
        # -----------------------
        if os.path.exists(path_fase1(seed)):
            logger.info(f"🔁 Cargando modelo FASE 1 de seed={seed}")
            model_f1 = lgb.Booster(model_file=path_fase1(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 1 seed={seed}")
            model_f1 = entrenar_modelo_single_seed(
                X_train_inicial, y_train_inicial, w_train_inicial,
                params, num_boost_round, seed
            )
            model_f1.save_model(path_fase1(seed))
            logger.info(f"💾 Guardado modelo FASE 1 → {path_fase1(seed)}")

        # Predicción limpia sobre junio (validación externa)
        y_pred_valid = model_f1.predict(X_valid)
        umbral, _, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)

        umbrales_individuales.append(float(umbral))
        ganancias_individuales.append(float(ganancia))
        probabilidades_valid.append(y_pred_valid)

        # -----------------------
        # ✔ FASE 2 CHECKPOINT
        # -----------------------
        if os.path.exists(path_fase2(seed)):
            logger.info(f"🔁 Cargando modelo FASE 2 de seed={seed}")
            model_f2 = lgb.Booster(model_file=path_fase2(seed))
        else:
            logger.info(f"⏳ Entrenando FASE 2 seed={seed}")
            model_f2 = entrenar_modelo_single_seed(
                X_train_completo, y_train_completo, w_train_completo,
                params, num_boost_round, seed
            )
            model_f2.save_model(path_fase2(seed))
            logger.info(f"💾 Guardado modelo FASE 2 → {path_fase2(seed)}")

        # Predicción test (agosto)
        y_pred_test = model_f2.predict(X_test)
        probabilidades_test.append(y_pred_test)

    # Guardamos umbrales limpios
    guardar_metadatos_umbral(nombre_experimento, semillas, umbrales_individuales, ganancias_individuales)

    # ===========================================================
    # FIN
    # ===========================================================
    return {
        "probabilidades_valid": probabilidades_valid,
        "probabilidades_test": probabilidades_test,
        "umbrales_individuales": umbrales_individuales,
    }



# ===========================================================
# 📌 Función: evaluar ensemble y calcular el umbral óptimo
# ===========================================================
def evaluar_ensemble_y_umbral(
    probabilidades_valid,
    probabilidades_test,
    w_valid,
    umbrales_individuales,
):
    """
    Evalúa el ensemble multisemilla:
    - Si hay predicciones de validación → calcula umbral óptimo del ensemble.
    - Si NO hay predicciones (modelos cargados) → usa promedio de umbrales individuales.
    """

    tiene_prob_valid = len(probabilidades_valid) > 0 and len(probabilidades_valid[0]) > 0

    if tiene_prob_valid:
        matriz_valid = np.array(probabilidades_valid)
        probabilidades_valid_ensemble = np.mean(matriz_valid, axis=0)

        umbral_ensemble, N_ensemble, ganancia_ensemble, curva = mejor_umbral_probabilidad(
            probabilidades_valid_ensemble, w_valid
        )
    else:
        umbral_ensemble = float(np.mean(umbrales_individuales))
        N_ensemble = 0
        ganancia_ensemble = 0
        probabilidades_valid_ensemble = np.array([])

    # Ensemble en test
    matriz_test = np.array(probabilidades_test)
    probabilidades_test_ensemble = np.mean(matriz_test, axis=0)
    pred_binaria = (probabilidades_test_ensemble >= umbral_ensemble).astype(int)

    return {
        "umbral_optimo_ensemble": umbral_ensemble,
        "N_en_umbral": N_ensemble,
        "ganancia_maxima_valid": ganancia_ensemble,
        "umbral_promedio_individual": float(np.mean(umbrales_individuales)),
        "probabilidades_valid_ensemble": probabilidades_valid_ensemble,
        "probabilidades_test_ensemble": probabilidades_test_ensemble,
        "prediccion_binaria": pred_binaria,
    }

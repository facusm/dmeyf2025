# src/training_predict.py

import os
import json
import numpy as np
import lightgbm as lgb

from config.config import MODELOS_PATH, SEMILLAS_ENSEMBLE, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


# ============================================================
# Helpers para guardar / cargar métricas de validación por semilla
# ============================================================

def _ruta_metricas_seed(nombre_experimento: str, seed: int) -> str:
    """
    Devuelve la ruta del archivo JSON donde se guardan las métricas de validación
    para una semilla determinada.
    """
    return os.path.join(MODELOS_PATH, f"{nombre_experimento}_seed{seed}_metrics.json")


def _guardar_metricas_seed(nombre_experimento: str, seed: int,
                           umbral: float, N_opt: int, ganancia: float) -> None:
    """
    Guarda en disco el umbral óptimo y métricas de validación de una semilla.
    """
    ruta = _ruta_metricas_seed(nombre_experimento, seed)
    info = {
        "umbral": float(umbral),
        "N_opt": int(N_opt),
        "ganancia": float(ganancia),
    }
    with open(ruta, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"💾 Métricas de validación guardadas en: {ruta}")


def _cargar_metricas_seed(nombre_experimento: str, seed: int):
    """
    Carga métricas de validación desde disco para una semilla.
    Devuelve un dict o None si no existe el archivo.
    """
    ruta = _ruta_metricas_seed(nombre_experimento, seed)
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r") as f:
        return json.load(f)


# ============================================================
# Entrenamiento por semilla
# ============================================================

def entrenar_modelo_single_seed(X_train, y_train, w_train, params, num_boost_round, seed):
    """
    Entrena un modelo LightGBM con una semilla específica.
    """
    params_seed = params.copy()
    params_seed["seed"] = seed

    train_dataset = lgb.Dataset(X_train, label=y_train, weight=w_train)

    model = lgb.train(
        params_seed,
        train_dataset,
        num_boost_round=num_boost_round,
    )

    logger.info(f"✅ Modelo entrenado con semilla {seed}")
    return model


# ============================================================
# Entrenamiento / carga del ensemble multisemilla
# ============================================================

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
    """
    Entrena un ensemble de modelos con múltiples semillas.
    Si el proceso se interrumpe, al reanudar verifica si los modelos
    ya existen en MODELOS_PATH y los carga en lugar de reentrenarlos.

    🔹 El umbral óptimo por semilla se calcula SOLO cuando se entrena por primera vez
       (con X_train_inicial + valid externa) y se guarda en disco.
    🔹 Si el modelo ya existe en disco, se cargan:
        - El modelo final desde MODELOS_PATH
        - El umbral y métricas desde el JSON correspondiente (si existe)
    🔹 Siempre se recalculan las probabilidades sobre X_valid y X_test, para poder
       armar el ensemble aunque el modelo haya sido entrenado en otra corrida.
    """
    semillas = semillas or SEMILLAS_ENSEMBLE

    probabilidades_valid = []
    probabilidades_test = []
    umbrales_individuales = []
    ganancias_individuales = []
    modelos_finales = []

    os.makedirs(MODELOS_PATH, exist_ok=True)

    logger.info(f"\n{'=' * 60}")
    logger.info("🌱 ENTRENANDO ENSEMBLE MULTISEMILLA (con reanudación automática)")
    logger.info(f"🏷️ Experimento: {nombre_experimento}")
    logger.info(f"🌱 Semillas: {semillas}")
    logger.info(f"{'=' * 60}")

    for i, seed in enumerate(semillas, 1):
        logger.info(f"\n🌱 Semilla {seed} ({i}/{len(semillas)})")

        filename = f"{nombre_experimento}_seed{seed}_final.txt"
        filepath = os.path.join(MODELOS_PATH, filename)

        # --- 🔁 Si el modelo ya existe, lo cargamos directamente ---
        if os.path.exists(filepath):
            logger.info(f"🔁 Modelo ya encontrado en disco. Cargando desde: {filepath}")
            model_final = lgb.Booster(model_file=filepath)

            # Siempre calculamos probabilidades (las necesitás para el ensemble)
            y_pred_valid = model_final.predict(X_valid)
            y_pred_test = model_final.predict(X_test)

            # Pero el UMBRAL lo leemos de disco si está guardado
            metricas = _cargar_metricas_seed(nombre_experimento, seed)

            if metricas is not None:
                umbral = metricas["umbral"]
                ganancia = metricas["ganancia"]
                # N_opt desde JSON, o lo derivamos por las dudas
                N_opt = metricas.get("N_opt", int((y_pred_valid >= umbral).sum()))

                logger.info(
                    f"✅ Métricas cargadas de disco → Umbral={umbral:.6f}, "
                    f"N={N_opt}, Ganancia=${ganancia:,.0f}"
                )
            else:
                # Fallback para modelos viejos (sin JSON de métricas)
                logger.warning(
                    "⚠️ No se encontró archivo de métricas para esta semilla. "
                    "Recalculando umbral sobre validación..."
                )
                umbral, N_opt, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)

            umbrales_individuales.append(umbral)
            ganancias_individuales.append(ganancia)
            probabilidades_valid.append(y_pred_valid)
            probabilidades_test.append(y_pred_test)
            modelos_finales.append(model_final)

            continue  # pasa a la siguiente semilla

        # --- 🚀 Si no existe, entrenamos normalmente ---
        logger.info("⏳ Entrenando modelo nuevo...")

        # FASE 1: Entrenar con datos iniciales y evaluar en validación externa
        model_valid = entrenar_modelo_single_seed(
            X_train_inicial, y_train_inicial, w_train_inicial, params, num_boost_round, seed
        )
        y_pred_valid = model_valid.predict(X_valid)
        umbral, N_opt, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)

        # Guardamos métricas de esta semilla (solo se calcula una vez)
        _guardar_metricas_seed(nombre_experimento, seed, umbral, N_opt, ganancia)

        umbrales_individuales.append(umbral)
        ganancias_individuales.append(ganancia)
        probabilidades_valid.append(y_pred_valid)

        logger.info(
            f"📊 Umbral validación (semilla {seed}): {umbral:.6f}, "
            f"N={N_opt}, Ganancia=${ganancia:,.0f}"
        )

        # FASE 2: Reentrenar con datos completos
        model_final = entrenar_modelo_single_seed(
            X_train_completo, y_train_completo, w_train_completo, params, num_boost_round, seed
        )
        y_pred_test = model_final.predict(X_test)
        probabilidades_test.append(y_pred_test)
        modelos_finales.append(model_final)

        if guardar_modelos:
            model_final.save_model(filepath)
            logger.info(f"💾 Modelo final guardado en: {filepath}")

    logger.info(f"\n{'=' * 60}")
    logger.info("✅ ENSEMBLE MULTISEMILLA COMPLETADO")
    logger.info(f"🏷️ Experimento: {nombre_experimento}")
    logger.info(f"{'=' * 60}")

    return {
        "probabilidades_valid": probabilidades_valid,
        "probabilidades_test": probabilidades_test,
        "umbrales_individuales": umbrales_individuales,
        "ganancias_individuales": ganancias_individuales,
        "modelos_finales": modelos_finales,
    }


# ============================================================
# Funciones de ensemble y evaluación
# ============================================================

def crear_ensemble_predictions(probabilidades_list):
    """
    Crea predicciones ensemble promediando probabilidades.
    """
    matriz = np.array(probabilidades_list)
    ensemble = np.mean(matriz, axis=0)

    logger.info(f"📊 Ensemble creado: shape={matriz.shape}")
    logger.info(
        f"   Min={ensemble.min():.6f}, Max={ensemble.max():.6f}, "
        f"Mean={ensemble.mean():.6f}"
    )

    return ensemble


def evaluar_ensemble_y_umbral(
    probabilidades_valid,
    probabilidades_test,
    w_valid,
    umbrales_individuales,
):
    """
    Evalúa el ensemble multisemilla:
      - Promedia predicciones en validación para encontrar umbral óptimo.
      - Compara con el promedio de umbrales individuales.
      - Aplica el umbral óptimo al ensemble en test final.
    """
    # Ensemble en validación
    matriz_valid = np.array(probabilidades_valid)
    probabilidades_valid_ensemble = np.mean(matriz_valid, axis=0)

    logger.info(f"\n{'=' * 60}")
    logger.info("🎯 CREANDO ENSEMBLE EN VALIDACIÓN Y OPTIMIZANDO UMBRAL")
    logger.info(f"{'=' * 60}")
    logger.info(f"📊 Ensemble validación: shape={matriz_valid.shape}")

    umbral_ensemble, N_ensemble, ganancia_ensemble, curva_ensemble = mejor_umbral_probabilidad(
        probabilidades_valid_ensemble,
        w_valid,
    )

    logger.info(f"✅ UMBRAL ÓPTIMO DEL ENSEMBLE (valid): {umbral_ensemble:.6f}")
    logger.info(f"   N={N_ensemble}, Ganancia=${ganancia_ensemble:,.0f}")

    # Comparar con umbrales individuales
    umbral_promedio_individual = np.mean(umbrales_individuales)
    logger.info(f"   Umbral promedio individual: {umbral_promedio_individual:.6f}")
    logger.info(f"   Desv. std umbrales:          {np.std(umbrales_individuales):.6f}")

    # Ensemble en test
    matriz_test = np.array(probabilidades_test)
    probabilidades_test_ensemble = np.mean(matriz_test, axis=0)

    logger.info(f"\n{'=' * 60}")
    logger.info("🚀 APLICANDO UMBRAL AL ENSEMBLE EN TEST FINAL")
    logger.info(f"{'=' * 60}")
    logger.info(f"📊 Ensemble test final: shape={matriz_test.shape}")

    prediccion_final_binaria = (probabilidades_test_ensemble >= umbral_ensemble).astype(int)
    N_enviados_final = prediccion_final_binaria.sum()

    logger.info("✅ PREDICCIÓN FINAL CON ENSEMBLE")
    logger.info(f"   🎯 Umbral usado:          {umbral_ensemble:.6f}")
    logger.info(f"   📮 Clientes marcados:     {N_enviados_final:,}")
    logger.info(
        f"   📊 Proporción positivos: "
        f"{N_enviados_final / len(prediccion_final_binaria) * 100:.2f}%"
    )

    return {
        "umbral_optimo_ensemble": umbral_ensemble,
        "N_en_umbral": N_ensemble,
        "ganancia_maxima_valid": ganancia_ensemble,
        "umbral_promedio_individual": umbral_promedio_individual,
        "probabilidades_valid_ensemble": probabilidades_valid_ensemble,
        "probabilidades_test_ensemble": probabilidades_test_ensemble,
        "prediccion_binaria": prediccion_final_binaria,
        "N_enviados": N_enviados_final,
        "curva_ganancia": curva_ensemble,
    }

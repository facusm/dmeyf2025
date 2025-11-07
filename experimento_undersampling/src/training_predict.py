# src/training_predict.py

import lightgbm as lgb
import numpy as np
import os
from config.config import MODELOS_PATH, SEMILLAS, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


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
        num_boost_round=num_boost_round
    )

    logger.info(f"✅ Modelo entrenado con semilla {seed}")

    return model


def entrenar_ensemble_multisemilla(X_train_inicial, y_train_inicial, w_train_inicial,
                                   X_train_completo, y_train_completo, w_train_completo,
                                   X_valid, w_valid,
                                   X_test,
                                   params, num_boost_round,
                                   semillas=None,
                                   guardar_modelos=True,
                                   nombre_experimento=NOMBRE_EXPERIMENTO):
    """
    Entrena un ensemble de modelos con múltiples semillas.

    Fase 1:
        - Entrena con X_train_inicial (histórico hasta cierto corte).
        - Predice sobre X_valid (mes de validación externa).
        - Usa esas predicciones para calcular umbrales por semilla.

    Fase 2:
        - Re-entrena con X_train_completo (histórico + validaciones).
        - Predice sobre X_test (meses de test final).
    """
    semillas = semillas or SEMILLAS

    probabilidades_valid = []
    probabilidades_test = []
    umbrales_individuales = []
    ganancias_individuales = []
    modelos_finales = []

    logger.info(f"\n{'='*60}")
    logger.info(f"🌱 ENTRENANDO ENSEMBLE CON {len(semillas)} SEMILLAS")
    logger.info(f"{'='*60}")

    for i, seed in enumerate(semillas, 1):
        logger.info(f"\n🌱 Semilla {seed} ({i}/{len(semillas)})")

        # --- FASE 1: Entrenar con datos iniciales y predecir en validación ---
        model_valid = entrenar_modelo_single_seed(
            X_train_inicial, y_train_inicial, w_train_inicial,
            params, num_boost_round, seed
        )

        y_pred_valid = model_valid.predict(X_valid)
        probabilidades_valid.append(y_pred_valid)

        # Calcular umbral y ganancia individual en validación
        umbral, N_opt, ganancia, _ = mejor_umbral_probabilidad(y_pred_valid, w_valid)
        umbrales_individuales.append(umbral)
        ganancias_individuales.append(ganancia)

        logger.info(f"   📊 Umbral (valid): {umbral:.6f}, N={N_opt}, Ganancia=${ganancia:,.0f}")

        # --- FASE 2: Re-entrenar con datos completos y predecir en test final ---
        model_final = entrenar_modelo_single_seed(
            X_train_completo, y_train_completo, w_train_completo,
            params, num_boost_round, seed
        )

        y_pred_test = model_final.predict(X_test)
        probabilidades_test.append(y_pred_test)
        modelos_finales.append(model_final)

        # Guardar modelo final si está habilitado
        if guardar_modelos:
            filename = f"{nombre_experimento}_seed{seed}_final.txt"
            filepath = os.path.join(MODELOS_PATH, filename)
            model_final.save_model(filepath)
            logger.info(f"   💾 Modelo guardado: {filepath}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ENSEMBLE COMPLETADO")
    logger.info(f"{'='*60}")

    return {
        "probabilidades_valid": probabilidades_valid,
        "probabilidades_test": probabilidades_test,
        "umbrales_individuales": umbrales_individuales,
        "ganancias_individuales": ganancias_individuales,
        "modelos_finales": modelos_finales
    }


def crear_ensemble_predictions(probabilidades_list):
    """
    Crea predicciones ensemble promediando probabilidades.
    """
    matriz = np.array(probabilidades_list)
    ensemble = np.mean(matriz, axis=0)

    logger.info(f"📊 Ensemble creado: shape={matriz.shape}")
    logger.info(f"   Min={ensemble.min():.6f}, Max={ensemble.max():.6f}, Mean={ensemble.mean():.6f}")

    return ensemble


def evaluar_ensemble_y_umbral(probabilidades_valid, probabilidades_test,
                              w_valid, umbrales_individuales):
    """
    Evalúa el ensemble multisemilla:
      - Promedia predicciones en validación para encontrar umbral óptimo.
      - Aplica ese umbral al ensemble en test final.
    """
    # Promediar predicciones en validación
    matriz_valid = np.array(probabilidades_valid)
    probabilidades_valid_ensemble = np.mean(matriz_valid, axis=0)

    logger.info(f"\n{'='*60}")
    logger.info("🎯 CREANDO ENSEMBLE EN VALIDACIÓN Y OPTIMIZANDO UMBRAL")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Ensemble validación: shape={matriz_valid.shape}")

    # Encontrar umbral óptimo del ensemble en validación
    umbral_ensemble, N_ensemble, ganancia_ensemble, curva_ensemble = mejor_umbral_probabilidad(
        probabilidades_valid_ensemble,
        w_valid
    )

    logger.info(f"✅ UMBRAL ÓPTIMO DEL ENSEMBLE (valid): {umbral_ensemble:.6f}")
    logger.info(f"   N={N_ensemble}, Ganancia=${ganancia_ensemble:,.0f}")

    # Comparar con umbral promedio individual
    umbral_promedio_individual = np.mean(umbrales_individuales)
    logger.info(f"   Umbral promedio individual={umbral_promedio_individual:.6f}")
    logger.info(f"   Desv. std umbrales={np.std(umbrales_individuales):.6f}")

    # Promediar predicciones en test final
    matriz_test = np.array(probabilidades_test)
    probabilidades_test_ensemble = np.mean(matriz_test, axis=0)

    logger.info(f"\n{'='*60}")
    logger.info("🚀 APLICANDO UMBRAL AL ENSEMBLE EN TEST FINAL")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Ensemble test final: shape={matriz_test.shape}")

    # Aplicar umbral óptimo encontrado en validación al test final
    prediccion_final_binaria = (probabilidades_test_ensemble >= umbral_ensemble).astype(int)
    N_enviados_final = prediccion_final_binaria.sum()

    logger.info(f"✅ PREDICCIÓN FINAL CON ENSEMBLE")
    logger.info(f"   🎯 Umbral usado: {umbral_ensemble:.6f}")
    logger.info(f"   📮 Clientes marcados: {N_enviados_final:,}")
    logger.info(f"   📊 Proporción de positivos: {N_enviados_final / len(prediccion_final_binaria) * 100:.2f}%")

    return {
        "umbral_optimo_ensemble": umbral_ensemble,
        "N_en_umbral": N_ensemble,
        "ganancia_maxima_valid": ganancia_ensemble,
        "umbral_promedio_individual": umbral_promedio_individual,
        "probabilidades_valid_ensemble": probabilidades_valid_ensemble,
        "probabilidades_test_ensemble": probabilidades_test_ensemble,
        "prediccion_binaria": prediccion_final_binaria,
        "N_enviados": N_enviados_final,
        "curva_ganancia": curva_ensemble
    }

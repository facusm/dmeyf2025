# main.py

import os
import logging
from datetime import datetime

from config.config import (
    PARAMS,
    SEMILLAS,
    SUFIJO_FE,
    MES_TEST_FINAL,
    FE_PATH,
    LOGS_PATH,
    NOMBRE_EXPERIMENTO,
)
from src.data_load_preparation import (
    cargar_datos,
    preparar_clases_y_pesos,
    preparar_train_optuna,
    preparar_validacion_optuna,
    preparar_validacion,
    preparar_test_final,
    preparar_train_completo,
)
from src.optuna_optimization import ejecutar_optimizacion
from src.training_predict import (
    entrenar_ensemble_multisemilla,
    evaluar_ensemble_y_umbral,
)
from src.resultados_ensemble import generar_reporte_ensemble
from src.utils import logger


# =========================
# CONFIGURACIÓN DEL LOGGER
# =========================
def setup_logger():
    os.makedirs(LOGS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_PATH, f"main_{timestamp}.log")

    logger.setLevel(logging.INFO)

    # Evitar agregar múltiples handlers si se llama varias veces
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # Consola
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)

        # Archivo
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    logger.info(f"📂 Logging iniciado. Archivo: {log_path}")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"🏷️ Versión FE: {SUFIJO_FE}")
    return logger


# ==============
# PIPELINE MAIN
# ==============
def main():
    setup_logger()
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🚀 INICIO PIPELINE")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"🏷️ FE utilizado: {SUFIJO_FE}")
    logger.info(f"{'=' * 80}\n")

    # 1️⃣ Carga de datos (siempre desde el FE_PATH definido en config)
    path_input = FE_PATH
    logger.info(f"📥 Cargando dataset FE desde: {path_input}")
    data = cargar_datos(path_input)
    data = preparar_clases_y_pesos(data)

    # 2️⃣ Armado de splits
    logger.info("🧩 Preparando datasets...")

    # Train base para Optuna (MESES_TRAIN) con undersampling (si está activado)
    X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_optuna(data)

    # Validación interna Optuna (MES_VAL_OPTUNA), sin undersampling
    X_valid_optuna, y_valid_optuna, w_valid_optuna = preparar_validacion_optuna(data)

    # Validación externa para umbral (MES_VALID), sin undersampling
    X_valid, y_valid, w_valid = preparar_validacion(data)

    # Test final (MES_TEST_FINAL), sin undersampling
    X_test, clientes_test = preparar_test_final(data)

    # Train inicial para ensemble:
    #   = train_optuna + valid_optuna (histórico hasta el corte de valid interna)
    X_train_inicial, y_train_inicial, w_train_inicial = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=None,
    )

    # Train completo final:
    #   = train_optuna + valid_optuna + valid_externa (incluye mes usado para umbral)
    X_train_completo, y_train_completo, w_train_completo = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=(X_valid, y_valid, w_valid),
    )

    # 3️⃣ Optimización de hiperparámetros con Optuna (validación temporal + multisemilla)
    logger.info("\n🎯 Iniciando optimización con Optuna (validación temporal + multisemilla)...")
    study = ejecutar_optimizacion(
        X_train_optuna,
        y_train_optuna,
        w_train_optuna,
        X_valid_optuna,
        y_valid_optuna,
        w_valid_optuna,
        semilleros=SEMILLAS,
        seed=SEMILLAS[0],
    )

    best_params = study.best_params
    best_iter = study.best_trial.user_attrs.get(
        "best_iter",
        PARAMS.get("num_boost_round", 1000),
    )

    logger.info(f"✅ Mejor trial #{study.best_trial.number} con ganancia {study.best_value:,.0f}")
    logger.info(f"   Parámetros óptimos: {best_params}")
    logger.info(f"   Iteraciones óptimas (promedio): {best_iter}")

    # 4️⃣ Entrenamiento ensemble multisemilla
    logger.info("\n🌱 Entrenando ensemble multisemilla...")
    ensemble_result = entrenar_ensemble_multisemilla(
        X_train_inicial, y_train_inicial, w_train_inicial,
        X_train_completo, y_train_completo, w_train_completo,
        X_valid, w_valid,
        X_test,
        params={**best_params, "objective": "binary", "metric": "None"},
        num_boost_round=best_iter,
        semillas=SEMILLAS,
        guardar_modelos=True,
    )

    # 5️⃣ Evaluación del ensemble y determinación del umbral óptimo
    logger.info("\n📈 Evaluando ensemble y determinando umbral óptimo...")
    eval_result = evaluar_ensemble_y_umbral(
        ensemble_result["probabilidades_valid"],
        ensemble_result["probabilidades_test"],
        w_valid,
        ensemble_result["umbrales_individuales"],
    )

    # 6️⃣ Generación del archivo final de submission
    logger.info("\n📦 Generando submission final...")

    generar_reporte_ensemble(
        test_data=data[data["foto_mes"].isin(MES_TEST_FINAL)],
        prediccion_final_binaria=eval_result["prediccion_binaria"],
        probabilidades_test_ensemble=eval_result["probabilidades_test_ensemble"],
        umbrales_individuales=ensemble_result["umbrales_individuales"],
        umbral_promedio_individual=eval_result["umbral_promedio_individual"],
        umbral_ensemble=eval_result["umbral_optimo_ensemble"],
        umbral_aplicado_test=eval_result["umbral_optimo_ensemble"],
        ganancia_ensemble=eval_result["ganancia_maxima_valid"],
        N_ensemble=eval_result["N_en_umbral"],
        semillas=SEMILLAS,
        N_enviados_final=eval_result["N_enviados"],
        nombre_modelo="ensemble_lgbm",
        trial_number=study.best_trial.number,
    )

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

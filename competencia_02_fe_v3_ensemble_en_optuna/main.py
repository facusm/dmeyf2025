# main.py

import os
import logging
from datetime import datetime

from config.config import (
    PARAMS,
    SEMILLAS_OPTUNA,
    SEMILLAS_ENSEMBLE,
    SUFIJO_FE,
    MES_TEST_FINAL,
    FE_PATH,
    LOGS_PATH,
    NOMBRE_EXPERIMENTO,
    EXPERIMENT_DIR
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

    # 🔧 Evitar duplicados
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

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

    X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_optuna(data)
    X_valid_optuna, y_valid_optuna, w_valid_optuna = preparar_validacion_optuna(data)
    X_valid, y_valid, w_valid = preparar_validacion(data)
    X_test, clientes_test = preparar_test_final(data)

    X_train_inicial, y_train_inicial, w_train_inicial = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=None,
    )

    X_train_completo, y_train_completo, w_train_completo = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=(X_valid, y_valid, w_valid),
    )

    # 3️⃣ Optimización de hiperparámetros con Optuna
    logger.info("\n🎯 Iniciando optimización con Optuna (validación temporal + multisemilla)...")
    study = ejecutar_optimizacion(
        X_train_optuna,
        y_train_optuna,
        w_train_optuna,
        X_valid_optuna,
        y_valid_optuna,
        w_valid_optuna,
        semilleros=SEMILLAS_OPTUNA,
        seed=SEMILLAS_OPTUNA[0],
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
        semillas=SEMILLAS_ENSEMBLE,
        guardar_modelos=True,
    )

    # 5️⃣ Evaluación del ensemble
    logger.info("\n📈 Evaluando ensemble y determinando umbral óptimo...")
    eval_result = evaluar_ensemble_y_umbral(
        ensemble_result["probabilidades_valid"],
        ensemble_result["probabilidades_test"],
        w_valid,
        ensemble_result["umbrales_individuales"],
    )

    # 5️⃣.1️⃣ Reporte adicional: Ganancia estimada del ensemble
    gan_ens = eval_result["ganancia_maxima_valid"]
    N_opt_ens = eval_result["N_en_umbral"]
    umbral_ens = eval_result["umbral_optimo_ensemble"]

    logger.info("\n🎯 RESULTADOS DEL ENSEMBLE (VALIDACIÓN EXTERNA 202106)")
    logger.info(f"   ✨ Ganancia óptima del ensemble: ${gan_ens:,.0f}")
    logger.info(f"   📮 N óptimo de envíos: {N_opt_ens:,}")
    logger.info(f"   🔪 Umbral óptimo del ensemble: {umbral_ens:.6f}\n")

    # 5️⃣.2️⃣ Guardado de eval_result en JSON para reproducibilidad
    import json

    eval_result_path = os.path.join(EXPERIMENT_DIR, "eval_result.json")

    # Convertir objetos no serializables (numpy arrays → listas)
    eval_result_serializable = {
        key: (
            value.tolist() if hasattr(value, "tolist") else value
        )
        for key, value in eval_result.items()
    }

    with open(eval_result_path, "w") as f:
        json.dump(eval_result_serializable, f, indent=4)

    logger.info(f"💾 eval_result guardado en: {eval_result_path}")



    # 6️⃣ GENERACIÓN DEL ARCHIVO FINAL USANDO generar_reporte_ensemble
    logger.info("\n📦 Generando submission final del ensemble...")

    # Extraemos el conjunto de test (202108)
    data_test = data[data["foto_mes"].isin(MES_TEST_FINAL)]

    # Creamos máscara para 202108
    mask_test = data["foto_mes"].isin(MES_TEST_FINAL)

    # Filtrar predicciones SOLO para el test final
    pred_final = eval_result["prediccion_binaria"][mask_test.values]
    prob_final = eval_result["probabilidades_test_ensemble"][mask_test.values]

    # ===== VALIDACIÓN DE ALINEACIÓN =====
    assert len(data_test) == len(pred_final), \
        f"ERROR: data_test tiene {len(data_test)} filas pero pred_final tiene {len(pred_final)}."

    assert all(data_test.index == data.index[mask_test]), \
        "ERROR: El orden de data_test NO coincide con el orden de las predicciones."

    logger.info("Validación de alineación correcta → listas finales alineadas 1 a 1")

    # Guardar CSV final y reporte completo del ensemble
    submission_path = generar_reporte_ensemble(
        test_data=data_test,
        prediccion_final_binaria=pred_final,
        probabilidades_test_ensemble=prob_final,
        umbrales_individuales=ensemble_result["umbrales_individuales"],
        umbral_promedio_individual=eval_result["umbral_promedio_individual"],
        umbral_ensemble=eval_result["umbral_optimo_ensemble"],
        umbral_aplicado_test=eval_result["umbral_optimo_ensemble"],
        ganancia_ensemble=eval_result["ganancia_maxima_valid"],
        N_ensemble=eval_result["N_en_umbral"],
        semillas=SEMILLAS_ENSEMBLE,
        N_enviados_final=(pred_final == 1).sum(),
        nombre_modelo=f"ensemble_lgbm_{MES_TEST_FINAL[0]}",
        trial_number=study.best_trial.number,
    )

    logger.info(f"📄 Submission final guardado en: {submission_path}")


    logger.info(f"\n{'=' * 80}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

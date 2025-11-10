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
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
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

    # 5️⃣ Evaluación del ensemble (sobre valid externa) y obtención de umbral óptimo global
    logger.info("\n📈 Evaluando ensemble y determinando umbral óptimo...")
    eval_result = evaluar_ensemble_y_umbral(
        ensemble_result["probabilidades_valid"],
        ensemble_result["probabilidades_test"],
        w_valid,
        ensemble_result["umbrales_individuales"],
    )

    # Extraemos predicciones y probabilidades del ensemble sobre el test completo
    prob_test_ensemble = eval_result["probabilidades_test_ensemble"]
    pred_test_binaria = eval_result["prediccion_binaria"]

    # 6️⃣ Generación de archivos finales de submission — uno por cada mes de test
    logger.info("\n📦 Generando submissions por mes de test...")

    # Subset de test alineado con las predicciones del ensemble
    data_test = data[data["foto_mes"].isin(MES_TEST_FINAL)].copy()

    for mes in MES_TEST_FINAL:
        logger.info(f"\n📅 Generando submission para mes de test: {mes}")

        # Máscara dentro del subset de test
        mask_mes = (data_test["foto_mes"] == mes)

        test_mes = data_test.loc[mask_mes]
        pred_mes = pred_test_binaria[mask_mes.values]
        prob_mes = prob_test_ensemble[mask_mes.values]

        N_enviados_mes = int((pred_mes == 1).sum())

        generar_reporte_ensemble(
            test_data=test_mes,
            prediccion_final_binaria=pred_mes,
            probabilidades_test_ensemble=prob_mes,
            umbrales_individuales=ensemble_result["umbrales_individuales"],
            umbral_promedio_individual=eval_result["umbral_promedio_individual"],
            umbral_ensemble=eval_result["umbral_optimo_ensemble"],
            umbral_aplicado_test=eval_result["umbral_optimo_ensemble"],
            ganancia_ensemble=eval_result["ganancia_maxima_valid"],
            N_ensemble=eval_result["N_en_umbral"],
            semillas=SEMILLAS,
            N_enviados_final=N_enviados_mes,
            nombre_modelo=f"ensemble_lgbm_{mes}",
            trial_number=study.best_trial.number,
        )

    # 7️⃣ Diagnóstico ex-post de ganancia por mes de test (usa labels reales si existen)
    logger.info(
        "\n🧪 Diagnóstico ex-post: ganancia por mes de test "
        "(usa clase_ternaria real, solo análisis fuera de competencia)..."
    )

    for mes in MES_TEST_FINAL:
        mask_mes = (data_test["foto_mes"] == mes)
        test_mes = data_test.loc[mask_mes]

        # Si no hay clase_ternaria (como en un test real), se salta el diagnóstico
        if "clase_ternaria" not in test_mes.columns:
            logger.info(f"   ⚠️ Mes {mes}: no hay clase_ternaria, se omite diagnóstico.")
            continue

        pred_mes = pred_test_binaria[mask_mes.values]
        is_envio = (pred_mes == 1)
        enviados = int(is_envio.sum())

        es_baja2 = (test_mes["clase_ternaria"] == "BAJA+2").values
        aciertos_baja2 = int((is_envio & es_baja2).sum())

        gan_mes = aciertos_baja2 * GANANCIA_ACIERTO + enviados * COSTO_ESTIMULO

        logger.info(
            f"   Mes {mes}: enviados={enviados:,}, "
            f"aciertos_BAJA2={aciertos_baja2:,}, "
            f"ganancia_ex_post=${gan_mes:,.0f}"
        )

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"{'=' * 80}\n")



if __name__ == "__main__":
    main()

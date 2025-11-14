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


# =====================================================================================
# CONFIGURACIÓN DEL LOGGER
# =====================================================================================
def setup_logger():
    os.makedirs(LOGS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_PATH, f"main_{timestamp}.log")

    # Reset handlers
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

    logger.info(f"📂 Logging iniciado → {log_path}")
    logger.info(f"🏷️ Experimento = {NOMBRE_EXPERIMENTO}")
    logger.info(f"🏷️ Versión FE = {SUFIJO_FE}")
    return logger


# =====================================================================================
# PIPELINE PRINCIPAL
# =====================================================================================
def main():
    setup_logger()
    logger.info(f"\n{'='*90}")
    logger.info("🚀 INICIO DEL PIPELINE")
    logger.info(f"{'='*90}\n")

    # -----------------------------------------------------------------------------
    # 1) Carga de dataset FE
    # -----------------------------------------------------------------------------
    logger.info(f"📥 Cargando dataset desde: {FE_PATH}")
    data = cargar_datos(FE_PATH)
    data = preparar_clases_y_pesos(data)

    # -----------------------------------------------------------------------------
    # 2) Splits
    # -----------------------------------------------------------------------------
    X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_optuna(data)
    X_valid_optuna, y_valid_optuna, w_valid_optuna = preparar_validacion_optuna(data)
    X_valid, y_valid, w_valid = preparar_validacion(data)
    X_test, clientes_test = preparar_test_final(data)

    # Train hasta abril (FASE 1)
    X_train_inicial, y_train_inicial, w_train_inicial = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=None,
    )

    # Train completo hasta junio (FASE 2)
    X_train_completo, y_train_completo, w_train_completo = preparar_train_completo(
        train_optuna=(X_train_optuna, y_train_optuna, w_train_optuna),
        valid_optuna=(X_valid_optuna, y_valid_optuna, w_valid_optuna),
        valid_externa=(X_valid, y_valid, w_valid),
    )

    # -----------------------------------------------------------------------------
    # 3) OPTUNA
    # -----------------------------------------------------------------------------
    logger.info("🎯 Optimizando hiperparámetros con Optuna…")
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
    best_iter = study.best_trial.user_attrs.get("best_iter", PARAMS.get("num_boost_round", 1000))

    logger.info(f"🏅 Mejor trial = #{study.best_trial.number}")
    logger.info(f"   → Ganancia = ${study.best_value:,.0f}")
    logger.info(f"   → Best params = {best_params}")
    logger.info(f"   → Best iter promedio = {best_iter}")

    # -----------------------------------------------------------------------------
    # 4) Entrenamiento ensemble multisemilla (FASE1 + FASE2 con checkpoints)
    # -----------------------------------------------------------------------------
    logger.info("\n🌱 Entrenando ensemble multisemilla…")

    ensemble_result = entrenar_ensemble_multisemilla(
        X_train_inicial, y_train_inicial, w_train_inicial,   # FASE 1 (hasta abril)
        X_train_completo, y_train_completo, w_train_completo, # FASE 2 (hasta junio)
        X_valid, w_valid,                                     # Valid externa (junio)
        X_test,                                               # Test (agosto)
        params={**best_params, "objective": "binary", "metric": "None"},
        num_boost_round=best_iter,
        semillas=SEMILLAS,
    )

    # -----------------------------------------------------------------------------
    # 5) Calcular umbral óptimo SOLO en junio
    # -----------------------------------------------------------------------------
    logger.info("\n📈 Calculando umbral óptimo del ensemble…")

    eval_result = evaluar_ensemble_y_umbral(
        ensemble_result["probabilidades_valid"],
        ensemble_result["probabilidades_test"],
        w_valid,
        ensemble_result["umbrales_individuales"],
    )

    prob_test_ensemble = eval_result["probabilidades_test_ensemble"]
    pred_test_binaria = eval_result["prediccion_binaria"]

    # -----------------------------------------------------------------------------
    # 6) Generar submissions por mes de test
    # -----------------------------------------------------------------------------
    data_test = data[data["foto_mes"].isin(MES_TEST_FINAL)].copy()

    logger.info("\n📦 Generando submissions finales…")

    for mes in MES_TEST_FINAL:
        logger.info(f"\n🗓️ Mes test: {mes}")

        mask = (data_test["foto_mes"] == mes)
        test_mes = data_test.loc[mask]

        pred_mes = pred_test_binaria[mask.values]
        prob_mes = prob_test_ensemble[mask.values]

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
            N_enviados_final=int((pred_mes == 1).sum()),
            nombre_modelo=f"ensemble_lgbm_{mes}",
            trial_number=study.best_trial.number,
        )

    logger.info(f"\n{'='*90}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"{'='*90}\n")


if __name__ == "__main__":
    main()

import os
import logging
from datetime import datetime

# === IMPORTS INTERNOS === #
from config.config import (
    PARAMS,
    SEMILLAS,
    SUFIJO_FE,
    MES_TEST_FINAL,
    BUCKET_PATH_b1,   
    FILE_BASE,
    VERSION,
    LOGS_PATH        
)
from src.data_load_preparation import (
    cargar_datos,
    preparar_clases_y_pesos,
    preparar_train_optuna,
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


# === CONFIGURACIÓN DE LOGGING GLOBAL === #
def setup_logger():
    """Configura el logger para guardar logs dentro del bucket (b1/logs/)"""
    os.makedirs(LOGS_PATH, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_PATH, f"main_{timestamp}.log")

    logger.setLevel(logging.INFO)

    # Evitar duplicados si se ejecuta varias veces
    if not logger.handlers:
        # Consola
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)

        # Archivo dentro del bucket
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    logger.info(f"📂 Logging iniciado. Archivo: {log_path}")
    return logger  # ✅ devolvemos el logger configurado


# === MAIN PIPELINE === #
def main():
    setup_logger()
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 INICIO DEL PIPELINE PRINCIPAL - FE: {SUFIJO_FE}")
    logger.info(f"{'='*80}\n")

    # --- 1️⃣ CARGA Y PREPARACIÓN DE DATOS --- #
    logger.info("📥 Cargando dataset procesado...")
    path_input = os.path.join(BUCKET_PATH_b1, f"{FILE_BASE}_FE_{VERSION}.parquet")  # ✅ formato Parquet (más eficiente)
    data = cargar_datos(path_input)

    logger.info(f"✅ Dataset cargado correctamente. Shape: {data.shape[0]:,} filas × {data.shape[1]:,} columnas")

    data = preparar_clases_y_pesos(data)


    # --- 2️⃣ DIVISIÓN EN TRAIN/VALID/TEST --- #
    logger.info("🧩 Preparando datasets...")
    X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_optuna(data)
    X_valid, y_valid, w_valid = preparar_validacion(data)
    X_test, clientes_test = preparar_test_final(data)
    X_train_completo, y_train_completo, w_train_completo = preparar_train_completo(
        (X_train_optuna, y_train_optuna, w_train_optuna),
        X_valid, y_valid, w_valid
    )

    # --- 3️⃣ OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA --- #
    logger.info("\n🎯 Iniciando optimización bayesiana con Optuna...")
    study = ejecutar_optimizacion(
        X_train_optuna, y_train_optuna, w_train_optuna,
        seed=SEMILLAS[0]
    )
    best_params = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iter", PARAMS["num_boost_round"])
    logger.info(f"✅ Mejor trial #{study.best_trial.number} con ganancia {study.best_value:,.0f}")
    logger.info(f"   Parámetros óptimos: {best_params}")
    logger.info(f"   Iteraciones óptimas: {best_iter}")

    # --- 4️⃣ ENTRENAMIENTO MULTISEMILLA Y ENSEMBLE --- #
    logger.info("\n🌱 Entrenando ensemble multisemilla...")
    ensemble_result = entrenar_ensemble_multisemilla(
        X_train_optuna, y_train_optuna, w_train_optuna,
        X_train_completo, y_train_completo, w_train_completo,
        X_valid, w_valid,
        X_test,
        params={**best_params, "objective": "binary", "metric": "None"},
        num_boost_round=best_iter,
        semillas=SEMILLAS,
        guardar_modelos=True
    )

    # --- 5️⃣ EVALUACIÓN Y GENERACIÓN DE ENSEMBLE FINAL --- #
    logger.info("\n📈 Evaluando ensemble y determinando umbral óptimo...")
    eval_result = evaluar_ensemble_y_umbral(
        ensemble_result['probabilidades_abril'],
        ensemble_result['probabilidades_junio'],
        w_valid,
        ensemble_result['umbrales_individuales']
    )

    # --- 6️⃣ GENERACIÓN DEL ARCHIVO FINAL --- #
    logger.info("\n📦 Generando submission final...")
    generar_reporte_ensemble(
        test_data=data[data['foto_mes'].isin(MES_TEST_FINAL)],
        prediccion_final_binaria=eval_result['prediccion_binaria'],
        probabilidades_junio_ensemble=eval_result['probabilidades_junio_ensemble'],
        umbrales_individuales=ensemble_result['umbrales_individuales'],
        umbral_promedio_individual=eval_result['umbral_promedio_individual'],
        umbral_ensemble=eval_result['umbral_optimo_ensemble'],
        umbral_junio=eval_result['umbral_optimo_ensemble'],
        ganancia_ensemble=eval_result['ganancia_maxima_abril'],
        N_ensemble=eval_result['N_en_umbral'],
        semillas=SEMILLAS,
        N_enviados_final=eval_result['N_enviados'],
        nombre_modelo="ensemble_lgbm",
        trial_number=study.best_trial.number
    )  # ✅ se elimina output_dir

    logger.info(f"\n{'='*80}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

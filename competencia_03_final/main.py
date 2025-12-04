# main.py

import pandas as pd
import os
import logging
from datetime import datetime

from config.config import (
    SEMILLAS_OPTUNA,
    SEMILLAS_ENSEMBLE,
    SUFIJO_FE,
    FE_PATH,
    LOGS_PATH,
    NOMBRE_EXPERIMENTO,
    EXPERIMENT_DIR,
    LGBM_PARAMS_BASE,
    MESES_TRAIN_OPTUNA,
    MES_VAL_OPTUNA,
    MESES_TRAIN_PARA_VAL_EXT,
    MES_VALID_EXT,
    MESES_TRAIN_COMPLETO_PARA_TEST_FINAL,
    MES_TEST_FINAL,
    MODEL_DIR_VAL_EXT,
    MODEL_DIR_TEST_FINAL,
    RESULTADOS_PREDICCION_DIR,
    APO_CORTES_ENVIO,
    SEMILLAS_APO,
    APO_K_SEM,
    APO_N_APO,
    RATIO_UNDERSAMPLING_VAL_EXT,
    RATIO_UNDERSAMPLING_TEST_FINAL,
)

from src.data_load_preparation import (
    cargar_datos,
    preparar_clases_y_pesos,
    preparar_train_meses,
    preparar_validacion_meses,
    preparar_test_final_meses
    
)

from src.apo_validacion_externa import (
    entrenar_modelos_val_externa,
    seleccionar_N_optimo_APO,
)


from src.optuna_optimization import ejecutar_optimizacion
from src.training_predict import (
    entrenar_ensemble_test_final
)

from src.utils import logger, rescalar_hp_apostyle


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
    data = preparar_clases_y_pesos(data) # Crea las columnas 'clase_binaria1', 'clase_binaria2' y 'clase_peso' en el dataframe data

    # 2️⃣ Armado de splits
    logger.info("🧩 Preparando datasets...")

    # Splits de train y validación para Optuna
    X_train_optuna, y_train_optuna, w_train_optuna = preparar_train_meses(data, MESES_TRAIN_OPTUNA, nombre_split = "Train Optuna")
    X_valid_optuna, y_valid_optuna, w_valid_optuna = preparar_validacion_meses(data, MES_VAL_OPTUNA, nombre_split = "Val Optuna")

    # Splits de train inicial para validación externa y set de validación externa 
    X_train_inicial, y_train_inicial, w_train_inicial = preparar_train_meses(data, MESES_TRAIN_PARA_VAL_EXT, nombre_split = "Train Val Ext", ratio=RATIO_UNDERSAMPLING_VAL_EXT)
    X_valid, y_valid, w_valid = preparar_validacion_meses(data, MES_VALID_EXT, nombre_split = "Val Ext")

    # Split de train completo para test final + set de test final
    X_train_completo, y_train_completo, w_train_completo = preparar_train_meses(data, MESES_TRAIN_COMPLETO_PARA_TEST_FINAL, nombre_split = "Train Completo para Test Final", ratio=RATIO_UNDERSAMPLING_TEST_FINAL)
    X_test, clientes_test = preparar_test_final_meses(data, MES_TEST_FINAL, nombre_split = f"Test Final {MES_TEST_FINAL[0]}")


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

    best_trial = study.best_trial

    # ✅ 1) Params entrenables de LGBM: vienen guardados en user_attrs
    lgbm_params_final = best_trial.user_attrs.get("lgb_params")
    if lgbm_params_final is None:
        logger.warning(
            "⚠️ best_trial.user_attrs['lgb_params'] no existe. "
            "Uso solo LGBM_PARAMS_BASE (no se puede reconstruir desde trial.params)."
        )
        lgbm_params_final = LGBM_PARAMS_BASE.copy()

    # Asegurar copia
    lgbm_params_final = dict(lgbm_params_final)

    # ✅ 2) Iteraciones óptimas (promedio)
    best_iter = best_trial.user_attrs.get("best_iter")
    if best_iter is None:
        raise KeyError("best_iter no está en user_attrs del best_trial (corriste un estudio viejo?).")
    best_iter = int(best_iter)

    # ✅ 3) Info extra
    N_opt_ensemble = best_trial.user_attrs.get("N_opt_ensemble")
    umbral_ensemble = best_trial.user_attrs.get("umbral_ensemble")

    logger.info(f"✅ Mejor trial #{best_trial.number} con ganancia {study.best_value:,.0f}")
    logger.info(f"📌 Params finales LightGBM (lgb_params): {lgbm_params_final}")
    logger.info(f"🧾 Optuna params crudos (trial.params): {best_trial.params}")
    logger.info(f"🔁 Iteraciones óptimas (best_iter): {best_iter}")

    if N_opt_ensemble is not None and umbral_ensemble is not None:
        logger.info(
            f"🎯 N óptimo ensemble (valid meseta): {N_opt_ensemble:,} | "
            f"umbral ensemble: {umbral_ensemble:.6f}"
        )


    # 4️⃣ Entrenamiento modelos para validación externa (APO sobre mes de validación externa)
    logger.info("\n Re escalando min_data_in_leaf para validación externa...")

    params_valext = rescalar_hp_apostyle(lgbm_params_final,
                                     n_old=len(X_train_optuna),
                                     n_new=len(X_train_inicial))
    
    logger.info("\n min_data_in_leaf para validación externa escalado...")
    
    logger.info("\n🌱 Entrenando modelos para validación externa (APO)...")
    entrenar_modelos_val_externa(
        X_train_inicial,
        y_train_inicial,
        w_train_inicial,
        params=params_valext,
        num_boost_round=best_iter,
        semillas=SEMILLAS_APO,
        model_dir=MODEL_DIR_VAL_EXT,
    )

    # 5️⃣ Selección de N óptimo vía APO usando 202107 como pseudo-futuro
    logger.info("\n📊 Seleccionando N óptimo vía APO sobre validación externa...")
    N_opt_APO, ganancias_prom_cortes, mganancias = seleccionar_N_optimo_APO(
        X_valid,
        w_valid,
        semillas=SEMILLAS_APO,
        cortes=APO_CORTES_ENVIO,
        model_dir=MODEL_DIR_VAL_EXT,
        ksem=APO_K_SEM,
        n_apo=APO_N_APO,
        num_boost_round=best_iter,
    )

    logger.info(f"🎯 N óptimo APO (valid_ext {MES_VALID_EXT[0]}): {N_opt_APO}")


    # 6️⃣ Entrenamiento FINAL (train completo) + predicción en MES_TEST_FINAL

    logger.info("\n Re escalando min_data_in_leaf para test final...")

    params_final = rescalar_hp_apostyle(lgbm_params_final,
                                    n_old=len(X_train_optuna),
                                    n_new=len(X_train_completo))
    
    logger.info("\n min_data_in_leaf para validación externa escalado...")

    logger.info("\n🌳 Entrenando ensemble FINAL multisemilla y prediciendo test...")

    ensemble_final = entrenar_ensemble_test_final(
        X_train=X_train_completo,
        y_train=y_train_completo,
        w_train=w_train_completo,
        X_test=X_test,
        params=params_final,
        num_boost_round=best_iter,
        semillas=SEMILLAS_ENSEMBLE,      # acá usás las seeds del ensemble final
        N_envios=N_opt_APO,              # viene de apo_validacion_externa
        guardar_modelos=True,
        model_dir=MODEL_DIR_TEST_FINAL,
    )

    prob_test_ensemble = ensemble_final["prob_test_ensemble"]
    pred_test_binaria = ensemble_final["pred_test_binaria"]
    N_envios_usado = ensemble_final["N_envios_usado"]

    logger.info(
        f"✅ Ensemble FINAL test listo. N_envios_usado={N_envios_usado} "
        f"sobre MES_TEST_FINAL={MES_TEST_FINAL}."
    )


    # 7️⃣ Generación del CSV final de envío (solo clientes con Predicted=1, sin header)
    os.makedirs(RESULTADOS_PREDICCION_DIR, exist_ok=True)

    df_envio = pd.DataFrame(
        {
            "numero_de_cliente": clientes_test,
            "Predicted": pred_test_binaria.astype(int),
        }
    )

    path_envio = os.path.join(
        RESULTADOS_PREDICCION_DIR,
        f"envio_{NOMBRE_EXPERIMENTO}_N{N_envios_usado}.csv",
    )

    # Solo clientes con Predicted = 1 y sin header, formato competencia
    df_envio.loc[df_envio["Predicted"] == 1, ["numero_de_cliente"]].to_csv(
        path_envio,
        index=False,
        header=False,
    )

    logger.info(f"📄 Archivo de envío generado: {path_envio}")


    logger.info(f"\n{'=' * 80}")
    logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()

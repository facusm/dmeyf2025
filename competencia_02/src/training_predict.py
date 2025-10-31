# src/training.py
import lightgbm as lgb
import numpy as np
import os
from config.config import MODELOS_PATH, SEMILLAS, NOMBRE_EXPERIMENTO
from src.utils import logger, mejor_umbral_probabilidad


def entrenar_modelo_single_seed(X_train, y_train, w_train, params, num_boost_round, seed):
    """
    Entrena un modelo LightGBM con una semilla específica.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento.
    y_train : pd.Series
        Target de entrenamiento.
    w_train : pd.Series
        Pesos de entrenamiento.
    params : dict
        Hiperparámetros del modelo.
    num_boost_round : int
        Número de iteraciones.
    seed : int
        Semilla para el modelo.
    
    Returns
    -------
    lgb.Booster
        Modelo entrenado.
    """
    params_seed = params.copy()
    params_seed['seed'] = seed
    
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
    
    Para cada semilla:
    1. Entrena con datos iniciales (meses_train) y predice en validación
    2. Re-entrena con datos completos (meses_train + mes_valid) y predice en test
    
    Parameters
    ----------
    X_train_inicial : pd.DataFrame
        Features de entrenamiento inicial (meses_train).
    y_train_inicial : pd.Series
        Target de entrenamiento inicial.
    w_train_inicial : pd.Series
        Pesos de entrenamiento inicial.
    X_train_completo : pd.DataFrame
        Features de entrenamiento completo (meses_train + mes_valid).
    y_train_completo : pd.Series
        Target de entrenamiento completo.
    w_train_completo : pd.Series
        Pesos de entrenamiento completo.
    X_valid : pd.DataFrame
        Features de validación.
    w_valid : pd.Series
        Pesos de validación.
    X_test : pd.DataFrame
        Features de test final.
    params : dict
        Hiperparámetros base del modelo.
    num_boost_round : int
        Número de iteraciones.
    semillas : list, optional
        Lista de semillas. Si no se provee, usa SEMILLAS de config.
    guardar_modelos : bool
        Si True, guarda los modelos en formato .txt.
    nombre_experimento : str
        Nombre para identificar los archivos guardados.
    
    Returns
    -------
    dict
        Diccionario con:
        - probabilidades_abril: lista de arrays de predicciones en validación
        - probabilidades_junio: lista de arrays de predicciones en test
        - umbrales_individuales: umbrales óptimos de cada semilla
        - ganancias_individuales: ganancias de cada semilla
        - modelos_finales: lista de modelos entrenados con datos completos
    """
    semillas = semillas or SEMILLAS
    
    probabilidades_abril = []
    probabilidades_junio = []
    umbrales_individuales = []
    ganancias_individuales = []
    modelos_finales = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🌱 ENTRENANDO ENSEMBLE CON {len(semillas)} SEMILLAS")
    logger.info(f"{'='*60}")
    
    for i, seed in enumerate(semillas, 1):
        logger.info(f"\n🌱 Semilla {seed} ({i}/{len(semillas)})")
        
        # --- FASE 1: Entrenar con datos iniciales y predecir en abril ---
        model_abril = entrenar_modelo_single_seed(
            X_train_inicial, y_train_inicial, w_train_inicial,
            params, num_boost_round, seed
        )
        
        y_pred_abril = model_abril.predict(X_valid)
        probabilidades_abril.append(y_pred_abril)
        
        # Calcular umbral y ganancia individual
        umbral, N_opt, ganancia, _ = mejor_umbral_probabilidad(y_pred_abril, w_valid)
        umbrales_individuales.append(umbral)
        ganancias_individuales.append(ganancia)
        
        logger.info(f"   📊 Umbral: {umbral:.6f}, N={N_opt}, Ganancia=${ganancia:,.0f}")
        
        # --- FASE 2: Re-entrenar con datos completos y predecir en junio ---
        model_final = entrenar_modelo_single_seed(
            X_train_completo, y_train_completo, w_train_completo,
            params, num_boost_round, seed
        )
        
        y_pred_junio = model_final.predict(X_test)
        probabilidades_junio.append(y_pred_junio)
        modelos_finales.append(model_final)
        
        # Guardar modelo final si está habilitado
        if guardar_modelos:
            os.makedirs(MODELOS_PATH, exist_ok=True)
            filename = f"{nombre_experimento}_seed{seed}_final.txt"
            filepath = os.path.join(MODELOS_PATH, filename)
            model_final.save_model(filepath)
            logger.info(f"   💾 Modelo guardado: {filename}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ ENSEMBLE COMPLETADO")
    logger.info(f"{'='*60}")
    
    return {
        'probabilidades_abril': probabilidades_abril,
        'probabilidades_junio': probabilidades_junio,
        'umbrales_individuales': umbrales_individuales,
        'ganancias_individuales': ganancias_individuales,
        'modelos_finales': modelos_finales
    }


def crear_ensemble_predictions(probabilidades_list):
    """
    Crea predicciones ensemble promediando probabilidades.
    
    Parameters
    ----------
    probabilidades_list : list
        Lista de arrays de probabilidades.
    
    Returns
    -------
    np.array
        Probabilidades promediadas.
    """
    matriz = np.array(probabilidades_list)
    ensemble = np.mean(matriz, axis=0)
    
    logger.info(f"📊 Ensemble creado: shape={matriz.shape}")
    logger.info(f"   Min={ensemble.min():.6f}, Max={ensemble.max():.6f}, Mean={ensemble.mean():.6f}")
    
    return ensemble


def evaluar_ensemble_y_umbral(probabilidades_abril, probabilidades_junio, 
                              w_valid, umbrales_individuales):
    """
    Evalúa el ensemble multisemilla, optimiza el umbral en validación
    y aplica ese umbral al conjunto final.
    """
    # Promediar predicciones de abril
    matriz_abril = np.array(probabilidades_abril)
    probabilidades_abril_ensemble = np.mean(matriz_abril, axis=0)

    logger.info(f"\n{'='*60}")
    logger.info("🎯 CREANDO ENSEMBLE DE ABRIL Y OPTIMIZANDO UMBRAL")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Ensemble abril: shape={matriz_abril.shape}")

    # Encontrar umbral óptimo del ensemble
    umbral_ensemble, N_ensemble, ganancia_ensemble, curva_ensemble = mejor_umbral_probabilidad(
        probabilidades_abril_ensemble,
        w_valid
    )

    logger.info(f"✅ UMBRAL ÓPTIMO DEL ENSEMBLE: {umbral_ensemble:.6f}")
    logger.info(f"   N={N_ensemble}, Ganancia=${ganancia_ensemble:,.0f}")

    # Comparar con umbral promedio individual
    umbral_promedio_individual = np.mean(umbrales_individuales)
    logger.info(f"   Umbral promedio individual={umbral_promedio_individual:.6f}")
    logger.info(f"   Desv. std umbrales={np.std(umbrales_individuales):.6f}")

    # Promediar predicciones de junio
    matriz_junio = np.array(probabilidades_junio)
    probabilidades_junio_ensemble = np.mean(matriz_junio, axis=0)

    logger.info(f"\n{'='*60}")
    logger.info("🚀 APLICANDO UMBRAL AL ENSEMBLE DE JUNIO")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Ensemble junio: shape={matriz_junio.shape}")

    # Aplicar umbral del ensemble de abril
    prediccion_final_binaria = (probabilidades_junio_ensemble >= umbral_ensemble).astype(int)
    N_enviados_final = prediccion_final_binaria.sum()

    logger.info(f"✅ PREDICCIÓN FINAL CON ENSEMBLE")
    logger.info(f"   🎯 Umbral usado: {umbral_ensemble:.6f}")
    logger.info(f"   📮 Clientes marcados: {N_enviados_final:,}")
    logger.info(f"   📊 Proporción de positivos: {N_enviados_final/len(prediccion_final_binaria)*100:.2f}%")

    return {
        'umbral_optimo_ensemble': umbral_ensemble,
        'N_en_umbral': N_ensemble,
        'ganancia_maxima_abril': ganancia_ensemble,
        'umbral_promedio_individual': umbral_promedio_individual,
        'probabilidades_abril_ensemble': probabilidades_abril_ensemble,
        'probabilidades_junio_ensemble': probabilidades_junio_ensemble,
        'prediccion_binaria': prediccion_final_binaria,
        'N_enviados': N_enviados_final,
        'curva_ganancia': curva_ensemble
    }
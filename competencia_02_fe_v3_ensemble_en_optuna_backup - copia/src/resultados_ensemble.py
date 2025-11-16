# src/resultados_ensemble.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

from config.config import RESULTADOS_PREDICCION_PATH, NOMBRE_EXPERIMENTO
from src.utils import logger  # usamos el logger global para consistencia


def generar_reporte_ensemble(
    test_data,
    prediccion_final_binaria,
    probabilidades_test_ensemble,
    umbrales_individuales,
    umbral_promedio_individual,
    umbral_ensemble,        # umbral óptimo encontrado en validación
    umbral_aplicado_test,   # umbral finalmente aplicado en test (puede ser igual al anterior)
    ganancia_ensemble,      # ganancia medida en validación
    N_ensemble,             # N óptimo en validación
    semillas,
    N_enviados_final,
    nombre_modelo="modelo",
    trial_number=None
):
    """
    Genera el archivo final del ensemble y un resumen detallado.

    - test_data: dataset de test final (ej: MES_TEST_FINAL).
    - prediccion_final_binaria: predicciones 0/1 usando el umbral aplicado sobre el ensemble en test.
    - probabilidades_test_ensemble: probabilidades promedio del ensemble en test.
    - umbral_ensemble: umbral óptimo hallado en el conjunto de validación.
    - umbral_aplicado_test: umbral usado efectivamente en test (puede ser igual o ajustado).
    - ganancia_ensemble: ganancia obtenida en validación con el umbral óptimo.
    - N_ensemble: N óptimo de enviados en validación.
    """

    # === Directorio de salida (ya específico del experimento desde config) ===
    output_dir = RESULTADOS_PREDICCION_PATH
    os.makedirs(output_dir, exist_ok=True)

        # === Generar CSV de submission (formato requerido) ===
    submission = pd.DataFrame({
        "numero_de_cliente": test_data["numero_de_cliente"].astype(int).values,
        "Predicted": np.asarray(prediccion_final_binaria, dtype=int)
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_filename = (
        f"{NOMBRE_EXPERIMENTO}_{nombre_modelo}_"
        f"T{trial_number or 'final'}_U{umbral_aplicado_test:.6f}_"
        f"N{N_enviados_final}_{timestamp}.csv"
    )
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)

    # === Logging ===
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("📦 GENERANDO SUBMISSION CON ENSEMBLE")
    logger.info(f"{sep}")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"📁 Carpeta resultados: {output_dir}")
    logger.info(f"📝 Archivo submission: {submission_filename}")
    logger.info(f"🎯 Umbral usado en test final: {umbral_aplicado_test:.6f}")
    logger.info(f"📮 Número de envíos (test final): {N_enviados_final:,}")

    logger.info(f"\n💾 Archivo guardado en: {submission_path}")
    distrib = submission['Predicted'].value_counts().to_dict()
    logger.info(f"📊 Distribución Predicted: {distrib}")
    logger.info(f"   - Clase 0 (no enviar): {(prediccion_final_binaria == 0).sum():,}")
    logger.info(f"   - Clase 1 (enviar):    {(prediccion_final_binaria == 1).sum():,}")
    logger.info(f"   - Total registros:     {len(prediccion_final_binaria):,}")

    # Muestra de las primeras filas
    logger.info("\n📋 Muestra del submission:")
    logger.info(f"\n{submission.head(10).to_string(index=False)}")

    # === Estadísticas completas del ensemble ===
    logger.info(f"\n{sep}")
    logger.info("📊 RESUMEN COMPLETO DEL ENSEMBLE")
    logger.info(f"{sep}")
    logger.info(f"🌱 Número de semillas usadas: {len(semillas)}")
    logger.info(f"   Semillas: {semillas}")

    logger.info("\n📐 Umbrales individuales (por semilla):")
    for i, (seed, umbral) in enumerate(zip(semillas, umbrales_individuales), 1):
        logger.info(f"   {i}. Seed {seed}: {umbral:.6f}")

    logger.info("\n🎯 Umbrales finales:")
    logger.info(f"   Promedio umbrales individuales: {umbral_promedio_individual:.6f}")
    logger.info(f"   Umbral óptimo en validación:    {umbral_ensemble:.6f}")
    logger.info(f"   Umbral aplicado en test final:  {umbral_aplicado_test:.6f}")
    if umbral_aplicado_test != umbral_ensemble:
        ajuste = ((umbral_aplicado_test - umbral_ensemble) / umbral_ensemble) * 100
        logger.info(f"   Ajuste aplicado vs validación: {ajuste:+.2f}%")
    logger.info(f"   Desviación estándar umbrales:   {np.std(umbrales_individuales):.6f}")

    logger.info(f"\n💰 Ganancia en validación: ${ganancia_ensemble:,.0f}")
    logger.info(f"   N óptimo en validación: {N_ensemble:,}")

    # === Probabilidades del ensemble en test final ===
    logger.info(f"\n📊 Probabilidades del ensemble en test final:")
    logger.info(f"   - Mínima:  {probabilidades_test_ensemble.min():.6f}")
    logger.info(f"   - Máxima:  {probabilidades_test_ensemble.max():.6f}")
    logger.info(f"   - Media:   {probabilidades_test_ensemble.mean():.6f}")
    logger.info(f"   - Mediana: {np.median(probabilidades_test_ensemble):.6f}")
    logger.info(f"   - Q1:      {np.percentile(probabilidades_test_ensemble, 25):.6f}")
    logger.info(f"   - Q3:      {np.percentile(probabilidades_test_ensemble, 75):.6f}")

    # === Resumen final ===
    logger.info(f"\n📮 Predicción final (test):")
    logger.info(
        f"   Clientes a contactar:     {N_enviados_final:,} "
        f"({N_enviados_final / len(prediccion_final_binaria) * 100:.2f}%)"
    )
    logger.info(
        f"   Clientes sin contactar:   {(prediccion_final_binaria == 0).sum():,}"
    )

    logger.info(f"\n✅ Archivo generado correctamente en: {submission_path}")
    logger.info(f"{sep}")

    return submission_path

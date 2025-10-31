import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def generar_reporte_ensemble(
    test_data,
    prediccion_final_binaria,
    probabilidades_junio_ensemble,
    umbrales_individuales,
    umbral_promedio_individual,
    umbral_ensemble,
    umbral_junio,
    ganancia_ensemble,
    N_ensemble,
    semillas,
    N_enviados_final,
    nombre_modelo="modelo",
    trial_number=None,
    output_dir="resultados_prediccion"
):
    """
    Genera el archivo final del ensemble y un resumen detallado.
    Guarda el CSV en resultados_prediccion/ y usa logger.info para reportar.
    """

    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    # === Generar CSV ===
    submission = pd.DataFrame({
        'numero_de_cliente': test_data['numero_de_cliente'].values,
        'Predicted': prediccion_final_binaria
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    submission_filename = f"{nombre_modelo}_T{trial_number or 'final'}_U{umbral_junio:.6f}_N{N_enviados_final}_{timestamp}.csv"
    submission_path = os.path.join(output_dir, submission_filename)
    submission.to_csv(submission_path, index=False)

    # === Logging ===
    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("📦 GENERANDO SUBMISSION CON ENSEMBLE")
    logger.info(f"{sep}")
    logger.info(f"🎯 Umbral usado para junio: {umbral_junio:.6f}")
    logger.info(f"📮 Número de envíos: {N_enviados_final:,}")

    logger.info(f"\n💾 Archivo guardado: {submission_path}")
    logger.info(f"📊 Distribución: {submission['Predicted'].value_counts().to_dict()}")
    logger.info(f"   - Clase 0 (no enviar): {(prediccion_final_binaria == 0).sum():,}")
    logger.info(f"   - Clase 1 (enviar):    {(prediccion_final_binaria == 1).sum():,}")
    logger.info(f"   - Total registros:     {len(prediccion_final_binaria):,}")

    # Muestra de las primeras filas
    logger.info("\n📋 Muestra del submission:")
    logger.info(f"\n{submission.head(10).to_string(index=False)}")

    # === Estadísticas completas ===
    logger.info(f"\n{sep}")
    logger.info("📊 RESUMEN COMPLETO DEL ENSEMBLE")
    logger.info(f"{sep}")
    logger.info(f"🌱 Número de semillas usadas: {len(semillas)}")
    logger.info(f"   Semillas: {semillas}")

    logger.info("\n📐 Umbrales individuales (por semilla):")
    for i, (seed, umbral) in enumerate(zip(semillas, umbrales_individuales), 1):
        logger.info(f"   {i}. Seed {seed}: {umbral:.6f}")

    logger.info("\n🎯 Umbrales finales:")
    logger.info(f"   Promedio de individuales:     {umbral_promedio_individual:.6f}")
    logger.info(f"   Umbral óptimo en Abril:       {umbral_ensemble:.6f}")
    logger.info(f"   Umbral usado en Junio:        {umbral_junio:.6f}")
    if umbral_junio != umbral_ensemble:
        ajuste = ((umbral_junio - umbral_ensemble) / umbral_ensemble) * 100
        logger.info(f"   Ajuste aplicado:              {ajuste:+.2f}%")
    logger.info(f"   Desviación estándar:          {np.std(umbrales_individuales):.6f}")

    logger.info(f"\n💰 Ganancia en validación (Abril): ${ganancia_ensemble:,.0f}")
    logger.info(f"   N óptimo en abril: {N_ensemble:,}")

    logger.info(f"\n📊 Probabilidades del ensemble (Junio):")
    logger.info(f"   - Mínima:  {probabilidades_junio_ensemble.min():.6f}")
    logger.info(f"   - Máxima:  {probabilidades_junio_ensemble.max():.6f}")
    logger.info(f"   - Media:   {probabilidades_junio_ensemble.mean():.6f}")
    logger.info(f"   - Mediana: {np.median(probabilidades_junio_ensemble):.6f}")
    logger.info(f"   - Q1:      {np.percentile(probabilidades_junio_ensemble, 25):.6f}")
    logger.info(f"   - Q3:      {np.percentile(probabilidades_junio_ensemble, 75):.6f}")

    logger.info(f"\n📮 Predicción final:")
    logger.info(f"   Clientes a contactar: {N_enviados_final:,} ({N_enviados_final/len(prediccion_final_binaria)*100:.2f}%)")
    logger.info(f"   Clientes sin contactar: {(prediccion_final_binaria == 0).sum():,}")

    logger.info(f"\n✅ Archivo generado en: {submission_path}")
    logger.info(f"{sep}")

    return submission_path

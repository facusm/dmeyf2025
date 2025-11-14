# src/resultados_ensemble.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

from config.config import RESULTADOS_PREDICCION_PATH, NOMBRE_EXPERIMENTO
from src.utils import logger


def generar_reporte_ensemble(
    test_data,
    prediccion_final_binaria,
    probabilidades_test_ensemble,
    umbrales_individuales,
    umbral_promedio_individual,
    umbral_ensemble,
    umbral_aplicado_test,
    ganancia_ensemble,
    N_ensemble,
    semillas,
    N_enviados_final,
    nombre_modelo="modelo",
    trial_number=None,
):
    """
    Genera CSV final + reporte detallado del ensemble.
    """
    os.makedirs(RESULTADOS_PREDICCION_PATH, exist_ok=True)

    submission = pd.DataFrame({
        "numero_de_cliente": test_data["numero_de_cliente"].values,
        "Predicted": prediccion_final_binaria
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = (
        f"{NOMBRE_EXPERIMENTO}_{nombre_modelo}_"
        f"T{trial_number or 'final'}_U{umbral_aplicado_test:.6f}_"
        f"N{N_enviados_final}_{timestamp}.csv"
    )
    path = os.path.join(RESULTADOS_PREDICCION_PATH, filename)
    submission.to_csv(path, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("📦 SUBMISSION DEL ENSEMBLE GENERADA")
    logger.info("=" * 60)
    logger.info(f"Archivo: {path}")
    logger.info(f"Total envíos test: {N_enviados_final:,}")
    logger.info(f"Umbral aplicado test: {umbral_aplicado_test:.6f}")
    logger.info(f"Umbral ensemble valid: {umbral_ensemble:.6f}")

    # Stats
    logger.info("\n📊 Probabilidades ensemble en test:")
    logger.info(f"Min:  {probabilidades_test_ensemble.min():.6f}")
    logger.info(f"Max:  {probabilidades_test_ensemble.max():.6f}")
    logger.info(f"Mean: {probabilidades_test_ensemble.mean():.6f}")
    logger.info(f"Std:  {probabilidades_test_ensemble.std():.6f}")

    return path

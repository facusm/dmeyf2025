# src/utils.py

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime

from config.config import (
    GANANCIA_ACIERTO,
    COSTO_ESTIMULO,
    LOGS_PATH,
    NOMBRE_EXPERIMENTO,  # ⬅️ para taggear outputs por experimento
)

# === CONFIGURACIÓN DE LOGGING === #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Evitar duplicar handlers si se importa múltiples veces
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def mejor_umbral_probabilidad(y_pred, weights, ganancia_acierto=None, costo_estimulo=None):
    """
    Encuentra el umbral de probabilidad óptimo que maximiza la ganancia.
    """
    ganancia_acierto = ganancia_acierto or GANANCIA_ACIERTO
    costo_estimulo = costo_estimulo or COSTO_ESTIMULO

    # Filtrar valores finitos
    mask = np.isfinite(y_pred)
    y_pred = np.array(y_pred)[mask]
    weights = np.array(weights)[mask]

    # Ordenar por probabilidad descendente
    orden = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[orden]
    weights_sorted = weights[orden]

    # Ganancia NETA por envío
    ganancias = np.where(
        weights_sorted == 1.00002,
        ganancia_acierto - abs(costo_estimulo),
        -abs(costo_estimulo),
    )

    gan_acum = np.cumsum(ganancias)

    if len(gan_acum) == 0:
        return 0, 0, 0, ([], [], [])

    # Buscamos el máximo en el top 70% para evitar colas ruidosas
    limite_busqueda = int(len(gan_acum) * 0.7)
    idx_max = np.argmax(gan_acum[:limite_busqueda])

    ganancia_max = gan_acum[idx_max]
    N_optimo = idx_max + 1
    umbral_optimo = y_pred_sorted[idx_max]

    ns = list(range(1, len(gan_acum) + 1))
    umbrales = list(y_pred_sorted)

    return umbral_optimo, N_optimo, ganancia_max, (ns, gan_acum, umbrales)


def aplicar_undersampling(
    data,
    target_col="clase_ternaria",
    id_col="numero_de_cliente",
    rate=1.0,
    seed=42,
    output_dir=None,
):
    """
    Aplica undersampling a nivel cliente SOLO a aquellos que son CONTINUA en todos sus registros.
    Guarda un CSV trazable con los clientes eliminados/conservados.

    - Si `output_dir` es None, se usa LOGS_PATH del experimento actual.
    - El archivo queda nombrado como:
        {NOMBRE_EXPERIMENTO}_undersampling_rate_{rate}_{timestamp}.csv
    """

    if not 0 < rate <= 1:
        raise ValueError("El parámetro 'rate' debe estar entre 0 y 1.")

    # Carpeta base: logs del experimento. Subcarpeta específica para undersampling.
    base_dir = output_dir or LOGS_PATH
    output_dir = os.path.join(base_dir, "undersampling")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("🔎 Iniciando undersampling a nivel cliente (CONTINUA puros)...")
    logger.info(f"🏷️ Experimento: {NOMBRE_EXPERIMENTO}")
    logger.info(f"🎯 Rate solicitado: {rate:.2f} | Seed: {seed}")

    # 1️⃣ Clientes CONTINUA puros
    clientes_continua_puros = (
        data.groupby(id_col)[target_col]
        .apply(lambda x: all(x == "CONTINUA"))
        .loc[lambda s: s]
        .index.to_numpy()
    )
    logger.info(f"Clientes CONTINUA puros encontrados: {len(clientes_continua_puros):,}")

    # 2️⃣ Clientes que alguna vez fueron BAJA
    clientes_baja = data.loc[data[target_col].isin(["BAJA+1", "BAJA+2"]), id_col].unique()
    logger.info(f"Clientes con alguna BAJA: {len(clientes_baja):,}")

    # 3️⃣ Submuestreo de CONTINUA puros
    if rate < 1.0:
        n_keep = int(len(clientes_continua_puros) * rate)
        rng = np.random.RandomState(seed)
        clientes_continua_keep = rng.choice(clientes_continua_puros, n_keep, replace=False)
        logger.info(
            f"Aplicado undersampling sobre CONTINUA puros: {rate*100:.0f}% "
            f"({n_keep:,} clientes retenidos de {len(clientes_continua_puros):,})"
        )
    else:
        clientes_continua_keep = clientes_continua_puros
        logger.info("No se aplica undersampling (rate=1.0): se conservan todos los CONTINUA puros.")

    # 4️⃣ Clientes finales a conservar
    clientes_finales = np.concatenate([clientes_continua_keep, clientes_baja])
    df_final = data[data[id_col].isin(clientes_finales)].reset_index(drop=True)

    # 5️⃣ Log detallado de eliminación/conservación
    clientes_eliminados = np.setdiff1d(clientes_continua_puros, clientes_continua_keep)
    logger.info(f"Clientes CONTINUA eliminados por undersampling: {len(clientes_eliminados):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        output_dir,
        f"{NOMBRE_EXPERIMENTO}_undersampling_rate_{rate:.2f}_{timestamp}.csv",
    )

    df_out = pd.DataFrame({
        "cliente": np.concatenate([clientes_eliminados, clientes_continua_keep]),
        "accion": (
            ["eliminado"] * len(clientes_eliminados)
            + ["conservado"] * len(clientes_continua_keep)
        ),
    })
    df_out.to_csv(csv_path, index=False)
    logger.info(f"📄 Log de undersampling guardado en: {csv_path}")

    # 6️⃣ Distribución post-undersampling
    distrib = df_final[target_col].value_counts(normalize=True).round(3).to_dict()
    logger.info(f"📊 Distribución post-undersampling (frecuencias relativas): {distrib}")

    return df_final

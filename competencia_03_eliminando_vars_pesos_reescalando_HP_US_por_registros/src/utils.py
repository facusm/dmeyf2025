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


import os
import pandas as pd
import numpy as np
from datetime import datetime

# Asumo que tienes configurado logger, si no, reemplaza por print
import logging
logger = logging.getLogger(__name__)


def aplicar_undersampling(
    data,
    target_col="clase_ternaria",
    id_col="numero_de_cliente",
    rate=1.0,
    seed=42,
    output_dir=None,
):
    """
    Aplica undersampling a nivel ROW (Registro) estilo 'Línea de Muerte'.
    
    Lógica:
    - Se genera un número aleatorio para CADA registro.
    - Se conservan SIEMPRE los registros que NO son 'CONTINUA' (ej. BAJA+1, BAJA+2).
    - De los registros 'CONTINUA', se conserva solo el % indicado por 'rate'.
    
    El archivo de log guardará [numero_de_cliente, foto_mes, accion].
    """

    if not 0 < rate <= 1:
        raise ValueError("El parámetro 'rate' debe estar entre 0 y 1.")

    # Configuración de directorios (adaptado a tu lógica original)
    # Nota: Si LOGS_PATH no es global, asegúrate de pasar output_dir
    base_dir = output_dir or "./logs" 
    output_dir = os.path.join(base_dir, "undersampling")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("🔎 Iniciando undersampling a nivel REGISTRO (Row-level)...")
    logger.info(f"🎯 Rate CONTINUA: {rate:.2f} | Seed: {seed}")

    # 1️⃣ Generar azar y máscara
    # Usamos la semilla para replicabilidad exacta
    np.random.seed(seed)
    
    # Vector de azar para TODAS las filas
    azar = np.random.uniform(0, 1, size=len(data))

    # Condición de conservación (Line of Death logic):
    # - Si el azar es bajo (entran por sorteo)
    # - O si NO son CONTINUA (son Bajas, entran siempre)
    # Asumimos que todo lo que no es BAJA+1/BAJA+2 es CONTINUA, o usamos != "CONTINUA"
    # Para ser precisos con tu taxonomía anterior:
    es_baja = data[target_col].isin(["BAJA+1", "BAJA+2"])
    # Si quieres ser estricto con que SOLO se borra CONTINUA:
    mask_keep = (azar <= rate) | es_baja

    # 2️⃣ Aplicar filtro
    df_final = data[mask_keep].reset_index(drop=True)
    
    # Estadísticas
    cant_original = len(data)
    cant_final = len(df_final)
    cant_eliminados = cant_original - cant_final
    
    logger.info(f"Registros originales: {cant_original:,}")
    logger.info(f"Registros eliminados: {cant_eliminados:,}")
    logger.info(f"Registros finales:    {cant_final:,}")

    # 3️⃣ Generar Log Trazable (CSV)
    # Identificamos qué se borró para el log
    # A diferencia de antes, ahora necesitamos 'foto_mes' para saber QUÉ registro se borró
    cols_log = [id_col]
    if 'foto_mes' in data.columns:
        cols_log.append('foto_mes')
    
    # Obtenemos los índices de lo que se borró (lo opuesto a mask_keep)
    df_eliminados = data.loc[~mask_keep, cols_log].copy()
    df_eliminados['accion'] = 'eliminado'
    
    # Para no hacer el log gigante, podemos guardar solo los eliminados
    # O si prefieres mantener la lógica anterior de guardar TODO (cuidado con el tamaño):
    # df_conservados = data.loc[mask_keep, cols_log].copy()
    # df_conservados['accion'] = 'conservado'
    # df_log = pd.concat([df_eliminados, df_conservados])
    
    # Por eficiencia, sugiero guardar solo lo eliminado en este enfoque masivo
    df_log = df_eliminados 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        output_dir,
        f"undersampling_row_rate_{rate:.2f}_{timestamp}.csv",
    )
    
    # Guardamos
    try:
        df_log.to_csv(csv_path, index=False)
        logger.info(f"📄 Log de registros ELIMINADOS guardado en: {csv_path}")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo guardar el log CSV: {e}")

    # 4️⃣ Distribución post-undersampling
    distrib = (
        df_final[target_col]
        .value_counts(normalize=True)
        .round(3)
        .to_dict()
    )
    logger.info(f"📊 Distribución post-undersampling: {distrib}")

    return df_final



def rescalar_hp_apostyle(params, n_old, n_new):
    factor = n_new / n_old
    params = params.copy()

    if "min_data_in_leaf" in params:
        params["min_data_in_leaf"] = int(round(params["min_data_in_leaf"] * factor))

    return params

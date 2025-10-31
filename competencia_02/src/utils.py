# src/utils.py
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from config.config import GANANCIA_ACIERTO, COSTO_ESTIMULO

# === CONFIGURACIÓN DE LOGGING === #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


def mejor_umbral_probabilidad(y_pred, weights, ganancia_acierto=None, costo_estimulo=None):
    """
    Encuentra el umbral de probabilidad óptimo que maximiza la ganancia.

    Parameters
    ----------
    y_pred : np.array
        Probabilidades predichas por el modelo.
    weights : np.array
        Pesos de las observaciones (clase_peso: 1.00002 = BAJA+2, 1.0 = CONTINUA, etc).
    ganancia_acierto : float, optional
        Ganancia por acierto positivo. Si no se provee, usa el valor de config.
    costo_estimulo : float, optional
        Costo de estímulo. Si no se provee, usa el valor de config.

    Returns
    -------
    umbral_optimo : float
        Umbral que maximiza la ganancia.
    N_en_umbral : int
        Número de casos por encima del umbral óptimo.
    ganancia_max : float
        Valor máximo de ganancia.
    curva : tuple
        (ns, gan_acum, umbrales) para análisis posterior.
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

    # Ganancia NETA por envío:
    # - BAJA+2 (peso 1.00002): ganamos ganancia_acierto pero pagamos costo_estimulo
    # - Otros: solo perdemos el costo_estimulo
    ganancias = np.where(
        weights_sorted == 1.00002,
        ganancia_acierto - abs(costo_estimulo),
        -abs(costo_estimulo)
    )

    gan_acum = np.cumsum(ganancias)

    if len(gan_acum) == 0:
        return 0, 0, 0, ([], [], [])

    # Buscar máximo en el primer 70%
    limite_busqueda = int(len(gan_acum) * 0.7)
    idx_max = np.argmax(gan_acum[:limite_busqueda])

    ganancia_max = gan_acum[idx_max]
    N_optimo = idx_max + 1
    umbral_optimo = y_pred_sorted[idx_max]

    # Preparar curva para graficar
    ns = list(range(1, len(gan_acum) + 1))
    umbrales = list(y_pred_sorted)

    return umbral_optimo, N_optimo, ganancia_max, (ns, gan_acum, umbrales)


def aplicar_undersampling(
    data,
    target_col="clase_ternaria",
    id_col="numero_de_cliente",
    rate=1.0,
    seed=42,
    output_dir="logs"
):
    """
    Aplica undersampling a nivel cliente SOLO a aquellos que son CONTINUA en todos sus registros.
    Los clientes que alguna vez fueron BAJA+1 o BAJA+2 se mantienen completos.

    Además, guarda un CSV en 'logs/' con los clientes eliminados y conservados.

    Parámetros
    ----------
    data : pd.DataFrame
        Dataset original con columna de clase (por ej. 'clase_ternaria').
    target_col : str
        Nombre de la columna target.
    id_col : str
        Columna identificadora del cliente.
    rate : float
        Porcentaje de clientes CONTINUA a mantener (0 < rate <= 1).
    seed : int
        Semilla para reproducibilidad.
    output_dir : str
        Carpeta donde guardar el log CSV.

    Retorna
    -------
    pd.DataFrame
        DataFrame con undersampling aplicado a nivel cliente.
    """

    if not 0 < rate <= 1:
        raise ValueError("El parámetro 'rate' debe estar entre 0 y 1.")

    os.makedirs(output_dir, exist_ok=True)

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

    # 3️⃣ Submuestreo
    if rate < 1.0:
        n_keep = int(len(clientes_continua_puros) * rate)
        rng = np.random.RandomState(seed)
        clientes_continua_keep = rng.choice(clientes_continua_puros, n_keep, replace=False)
        logger.info(f"Aplicado undersampling: CONTINUA puros -> {rate*100:.0f}% ({n_keep:,} clientes)")
    else:
        clientes_continua_keep = clientes_continua_puros
        logger.info("No se aplica undersampling (rate=1.0)")

    # 4️⃣ Clientes finales a conservar
    clientes_finales = np.concatenate([clientes_continua_keep, clientes_baja])
    df_final = data[data[id_col].isin(clientes_finales)].reset_index(drop=True)

    # 5️⃣ Log simplificado
    clientes_eliminados = np.setdiff1d(clientes_continua_puros, clientes_continua_keep)
    logger.info(f"Clientes CONTINUA eliminados: {len(clientes_eliminados):,}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"undersampling_rate_{rate:.1f}_{timestamp}.csv")

    df_out = pd.DataFrame({
        "cliente": np.concatenate([clientes_eliminados, clientes_continua_keep]),
        "accion": ["eliminado"]*len(clientes_eliminados) + ["conservado"]*len(clientes_continua_keep)
    })
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Archivo de log guardado en: {csv_path}")

    # 6️⃣ Distribución post-undersampling
    distrib = df_final[target_col].value_counts(normalize=True).round(3).to_dict()
    logger.info(f"Distribución post-undersampling: {distrib}")

    return df_final
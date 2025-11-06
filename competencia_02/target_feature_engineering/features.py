import pandas as pd
import duckdb
import logging
import numpy as np

logger = logging.getLogger("__name__")


def pisar_con_mes_anterior_duckdb(
    df: pd.DataFrame,
    variable: str,
    meses_anomalos: list,                 # ej: [201905, 202006]
    id_col: str = "numero_de_cliente",
    mes_col: str = "foto_mes",
) -> pd.DataFrame:
    """
    Para cada mes en `meses_anomalos`, pisa `variable` con el último valor válido
    PREVIO del mismo cliente, considerando solo meses NO anómalos.
    
    - Si no hay valor previo válido -> deja NaN en el mes anómalo.
    - No mira el futuro (solo valores anteriores).
    - Soporta meses anómalos consecutivos:
        toda la racha se rellena con el último mes sano previo (si existe).
    - Solo modifica filas cuyo mes está en meses_anomalos.
    """
    if not meses_anomalos:
        return df

    meses_anomalos = [int(m) for m in meses_anomalos]

    slim = df[[id_col, mes_col, variable]].copy()

    con = duckdb.connect()
    con.execute("PRAGMA threads = system;")
    con.register("t_in", slim)
    con.register("meses_bad", pd.DataFrame({mes_col: meses_anomalos}))

    query = f"""
    WITH base AS (
        SELECT
            {id_col} AS id_,
            CAST({mes_col} AS BIGINT) AS t_,
            CAST({variable} AS DOUBLE) AS v_
        FROM t_in
    ),
    mark AS (
        SELECT
            b.*,
            (m.{mes_col} IS NOT NULL) AS is_bad
        FROM base b
        LEFT JOIN meses_bad m
          ON b.t_ = CAST(m.{mes_col} AS BIGINT)
    ),
    -- Solo consideramos como "fuente válida" los meses NO anómalos
    fuente AS (
        SELECT
            id_,
            t_,
            CASE WHEN is_bad THEN NULL ELSE v_ END AS v_fuente,
            is_bad
        FROM mark
    ),
    -- prev_good: último valor válido de un mes NO anómalo hacia atrás
    w AS (
        SELECT
            *,
            LAST_VALUE(v_fuente) IGNORE NULLS OVER (
                PARTITION BY id_
                ORDER BY t_
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS prev_good
        FROM fuente
    ),
    resolved AS (
        SELECT
            id_  AS {id_col},
            t_   AS {mes_col},
            CASE
              WHEN is_bad THEN prev_good   -- mes anómalo -> último sano previo (o NULL)
              ELSE v_                      -- mes normal -> valor original
            END AS v_corr
        FROM w
    )
    SELECT * FROM resolved
    """

    corr = con.execute(query).df()
    out = df.merge(corr, on=[id_col, mes_col], how="left")

    # Solo tocar meses anómalos
    mask_bad = out[mes_col].isin(meses_anomalos)

    # Si tenemos prev_good (v_corr notna) -> pisamos
    mask_ok = mask_bad & out["v_corr"].notna()
    out.loc[mask_ok, variable] = out.loc[mask_ok, "v_corr"]

    # Si NO tenemos prev_good -> NaN (aunque el original fuera 0/basura)
    mask_no_prev = mask_bad & out["v_corr"].isna()
    out.loc[mask_no_prev, variable] = np.nan

    out.drop(columns=["v_corr"], inplace=True)
    return out


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados. Si la columna ya existe, no se vuelve a generar.
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                col_name = f"{attr}_lag_{i}"
                if col_name not in df.columns:
                    sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col_name}"
                else:
                    logger.warning(f"La columna {col_name} ya existe, no se vuelve a generar")

  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df




def feature_engineering_deltas(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera columnas de delta en SQL: valor actual menos lag correspondiente.
    Mantiene NULL si no hay información suficiente.
    """
    logger.info(f"Generando deltas (SQL) para {len(columnas)} columnas con {cant_lag} lags")

    sql = "SELECT *"

    for attr in columnas:
        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag_{i}"
            delta_col = f"{attr}_delta_{i}"

            if lag_col in df.columns and delta_col not in df.columns:
                sql += f", {attr} - {lag_col} AS {delta_col}"
            elif delta_col in df.columns:
                logger.warning(f"{delta_col} ya existe, no se genera nuevamente")
            else:
                logger.warning(f"{lag_col} no existe, no se puede generar {delta_col}")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Deltas generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out





def feature_engineering_medias_moviles(df: pd.DataFrame, columnas: list[str], window_size: int = 3) -> pd.DataFrame:
    """
    Genera columnas de medias móviles estrictas en SQL incluyendo el valor actual y los anteriores
    según el tamaño de ventana especificado. Solo calcula la media si la ventana está completa;
    de lo contrario deja NULL.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales
    columnas : list[str]
        Lista de columnas sobre las cuales generar la media móvil
    window_size : int
        Tamaño de la ventana para la media móvil

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas *_ma_{window_size} agregadas
    """
    logger.info(f"Generando medias móviles estrictas (SQL) para {len(columnas)} columnas con ventana de {window_size}")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron columnas para generar medias móviles.")
        return df

    sql = "SELECT *"
    for attr in columnas:
        ma_col_name = f"{attr}_ma_{window_size}"
        if ma_col_name not in df.columns:
            sql += (
                f', CASE '
                f'WHEN COUNT("{attr}") OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes '
                f'ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW) = {window_size} '
                f'THEN AVG("{attr}") OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes '
                f'ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW) '
                f'ELSE NULL END AS "{ma_col_name}"'
            )
        else:
            logger.warning(f"{ma_col_name} ya existe, no se vuelve a generar")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Medias móviles generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out



def feature_engineering_cum_sum(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
  """
  Genera columnas de suma acumulada por cliente para los atributos indicados.
  Los valores NULL se tratan como 0 solo para el cálculo de la suma acumulada,
  sin modificar las columnas originales.

  Parámetros
  ----------
  df : pd.DataFrame
      DataFrame con los datos originales
  columnas : list[str]
      Lista de columnas sobre las cuales generar la suma acumulada

  Retorna
  -------
  pd.DataFrame
  DataFrame con las nuevas columnas *_cumsum agregadas
  """
  if columnas is None or len(columnas) == 0:
    logger.warning("No se especificaron atributos para generar lags")
    return df

  sql = "select *"

  for attr in columnas:
    if attr in df.columns:
      if f"{attr}_cumsum" not in df.columns:
        sql += f", sum(coalesce({attr}, 0)) over (partition by numero_de_cliente order by foto_mes ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {attr}_cumsum"
      else:
        logger.warning(f"{attr}_cumsum ya existe, no se vuelve a generar")

  sql += f" from df"

  logger.debug(f"Consulta SQL: {sql}")

  # Ejecutar la consulta SQL
  con = duckdb.connect(database=":memory:")
  con.register("df", df)
  df = con.execute(sql).df()
  con.close()


  logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

  return df


def feature_engineering_min_max(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de min y max por cliente para los atributos especificados.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de atributos para los cuales generar min y max.

    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de min y max agregadas.
    """
    logger.info(f"Generando min y max para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar min y max")
        return df

    sql = "SELECT *"

    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"La columna {attr} no existe en el DataFrame")
            continue

        col_min = f"{attr}_min"
        col_max = f"{attr}_max"

        if col_min not in df.columns:
            sql += f", MIN({attr}) OVER (PARTITION BY numero_de_cliente) AS {col_min}"
        else:
            logger.warning(f"La columna {col_min} ya existe, no se vuelve a generar")

        if col_max not in df.columns:
            sql += f", MAX({attr}) OVER (PARTITION BY numero_de_cliente) AS {col_max}"
        else:
            logger.warning(f"La columna {col_max} ya existe, no se vuelve a generar")

    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()


    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df



def feature_engineering_ratios(df: pd.DataFrame, ratio_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Genera columnas de ratios entre pares válidos de columnas.
    Maneja NULL y división por cero.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales
    ratio_pairs : list[tuple[str, str]]
        Lista de tuplas (numerador, denominador) indicando los ratios a generar

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas *_ratio agregadas
    """
    if not ratio_pairs:
        logger.warning("No se especificaron pares de columnas para generar ratios")
        return df

    sql = "SELECT *"

    for numerador, denominador in ratio_pairs:
        ratio_col = f"{numerador}_over_{denominador}"

        if numerador in df.columns and denominador in df.columns:
            if ratio_col not in df.columns:
                # NULLIF evita división por cero
                sql += f", {numerador} / NULLIF({denominador}, 0) AS {ratio_col}"
            else:
                logger.warning(f"{ratio_col} ya existe, no se vuelve a generar")
        else:
            logger.warning(f"Columnas no encontradas: {numerador} o {denominador}, se omite el ratio {ratio_col}")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Ratios generados. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out

def feature_engineering_medias_moviles_lag(df: pd.DataFrame, columnas: list[str], window_size: int = 2) -> pd.DataFrame:
    """
    Genera columnas de medias móviles de los N meses previos sin incluir el mes actual.
    Solo calcula la media si la ventana está completa; de lo contrario deja NULL.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales
    columnas : list[str]
        Lista de columnas sobre las cuales generar la media móvil
    window_size : int
        Tamaño de la ventana para la media móvil (número de meses anteriores)

    Retorna
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas *_ma_lag_{window_size} agregadas
    """
    logger.info(f"Generando medias móviles lag de {window_size} meses para {len(columnas)} columnas")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron columnas para generar medias móviles.")
        return df

    sql = "SELECT *"
    for attr in columnas:
        ma_col_name = f"{attr}_ma_lag_{window_size}"
        if ma_col_name not in df.columns:
            sql += (
                f', CASE '
                f'WHEN COUNT("{attr}") OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes '
                f'ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING) = {window_size} '
                f'THEN AVG("{attr}") OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes '
                f'ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING) '
                f'ELSE NULL END AS "{ma_col_name}"'
            )
        else:
            logger.warning(f"{ma_col_name} ya existe, no se vuelve a generar")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    con.register("df", df)
    df_out = con.execute(sql).df()
    con.close()

    logger.info(f"Medias móviles lag generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def generar_shock_relativo_delta_lag(df: pd.DataFrame, columnas: list[str], window_size: int = 2) -> pd.DataFrame:
    """
    Genera columnas de 'shock relativo delta lag' frente a la media móvil lag ya existente.

    El shock relativo se calcula como la diferencia entre el valor actual y la media de
    los N meses previos (media móvil lag).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas originales y las medias móviles lag
    columnas : list[str]
        Lista de columnas originales para calcular el shock relativo
    window_size : int
        Ventana de la media móvil lag usada (para identificar el nombre de la columna)

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas 'shock_relativo_delta_lag_*' agregadas
    """
    for col in columnas:
        ma_lag_col = f"{col}_ma_lag_{window_size}"
        df[f"shock_relativo_delta_lag_{col}"] = df[col] - df[ma_lag_col]
    
    return df


def crear_indicador_aguinaldo(df: pd.DataFrame, columna_mes: str = "foto_mes") -> pd.DataFrame:
    """
    Crea una variable indicadora que vale 1 si el mes es de aguinaldo (junio o diciembre) y 0 en caso contrario.
    La columna 'foto_mes' se espera en formato YYYYMM (ej: 202106).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con la columna del mes
    columna_mes : str
        Nombre de la columna que contiene el mes (por ejemplo, 'foto_mes')

    Retorna
    -------
    pd.DataFrame
        DataFrame con la nueva columna 'mes_con_aguinaldo'
    """
    df["mes_con_aguinaldo"] = df[columna_mes].apply(lambda x: 1 if x % 100 in [6, 12] else 0)
    return df

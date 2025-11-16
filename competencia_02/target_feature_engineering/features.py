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
    PREVIO del mismo cliente (LOCF). Además crea un flag binario <variable>_locf que indica
    que esa celda fue corregida (o se intentó corregir y no había previo -> NaN).

    - No usa información futura.
    - Soporta rachas: toda la racha toma el último mes sano previo.
    - Solo modifica filas cuyo mes está en `meses_anomalos`.
    """
    if not meses_anomalos:
        return df

    if variable not in df.columns:
        logger.warning(f"[LOCF] La columna '{variable}' no existe. Se omite.")
        return df

    meses_anomalos = [int(m) for m in meses_anomalos]

    slim = df[[id_col, mes_col, variable]].copy()

    con = duckdb.connect()
    try:
        con.register("t_in", slim)
        con.register("meses_bad", pd.DataFrame({mes_col: meses_anomalos}))

        # Citar identificadores por si los nombres tienen caracteres especiales
        q_id = f'"{id_col}"'
        q_mes = f'"{mes_col}"'
        q_var = f'"{variable}"'

        query = f"""
        WITH base AS (
            SELECT
                {q_id} AS id_,
                CAST({q_mes} AS BIGINT) AS t_,
                CAST({q_var} AS DOUBLE) AS v_
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
        fuente AS (
            SELECT
                id_, t_, v_, is_bad,
                CASE WHEN is_bad THEN NULL ELSE v_ END AS v_fuente
            FROM mark
        ),
        w AS (
            SELECT *,
                MAX_BY(v_fuente, t_) OVER (
                    PARTITION BY id_
                    ORDER BY t_
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS prev_good
            FROM fuente
        ),
        resolved AS (
            SELECT
                id_  AS {q_id},
                t_   AS {q_mes},
                is_bad,
                v_   AS v_orig,
                prev_good,
                CASE WHEN is_bad THEN prev_good ELSE v_ END AS v_corr
            FROM w
        )
        SELECT * FROM resolved
        """

        corr = con.execute(query).df()
    finally:
        con.close()

    # Merge con el df original
    out = df.merge(
        corr[[id_col, mes_col, "is_bad", "v_orig", "v_corr"]],
        on=[id_col, mes_col],
        how="left"
    )

    mask_bad = out["is_bad"] == True

    # Flag: 1 si el mes es anómalo y el valor se cambió (o no había previo y queda NaN)
    flag_col = f"{variable}_locf"
    was_changed = mask_bad & (
        (out["v_corr"].notna() & (out["v_corr"] != out["v_orig"])) | (out["v_corr"].isna())
    )
    out[flag_col] = was_changed.astype(int)

    # Aplicar corrección (cuando hay previo válido)
    out.loc[mask_bad & out["v_corr"].notna(), variable] = out.loc[mask_bad & out["v_corr"].notna(), "v_corr"]

    # Sin previo válido -> NaN explícito
    out.loc[mask_bad & out["v_corr"].isna(), variable] = np.nan

    # Limpiar auxiliares
    out.drop(columns=["is_bad", "v_orig", "v_corr"], inplace=True)

    return out


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                col_name = f"{attr}_lag_{i}"
                if col_name not in df.columns:
                    sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {col_name}"
                else:
                    logger.warning(f"La columna {col_name} ya existe, no se vuelve a generar")

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df = con.execute(sql).df()
    finally:
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
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Deltas generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_medias_moviles(df: pd.DataFrame, columnas: list[str], window_size: int = 3) -> pd.DataFrame:
    """
    Genera columnas de medias móviles estrictas (incluye el valor actual) si la ventana está completa.
    """
    logger.info(f"Generando medias móviles estrictas (SQL) para {len(columnas)} columnas con ventana de {window_size}")

    if not columnas:
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
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Medias móviles generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_cum_sum(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera columnas de suma acumulada por cliente para los atributos indicados (solo hacia atrás).
    """
    if not columnas:
        logger.warning("No se especificaron atributos para cumsum")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr in df.columns:
            colname = f"{attr}_cumsum"
            if colname not in df.columns:
                sql += (
                    f", SUM(COALESCE({attr}, 0)) OVER ("
                    f"  PARTITION BY numero_de_cliente "
                    f"  ORDER BY foto_mes "
                    f"  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                    f") AS {colname}"
                )
            else:
                logger.warning(f"{colname} ya existe, no se vuelve a generar")

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Feature engineering cumsum completado. DataFrame resultante con {df.shape[1]} columnas")
    return df


def feature_engineering_min_max(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera min/max históricos por cliente HASTA CADA MES (sin mirar el futuro).
    """
    logger.info(f"Generando min y max históricos (sin fuga) para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar min y max")
        return df

    sql = "SELECT *"

    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"La columna {attr} no existe en el DataFrame, se omite en min/max")
            continue

        col_min = f"{attr}_min_hist"
        col_max = f"{attr}_max_hist"

        if col_min not in df.columns:
            sql += (
                f", MIN({attr}) OVER ("
                f"    PARTITION BY numero_de_cliente "
                f"    ORDER BY foto_mes "
                f"    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                f") AS {col_min}"
            )

        if col_max not in df.columns:
            sql += (
                f", MAX({attr}) OVER ("
                f"    PARTITION BY numero_de_cliente "
                f"    ORDER BY foto_mes "
                f"    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
                f") AS {col_max}"
            )

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Min/max históricos generados. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_ratios(df: pd.DataFrame, ratio_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Genera columnas de ratios entre pares válidos de columnas.
    Maneja NULL y división por cero.
    """
    if not ratio_pairs:
        logger.warning("No se especificaron pares de columnas para generar ratios")
        return df

    sql = "SELECT *"

    for numerador, denominador in ratio_pairs:
        ratio_col = f"{numerador}_over_{denominador}"

        if numerador in df.columns and denominador in df.columns:
            if ratio_col not in df.columns:
                # Forzar división flotante y evitar división por cero
                sql += f", (1.0 * {numerador}) / NULLIF({denominador}, 0) AS {ratio_col}"
            else:
                logger.warning(f"{ratio_col} ya existe, no se vuelve a generar")
        else:
            logger.warning(f"Columnas no encontradas: {numerador} o {denominador}, se omite el ratio {ratio_col}")

    sql += " FROM df"

    con = duckdb.connect(":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Ratios generados. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_medias_moviles_lag(df: pd.DataFrame, columnas: list[str], window_size: int = 2) -> pd.DataFrame:
    """
    Genera columnas de medias móviles de los N meses previos SIN incluir el mes actual.
    Solo calcula la media si la ventana está completa; de lo contrario deja NULL.
    """
    logger.info(f"Generando medias móviles lag de {window_size} meses para {len(columnas)} columnas")

    if not columnas:
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
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Medias móviles lag generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def generar_shock_relativo_delta_lag(df: pd.DataFrame, columnas: list[str], window_size: int = 2) -> pd.DataFrame:
    """
    Genera columnas de 'shock relativo delta lag' frente a la media móvil lag ya existente:
    shock = valor_actual - media_movil_de_los_{window_size}_meses_previos
    """
    for col in columnas:
        ma_lag_col = f"{col}_ma_lag_{window_size}"
        if ma_lag_col not in df.columns or col not in df.columns:
            logger.warning(f"[shock] Se omite {col}: falta {ma_lag_col} o la columna original.")
            continue
        df[f"shock_relativo_delta_lag_{col}"] = df[col] - df[ma_lag_col]
    return df


def crear_indicador_aguinaldo(df: pd.DataFrame, columna_mes: str = "foto_mes") -> pd.DataFrame:
    """
    Crea variable indicadora: 1 si el mes es junio o diciembre (aguinaldo), 0 en caso contrario.
    """
    df["mes_con_aguinaldo"] = df[columna_mes].apply(lambda x: 1 if int(x) % 100 in [6, 12] else 0)
    return df

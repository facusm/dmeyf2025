# features.py

import pandas as pd
import duckdb
import logging
import numpy as np

logger = logging.getLogger(__name__)

import duckdb
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)



def _qident(col: str) -> str:
    return f'"{col.replace(chr(34), chr(34)*2)}"'

def detectar_variables_rotas_por_mes(
    df: pd.DataFrame,
    columnas: list[str] | None = None,
    id_col: str = "numero_de_cliente",
    mes_col: str = "foto_mes",
    strict: bool = True,
) -> dict[int, list[str]]:
    """
    Variable "rota" en mes m si para TODOS los clientes en ese mes:
      - (def) todos los valores son 0.
    Si strict=True además exige: no haya NULLs en ese mes para esa variable.
      (Así evitás marcar como "rota" un mes donde está todo NULL.)
    """

    if columnas is None:
        columnas = list(df.columns)

    cols = [c for c in columnas if c not in (id_col, mes_col)]
    if not cols:
        return {}

    # Nombres auxiliares (para no chocar con columnas reales)
    alias_max = {c: f"__maxabs__{c}" for c in cols}
    alias_nulls = {c: f"__nulls__{c}" for c in cols}

    cast_exprs = ",\n".join(
        f"TRY_CAST({_qident(c)} AS DOUBLE) AS {_qident(c)}" for c in cols
    )
    maxabs_exprs = ",\n".join(
        f"MAX(ABS(COALESCE({_qident(c)}, 0))) AS {_qident(alias_max[c])}" for c in cols
    )
    nulls_exprs = ",\n".join(
        f"SUM(CASE WHEN {_qident(c)} IS NULL THEN 1 ELSE 0 END) AS {_qident(alias_nulls[c])}" for c in cols
    )

    query = f"""
    WITH t AS (
      SELECT
        TRY_CAST({_qident(mes_col)} AS BIGINT) AS mes,
        {cast_exprs}
      FROM df
    ),
    agg AS (
      SELECT
        mes,
        COUNT(*) AS n,
        {maxabs_exprs},
        {nulls_exprs}
      FROM t
      GROUP BY mes
    )
    SELECT * FROM agg
    ORDER BY mes
    """

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        out = con.execute(query).df()
    finally:
        con.close()

    rotos: dict[int, list[str]] = {}
    for _, row in out.iterrows():
        mes = int(row["mes"])
        n = int(row["n"])
        bad_cols = []

        for c in cols:
            maxabs = row[alias_max[c]]
            nulls = int(row[alias_nulls[c]])

            # maxabs==0 => todos los valores numéricos (o NULL->0 por COALESCE) son 0
            if maxabs == 0:
                if strict:
                    # estricta: no permito NULLs; evita "mes todo NULL" marcado como roto
                    if nulls == 0:
                        bad_cols.append(c)
                else:
                    # relajada: permito NULLs (tratados como 0 en maxabs)
                    # si querés evitar "todo NULL" incluso en relaxed, descomentá:
                    # if nulls < n:
                    bad_cols.append(c)

        if bad_cols:
            rotos[mes] = bad_cols

    return rotos


def feature_engineering_tendencia(
    df: pd.DataFrame,
    columnas: list[str],
    window_size: int = 6,
    id_col: str = "numero_de_cliente",
    mes_col: str = "foto_mes",
    include_current: bool = True,
) -> pd.DataFrame:
    """
    Genera tendencia (pendiente) por cliente para cada columna usando REGR_SLOPE en DuckDB.

    - x = row_number() por cliente (equivalente a usar 1..k dentro de la ventana, salvo corrimiento).
    - Incluye NULLs: REGR_SLOPE ignora pares donde y es NULL.
    - Devuelve NULL si hay menos de 2 valores no-null en la ventana (como tu C: libre > 1)
    - Ventana:
        * include_current=True  -> últimos window_size incluyendo el mes actual
        * include_current=False -> últimos window_size previos (sin incluir el actual)
    """
    logger.info(
        f"Generando tendencia (regr_slope) con ventana {window_size} "
        f"({'incluye' if include_current else 'excluye'} mes actual) para {len(columnas) if columnas else 0} columnas"
    )

    if not columnas:
        return df

    # frame window
    if include_current:
        frame = f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
    else:
        frame = f"ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING"

    sql = """
    WITH base AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY {q_id}
          ORDER BY {q_mes}
        ) AS rn
      FROM df
    )
    SELECT
      base.*
    """.format(q_id=_qident(id_col), q_mes=_qident(mes_col))

    for col in columnas:
        if col not in df.columns:
            logger.warning(f"[trend] La columna {col} no existe, se omite")
            continue

        out_col = f"{col}_trend_{window_size}"
        if out_col in df.columns:
            logger.warning(f"[trend] {out_col} ya existe, no se vuelve a generar")
            continue

        q_col = _qident(col)
        q_out = _qident(out_col)

        # >=2 valores no-null en ventana, si no -> NULL (similar a libre > 1)
        sql += f""",
      CASE
        WHEN COUNT({q_col}) OVER (
          PARTITION BY {_qident(id_col)}
          ORDER BY {_qident(mes_col)}
          {frame}
        ) > 1
        THEN REGR_SLOPE({q_col}, rn) OVER (
          PARTITION BY {_qident(id_col)}
          ORDER BY {_qident(mes_col)}
          {frame}
        )
        ELSE NULL
      END AS {q_out}
        """

    sql += "\nFROM base"

    con = duckdb.connect(":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Tendencias generadas. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


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

    con = duckdb.connect(database=":memory:")
    try:
        con.register("t_in", slim)
        con.register("meses_bad", pd.DataFrame({mes_col: meses_anomalos}))

        q_id = _qident(id_col)
        q_mes = _qident(mes_col)
        q_var = _qident(variable)

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
                (m.{q_mes} IS NOT NULL) AS is_bad
            FROM base b
            LEFT JOIN meses_bad m
              ON b.t_ = CAST(m.{q_mes} AS BIGINT)
        ),
        fuente AS (
            SELECT
                id_, t_, v_, is_bad,
                CASE WHEN is_bad THEN NULL ELSE v_ END AS v_fuente
            FROM mark
        ),
        w AS (
            SELECT *,
                MAX_BY(
                    v_fuente,
                    CASE WHEN is_bad THEN -1 ELSE t_ END
                ) OVER (
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

    out = df.merge(
        corr[[id_col, mes_col, "is_bad", "v_orig", "v_corr"]],
        on=[id_col, mes_col],
        how="left"
    )

    mask_bad = out["is_bad"] == True

    flag_col = f"{variable}_locf"
    was_changed = mask_bad & (
        (out["v_corr"].notna() & (out["v_corr"] != out["v_orig"])) | (out["v_corr"].isna())
    )
    out[flag_col] = was_changed.astype(int)

    out.loc[mask_bad & out["v_corr"].notna(), variable] = out.loc[mask_bad & out["v_corr"].notna(), "v_corr"]
    out.loc[mask_bad & out["v_corr"].isna(), variable] = np.nan

    out.drop(columns=["is_bad", "v_orig", "v_corr"], inplace=True)

    return out


def agregar_drift_features_monetarias(
    df: pd.DataFrame,
    columnas_monetarias: list[str],
    id_col: str = "numero_de_cliente",
    mes_col: str = "foto_mes",
) -> pd.DataFrame:
    """
    Para cada variable monetaria v:
      - v_rz_mes: robust z-score por mes sobre signed-log1p(v) usando mediana e IQR (Q75-Q25)

    Importante:
      - Si v es NULL -> v_rz_mes queda NULL (preserva missingness)
      - Si IQR (q75-q25) es 0 o NULL -> v_rz_mes cae a 0 SOLO si v no es NULL
    """
    if not columnas_monetarias:
        return df

    cols = [c for c in columnas_monetarias if c in df.columns]
    if not cols:
        return df

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)

        slog_exprs = []
        for v in cols:
            qv = _qident(v)
            slog_exprs.append(
                f"""
                CASE
                  WHEN {qv} IS NULL THEN NULL
                  WHEN {qv} >= 0 THEN log(1 + {qv})
                  ELSE -log(1 + abs({qv}))
                END AS "{v}_slog1p"
                """
            )

        stats_exprs = []
        for v in cols:
            stats_exprs.append(f'approx_quantile("{v}_slog1p", 0.50) AS "{v}_med"')
            stats_exprs.append(f'approx_quantile("{v}_slog1p", 0.25) AS "{v}_q25"')
            stats_exprs.append(f'approx_quantile("{v}_slog1p", 0.75) AS "{v}_q75"')

        rz_exprs = []
        for v in cols:
            rz_exprs.append(
                f"""
                CASE
                  WHEN "{v}" IS NULL THEN NULL
                  ELSE COALESCE(
                    ("{v}_slog1p" - "{v}_med") / NULLIF(("{v}_q75" - "{v}_q25"), 0),
                    0
                  )
                END AS "{v}_rz_mes"
                """
            )

        query = f"""
        WITH t1 AS (
          SELECT
            *,
            {", ".join(slog_exprs)}
          FROM df
        ),
        stats AS (
          SELECT
            "{mes_col}",
            {", ".join(stats_exprs)}
          FROM t1
          GROUP BY "{mes_col}"
        )
        SELECT
          t1.* EXCLUDE ({", ".join([f'"{v}_slog1p"' for v in cols])}),
          {", ".join(rz_exprs)}
        FROM t1
        JOIN stats USING("{mes_col}")
        """

        return con.execute(query).df()
    finally:
        con.close()


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
    - Quotea columnas para soportar mayúsculas (Master_*, Visa_*) y nombres raros.
    """
    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if not columnas:
        logger.warning("No se especificaron atributos para generar lags")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"[lag] La columna {attr} no existe, se omite")
            continue

        q_attr = _qident(attr)
        for i in range(1, cant_lag + 1):
            col_name = f"{attr}_lag_{i}"
            if col_name in df.columns:
                logger.warning(f"[lag] La columna {col_name} ya existe, no se vuelve a generar")
                continue

            sql += (
                f", lag({q_attr}, {i}) OVER ("
                f"PARTITION BY {_qident('numero_de_cliente')} "
                f"ORDER BY {_qident('foto_mes')}) AS {_qident(col_name)}"
            )

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Feature engineering lag completado. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_deltas(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera columnas de delta en SQL: valor actual menos lag correspondiente.
    Mantiene NULL si no hay información suficiente.
    - Quotea columnas para soportar mayúsculas.
    """
    logger.info(f"Generando deltas (SQL) para {len(columnas) if columnas else 0} columnas con {cant_lag} lags")

    if not columnas:
        logger.warning("No se especificaron atributos para generar deltas")
        return df

    sql = "SELECT *"
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"[delta] La columna {attr} no existe, se omite")
            continue

        q_attr = _qident(attr)
        for i in range(1, cant_lag + 1):
            lag_col = f"{attr}_lag_{i}"
            delta_col = f"{attr}_delta_{i}"

            if lag_col not in df.columns:
                logger.warning(f"[delta] Falta {lag_col}, no se puede generar {delta_col}")
                continue
            if delta_col in df.columns:
                logger.warning(f"[delta] {delta_col} ya existe, no se genera nuevamente")
                continue

            sql += f", ({q_attr} - {_qident(lag_col)}) AS {_qident(delta_col)}"

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
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
        if attr not in df.columns:
            logger.warning(f"[ma] La columna {attr} no existe, se omite")
            continue

        ma_col_name = f"{attr}_ma_{window_size}"
        if ma_col_name not in df.columns:
            sql += (
                f', CASE '
                f'WHEN COUNT({_qident(attr)}) OVER (PARTITION BY {_qident("numero_de_cliente")} '
                f'ORDER BY {_qident("foto_mes")} '
                f'ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW) = {window_size} '
                f'THEN AVG({_qident(attr)}) OVER (PARTITION BY {_qident("numero_de_cliente")} '
                f'ORDER BY {_qident("foto_mes")} '
                f'ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW) '
                f'ELSE NULL END AS {_qident(ma_col_name)}'
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


def feature_engineering_cum_sum(
    df: pd.DataFrame,
    columnas: list[str],
    window_size: int | None = None,   # ej: 6
    strict: bool = False,             # si True, requiere ventana completa
) -> pd.DataFrame:
    """
    Genera columnas de suma acumulada por cliente (solo hacia atrás).

    - window_size=None: histórico completo (UNBOUNDED PRECEDING)
    - window_size=6: últimos 6 registros (5 PRECEDING + current)
    - strict=True: si no hay 6 filas previas+actual, deja NULL
    """
    if not columnas:
        logger.warning("No se especificaron atributos para cumsum")
        return df

    frame = (
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        if window_size is None
        else f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
    )

    sql = "SELECT *"
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"[cumsum] La columna {attr} no existe, se omite")
            continue

        colname = f"{attr}_cumsum" if window_size is None else f"{attr}_cumsum_{window_size}"
        if colname in df.columns:
            logger.warning(f"{colname} ya existe, no se vuelve a generar")
            continue

        base_sum = (
            f'SUM(COALESCE({_qident(attr)}, 0)) OVER ('
            f'  PARTITION BY {_qident("numero_de_cliente")} '
            f'  ORDER BY {_qident("foto_mes")} '
            f'  {frame}'
            f')'
        )

        if strict and window_size is not None:
            sql += (
                f', CASE WHEN COUNT(1) OVER ('
                f'  PARTITION BY {_qident("numero_de_cliente")} '
                f'  ORDER BY {_qident("foto_mes")} '
                f'  {frame}'
                f') = {window_size} THEN {base_sum} ELSE NULL END AS {_qident(colname)}'
            )
        else:
            sql += f', {base_sum} AS {_qident(colname)}'

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Feature engineering cumsum completado. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_min_max(
    df: pd.DataFrame,
    columnas: list[str],
    window_size: int | None = None,   # ej: 6
    strict: bool = False,
) -> pd.DataFrame:
    """
    Genera min/max por cliente HASTA CADA MES (sin mirar el futuro).

    - window_size=None: min/max histórico
    - window_size=6: min/max de los últimos 6 registros
    - strict=True: si no hay 6 filas en ventana, deja NULL
    """
    logger.info(
        f"Generando min/max {'históricos' if window_size is None else f'window {window_size}'} "
        f"para {len(columnas) if columnas else 0} atributos"
    )

    if not columnas:
        logger.warning("No se especificaron atributos para generar min y max")
        return df

    frame = (
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        if window_size is None
        else f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
    )

    sql = "SELECT *"
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"[minmax] La columna {attr} no existe en el DataFrame, se omite")
            continue

        suf = "hist" if window_size is None else str(window_size)
        col_min = f"{attr}_min_{suf}"
        col_max = f"{attr}_max_{suf}"

        min_expr = (
            f'MIN({_qident(attr)}) OVER ('
            f'  PARTITION BY {_qident("numero_de_cliente")} '
            f'  ORDER BY {_qident("foto_mes")} '
            f'  {frame}'
            f')'
        )
        max_expr = (
            f'MAX({_qident(attr)}) OVER ('
            f'  PARTITION BY {_qident("numero_de_cliente")} '
            f'  ORDER BY {_qident("foto_mes")} '
            f'  {frame}'
            f')'
        )

        if strict and window_size is not None:
            cnt_expr = (
                f'COUNT(1) OVER ('
                f'  PARTITION BY {_qident("numero_de_cliente")} '
                f'  ORDER BY {_qident("foto_mes")} '
                f'  {frame}'
                f')'
            )
            if col_min not in df.columns:
                sql += f', CASE WHEN {cnt_expr} = {window_size} THEN {min_expr} ELSE NULL END AS {_qident(col_min)}'
            if col_max not in df.columns:
                sql += f', CASE WHEN {cnt_expr} = {window_size} THEN {max_expr} ELSE NULL END AS {_qident(col_max)}'
        else:
            if col_min not in df.columns:
                sql += f', {min_expr} AS {_qident(col_min)}'
            if col_max not in df.columns:
                sql += f', {max_expr} AS {_qident(col_max)}'

    sql += " FROM df"

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(sql).df()
    finally:
        con.close()

    logger.info(f"Min/max generados. DataFrame resultante con {df_out.shape[1]} columnas")
    return df_out


def feature_engineering_ratios(df: pd.DataFrame, ratio_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """
    Genera columnas de ratios entre pares válidos de columnas.
    - Si numerador es NULL -> ratio NULL
    - Si denominador es NULL o 0 -> ratio NULL
    """
    if not ratio_pairs:
        logger.warning("No se especificaron pares de columnas para generar ratios")
        return df

    sql = "SELECT *"
    for numerador, denominador in ratio_pairs:
        if numerador not in df.columns or denominador not in df.columns:
            logger.warning(f"[ratio] Columnas no encontradas: {numerador} o {denominador}, se omite")
            continue

        ratio_col = f"{numerador}_over_{denominador}"
        if ratio_col in df.columns:
            logger.warning(f"[ratio] {ratio_col} ya existe, no se vuelve a generar")
            continue

        q_num = _qident(numerador)
        q_den = _qident(denominador)
        q_ratio = _qident(ratio_col)

        sql += f""",
            CASE
              WHEN {q_num} IS NULL THEN NULL
              WHEN {q_den} IS NULL OR {q_den} = 0 THEN NULL
              ELSE (1.0 * {q_num}) / {q_den}
            END AS {q_ratio}
        """

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
        if attr not in df.columns:
            logger.warning(f"[ma_lag] La columna {attr} no existe, se omite")
            continue

        ma_col_name = f"{attr}_ma_lag_{window_size}"
        if ma_col_name not in df.columns:
            sql += (
                f', CASE '
                f'WHEN COUNT({_qident(attr)}) OVER (PARTITION BY {_qident("numero_de_cliente")} '
                f'ORDER BY {_qident("foto_mes")} '
                f'ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING) = {window_size} '
                f'THEN AVG({_qident(attr)}) OVER (PARTITION BY {_qident("numero_de_cliente")} '
                f'ORDER BY {_qident("foto_mes")} '
                f'ROWS BETWEEN {window_size} PRECEDING AND 1 PRECEDING) '
                f'ELSE NULL END AS {_qident(ma_col_name)}'
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

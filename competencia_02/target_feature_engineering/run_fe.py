import pandas as pd
import os
import datetime
import logging

from src.data_load_preparation import cargar_datos
from .features import feature_engineering_lag, feature_engineering_min_max, feature_engineering_deltas, feature_engineering_medias_moviles, feature_engineering_cum_sum,         feature_engineering_ratios, feature_engineering_medias_moviles_lag, generar_shock_relativo_delta_lag, crear_indicador_aguinaldo



## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
monbre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{monbre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs("datos", exist_ok=True)
    path = "datos/competencia_01.csv"
    df = cargar_datos(path)

    df.drop(columns=["mprestamos_personales", "cprestamos_personales"], inplace=True)   

    #01 Feature Engineering
    atributos = [
    "mrentabilidad",
    "mcomisiones",
    "mpasivos_margen",
    "mcaja_ahorro",
    "mcaja_ahorro_dolares",
    "mcuentas_saldo",
    "ctarjeta_debito_transacciones",
    "mautoservicio",
    "ctarjeta_visa_transacciones",
    "mtarjeta_visa_consumo",
    "ctarjeta_master_transacciones",
    "mtarjeta_master_consumo",
    "mprestamos_prendarios",
    "mprestamos_hipotecarios",
    "mplazo_fijo_dolares",
    "mplazo_fijo_pesos",
    "cpayroll_trx",
    "cpayroll2_trx",
    "mcomisiones_mantenimiento",
    "ctrx_quarter",
    "Master_mlimitecompra",
    "Master_mconsumototal",
    "Visa_mlimitecompra",
    "Visa_mconsumototal",
    "mactivos_margen",
    "mcuenta_corriente",
    "mcuenta_corriente_adicional",
    "mcaja_ahorro_adicional",
    "minversion1_pesos",
    "minversion1_dolares",
    "minversion2"
    ]


    atributos_fe_lag = atributos 
    atributo_fe_deltas = atributos
    atributos_fe_medias_moviles = atributos
    atributos_cum_sum = atributos
    atributos_min_max = atributos

    ratio_pairs = [
    # ya definidos
    ("Master_mconsumototal", "Master_mlimitecompra"),
    ("Visa_mconsumototal", "Visa_mlimitecompra"),
    ("mtarjeta_master_consumo", "Master_mlimitecompra"),
    ("mtarjeta_visa_consumo", "Visa_mlimitecompra"),
    ("mprestamos_prendarios", "mcuentas_saldo"),
    ("mprestamos_hipotecarios", "mcuentas_saldo"),
    ("mcaja_ahorro", "mcuentas_saldo"),
    ("mcaja_ahorro_dolares", "mcuentas_saldo"),
    ("mcomisiones", "mrentabilidad"),
    
    # nuevos sugeridos
    ("mcuenta_corriente", "mcuentas_saldo"),
    ("mcuenta_corriente_adicional", "mcuentas_saldo"),
    ("minversion1_pesos", "mcuentas_saldo"),
    ("minversion1_dolares", "mcuentas_saldo"),
    ("minversion2", "mcuentas_saldo"),
    ("mactivos_margen", "mpasivos_margen"),  # margen activo/pasivo
    ("mcomisiones_mantenimiento", "mcomisiones"),  # proporción de mantenimiento sobre comisiones totales
    ("mtarjeta_master_consumo", "mcuentas_saldo"),
    ("mtarjeta_visa_consumo", "mcuentas_saldo")
    ]


    

    cant_lag = 2
    window_size = 2
    
    df_fe = feature_engineering_lag(df, columnas=atributos_fe_lag, cant_lag=cant_lag)
    df_fe = feature_engineering_deltas(df_fe, columnas=atributo_fe_deltas, cant_lag=cant_lag)
    df_fe = feature_engineering_medias_moviles(df_fe, columnas=atributos_fe_medias_moviles, window_size=window_size)
    
    df_fe = feature_engineering_medias_moviles_lag(df_fe, columnas=atributos_fe_medias_moviles, window_size=window_size)

    
    df_fe = generar_shock_relativo_delta_lag(df_fe, columnas=atributos_fe_medias_moviles, window_size=window_size)
    # df_fe = feature_engineering_cum_sum(df_fe, columnas=atributos_cum_sum)
    # df_fe = feature_engineering_min_max(df_fe, columnas=atributos_min_max)
    df_fe = feature_engineering_ratios(df_fe, ratio_pairs=ratio_pairs)
    df_fe = crear_indicador_aguinaldo(df_fe)
  
    #02 Guardar datos
    path = "datos/competencia_02_FE_v1.csv"
    df_fe.to_csv(path, index=False)

    #03 Convertir clase_ternaria a target binario
    # df_fe = convertir_clase_ternaria_a_target(df_fe)

  
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.{monbre_log}")

if __name__ == "__main__":
    main()
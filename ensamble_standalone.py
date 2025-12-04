#!/usr/bin/env python3
"""
Script standalone para reproducir mi envío.

1. Descarga el dataset de Google Cloud Storage
2. Ejecuta Config 1 (ensamble de ensambles)
3. Ejecuta Config 2 (ensamble a secas)
4. Ensambla las predicciones finales (promedio simple)
5. Selecciona top 11000 clientes
6. Guarda los numero_de_cliente a estimular
"""

import os
import sys
import gc
import logging
import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from google.cloud import storage

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# URL del dataset en Google Cloud
DATASET_GCS_URL = "gs://dosdesvios_bukito3/competencia_03/data/03_v2.parquet"

# Path local donde descargar el dataset
LOCAL_DATA_DIR = "./data"
LOCAL_DATASET_PATH = os.path.join(LOCAL_DATA_DIR, "03_v2.parquet")

VAL_MONTH = [202109]

N_SUBMISSIONS = 11000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG 1: modelo_por_año_sept_ok
# ============================================================================

CONFIG_1 = {
    'experiment_name': 'modelo_por_año_sept_ok',
    'n_experiments': 1,
    'val_month': [202109],
    'fixed_params': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'force_row_wise': True,
        'boost_from_average': True,
        'feature_pre_filter': True,
        'max_bin': 31
    },
    'model_2019': {
        'params': {
            'learning_rate': 0.04238779286,
            'feature_fraction_bynode': 0.9279856643,
            'feature_fraction': 0.09961437125,
            'min_data_in_leaf': 602,
            'bagging_freq': 0,
            'num_boost_round': 394,
            'num_leaves': 215
        },
        'semillerio': 10,
        'n_submissions': 11000,
        'sub_early_stop': 3,
        'months': [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201911, 201912],
        'undersampling_fraction': 0.31,
        'chosen_features': ['seleccion_500']
    },
    'model_2020': {
        'params': {
            'learning_rate': 0.04238779286,
            'feature_fraction_bynode': 0.9279856643,
            'feature_fraction': 0.09961437125,
            'min_data_in_leaf': 602,
            'bagging_freq': 0,
            'num_boost_round': 394,
            'num_leaves': 215
        },
        'semillerio': 10,
        'n_submissions': 11000,
        'sub_early_stop': 3,
        'months': [202001, 202003, 202004, 202005, 202007, 202008, 202009, 202010, 202011, 202012],
        'undersampling_fraction': 0.31,
        'chosen_features': ['seleccion_500']
    },
    'model_2021': {
        'params': {
            'learning_rate': 0.04238779286,
            'feature_fraction_bynode': 0.9279856643,
            'feature_fraction': 0.09961437125,
            'min_data_in_leaf': 602,
            'bagging_freq': 0,
            'num_boost_round': 394,
            'num_leaves': 215
        },
        'semillerio': 10,
        'n_submissions': 11000,
        'sub_early_stop': 3,
        'months': [202101, 202102, 202103, 202104, 202105, 202106, 202107],
        'undersampling_fraction': 1,
        'chosen_features': ['seleccion_500']
    }
}

# ============================================================================
# CONFIG 2: envio_septiembre_ok_v2
# ============================================================================

CONFIG_2 = {
    'experiment_name': 'envio_septiembre_ok_v2',
    'n_experiments': 1,
    'val_month': [202109],
    'fixed_params': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'force_row_wise': True,
        'boost_from_average': True,
        'feature_pre_filter': True,
        'max_bin': 31
    },
    'model_804': {
        'params': {
            'learning_rate': 0.016307703463148453,
            'feature_fraction': 0.0706163273701218,
            'min_data_in_leaf': 449,
            'bagging_freq': 3,
            'neg_bagging_fraction': 0.2,
            'pos_bagging_fraction': 0.29860597845871484,
            'num_boost_round': 804,
            'num_leaves': 775
        },
        'semillerio': 20,
        'sub_early_stop': 2,
        'n_submissions': 11000,
        'months': [201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 201911, 201912, 202001, 202003, 202004, 202005, 202007, 202008, 202009, 202010, 202011, 202012, 202101, 202102, 202103, 202104, 202105, 202106, 202107],
        'undersampling_fraction': 0.5,
        'chosen_features': ['seleccion_730']
    }
}

# ============================================================================
# FEATURES
# ============================================================================

FEATURES_SELECCION_500 = [
    'mcaja_ahorro', 'rankp_ctrx_quarter', 'rankp_mcaja_ahorro', 'rankn_ctrx_quarter',
    'ctrx_quarter_normalizado', 'ctrx_quarter', 'rankp_mtarjeta_visa_consumo', 'rankn_mcaja_ahorro',
    'mtarjeta_visa_consumo', 'ctrx_quarter_normalizado_lag1', 'ctrx_quarter_lag1',
    'rankp_ctarjeta_visa_transacciones', 'rankn_mtarjeta_visa_consumo', 'cpayroll_trx',
    'ctarjeta_visa_transacciones', 'mpayroll', 'mcuentas_saldo', 'mpayroll_sobre_edad',
    'mcaja_ahorro_lag1', 'mautoservicio', 'rankn_mpasivos_margen', 'rankp_ctarjeta_visa',
    'rankn_ctarjeta_visa_transacciones', 'rankp_cpayroll_trx', 'rankp_mprestamos_personales',
    'Visa_mpagospesos', 'mcaja_ahorro_min6', 'foto_mes', 'rankp_mcuentas_saldo',
    'rankp_ccaja_seguridad', 'rankp_ccaja_seguridad_1', 'cproductos_ratioavg6', 'mpasivos_margen',
    'Visa_status', 'rankn_cpayroll_trx', 'cdescubierto_preacordado_tend6', 'rankn_mpayroll',
    'ctarjeta_debito_transacciones', 'cpayroll_trx_lag1', 'mcaja_ahorro_ratioavg6',
    'rankp_mpasivos_margen', 'Visa_msaldopesos', 'rankn_mcuentas_saldo', 'ctrx_quarter_normalizado_lag2',
    'rankp_cprestamos_personales', 'rankp_Visa_cconsumos', 'ctarjeta_visa', 'Visa_Finiciomora',
    'rankp_mpayroll', 'rankp_mcuenta_corriente', 'cdescubierto_preacordado_ratioavg6', 'Visa_msaldototal',
    'mpayroll_sobre_edad_lag1', 'mcaja_ahorro_max6', 'rankp_ccaja_ahorro', 'mcuenta_corriente',
    'cpayroll_trx_lag2', 'cproductos_delta2', 'Visa_mpagospesos_lag1', 'mcaja_ahorro_lag2',
    'ctrx_quarter_lag2', 'rankp_cdescubierto_preacordado', 'Visa_mconsumospesos', 'rankp_Visa_msaldototal',
    'mpayroll_lag1', 'cdescubierto_preacordado', 'rankn_mprestamos_personales', 'mpayroll_sobre_edad_lag2',
    'rankp_cextraccion_autoservicio', 'mprestamos_personales', 'Visa_mpagominimo', 'cdescubierto_preacordado_delta2',
    'rankp_cproductos', 'mpayroll_lag2', 'rankn_mcuenta_corriente', 'cdescubierto_preacordado_delta3',
    'cproductos_tend6', 'rankp_mextraccion_autoservicio', 'rankp_Visa_msaldopesos', 'Master_status',
    'rankp_Visa_mconsumospesos', 'rankn_Visa_mconsumospesos', 'rankp_mautoservicio', 'rankp_Visa_mpagominimo',
    'Visa_delinquency', 'cproductos_delta3', 'ccomisiones_mantenimiento_delta3', 'rankn_Visa_msaldopesos',
    'rankp_Visa_mconsumosdolares', 'ccomisiones_mantenimiento_delta2', 'mtransferencias_recibidas',
    'Visa_status_delta2', 'rankp_cproductos_1', 'mprestamos_personales_lag1', 'ctarjeta_visa_delta2',
    'rankp_ccajas_consultas', 'mcuenta_corriente_tend6', 'cproductos_delta1', 'Visa_mconsumototal',
    'Master_Finiciomora', 'Visa_Finiciomora_delta1', 'ctrx_quarter_lag3', 'rankn_mautoservicio',
    'ctarjeta_visa_delta3', 'cprestamos_personales_delta3', 'ctarjeta_master_delta2', 'ctrx_quarter_delta1',
    'rankn_Visa_mconsumototal', 'cprestamos_personales', 'mtarjeta_visa_consumo_lag1', 'Visa_status_delta1',
    'mcuentas_saldo_lag1', 'rankp_tcallcenter', 'mpayroll_delta3', 'mpasivos_margen_max6',
    'mprestamos_personales_lag2', 'mcuentas_saldo_min6', 'rankp_Visa_mconsumospesos_1', 'mcomisiones_mantenimiento_delta3',
    'rankp_Visa_mfinanciacion_limite_1', 'mprestamos_personales_lag3', 'mprestamos_personales_delta3',
    'rankn_Visa_mpagominimo', 'cdescubierto_preacordado_min6', 'mprestamos_personales_delta2',
    'rankn_cprestamos_personales', 'mpasivos_margen_lag1', 'ctarjeta_master_delta3', 'mactivos_margen',
    'cpayroll_trx_lag3', 'rankn_Visa_msaldototal', 'rankp_ccajas_otras', 'Visa_Finiciomora_lag1',
    'ctarjeta_visa_delta1', 'rankp_ccallcenter_transacciones', 'mprestamos_personales_delta1',
    'ctrx_quarter_normalizado_delta1', 'mcuentas_saldo_tend6', 'mcomisiones_min6', 'Master_status_delta2',
    'ctrx_quarter_delta2', 'mcaja_ahorro_lag3', 'rankp_Visa_mconsumototal', 'rankn_Visa_cconsumos',
    'rankn_Visa_mconsumospesos_1', 'mpayroll_delta2', 'rankp_Visa_mfinanciacion_limite', 'mcuenta_corriente_delta3',
    'ctrx_quarter_normalizado_delta2', 'mpayroll_sobre_edad_delta3', 'rankp_mtarjeta_master_consumo',
    'mtarjeta_visa_consumo_delta3', 'Master_delinquency', 'mcuenta_corriente_ratioavg6', 'mtarjeta_master_consumo',
    'mpayroll_sobre_edad_lag3', 'ctarjeta_master_transacciones', 'cdescubierto_preacordado_delta1',
    'rankn_mtarjeta_master_consumo', 'ccomisiones_mantenimiento_delta1', 'rankn_ctarjeta_visa',
    'cprestamos_personales_delta2', 'mcuentas_saldo_delta3', 'rankp_Visa_mpagospesos', 'ctrx_quarter_normalizado_lag3',
    'ctarjeta_master', 'mrentabilidad_annual_min6', 'mcuentas_saldo_delta2', 'rankp_ccajas_transacciones',
    'Visa_cconsumos', 'mcomisiones_otras_lag3', 'rankp_Visa_status', 'mrentabilidad_annual_lag2',
    'mpayroll_lag3', 'cextraccion_autoservicio', 'Visa_mpagominimo_lag1', 'rankp_mactivos_margen',
    'mcuenta_corriente_delta1', 'mextraccion_autoservicio', 'mpayroll_sobre_edad_delta2', 'Visa_status_delta3',
    'rankn_cdescubierto_preacordado', 'Master_status_delta3', 'mcomisiones_lag3', 'mrentabilidad_annual_lag3',
    'mcuentas_saldo_max6', 'rankn_cextraccion_autoservicio', 'rankn_cproductos', 'Visa_mfinanciacion_limite',
    'rankp_thomebanking', 'rankn_Visa_mpagospesos', 'Visa_msaldototal_lag1', 'mcuentas_saldo_ratioavg6',
    'mautoservicio_lag2', 'mcuenta_corriente_delta2', 'rankp_Master_msaldototal', 'ccaja_ahorro_tend6',
    'mtarjeta_visa_consumo_lag2', 'ccaja_ahorro', 'mtarjeta_visa_consumo_delta2', 'ctarjeta_debito_transacciones_lag1',
    'mcuenta_corriente_lag1', 'rankn_mactivos_margen', 'rankp_Master_fultimo_cierre', 'mpasivos_margen_lag2',
    'rankp_mplazo_fijo_dolares', 'Visa_delinquency_delta2', 'rankp_mttarjeta_visa_debitos_automaticos',
    'Visa_mconsumospesos_lag1', 'ctarjeta_visa_transacciones_delta2', 'ctarjeta_master_delta1',
    'rankp_mcuenta_debitos_automaticos_1', 'rankp_ccomisiones_mantenimiento', 'mrentabilidad_annual_lag1',
    'mcomisiones_mantenimiento_delta1', 'catm_trx', 'mpagomiscuentas', 'mactivos_margen_lag2',
    'mactivos_margen_min6', 'mcomisiones_mantenimiento_delta2', 'rankp_mcuenta_debitos_automaticos',
    'rankp_mcomisiones_mantenimiento_1', 'Visa_delinquency_lag1', 'rankp_ccomisiones_otras',
    'mpasivos_margen_tend6', 'Visa_mpagosdolares_delta1', 'ctrx_quarter_normalizado_delta3', 'Visa_fechaalta',
    'Master_status_delta1', 'mcomisiones_ratioavg6', 'ctrx_quarter_delta3', 'mpasivos_margen_min6',
    'ccaja_ahorro_max6', 'rankn_cproductos_1', 'rankp_ctarjeta_visa_debitos_automaticos', 'cprestamos_personales_lag1',
    'mcaja_ahorro_tend6', 'mactivos_margen_lag1', 'mrentabilidad_annual_max6', 'Visa_mpagominimo_delta1',
    'cextraccion_autoservicio_lag1', 'ccaja_ahorro_ratioavg6', 'rankn_mrentabilidad_annual',
    'rankp_mcomisiones_mantenimiento', 'mcuentas_saldo_delta1', 'cprestamos_personales_lag2', 'Master_mpagominimo',
    'cproductos', 'mcomisiones_otras_lag2', 'mcomisiones_lag2', 'rankp_ccuenta_debitos_automaticos',
    'Master_fechaalta', 'ccallcenter_transacciones_delta2', 'mpasivos_margen_ratioavg6', 'cmobile_app_trx_lag2',
    'mtarjeta_visa_consumo_delta1', 'Master_mpagospesos', 'mcuentas_saldo_lag2', 'Visa_msaldopesos_lag1',
    'mrentabilidad_annual', 'cmobile_app_trx_lag3', 'Master_msaldopesos', 'Visa_mpagominimo_delta2',
    'ctarjeta_visa_transacciones_delta3', 'Visa_mpagominimo_lag2', 'rankp_ctransferencias_emitidas',
    'mcaja_ahorro_delta2', 'mcuenta_corriente_min6', 'rankp_tmobile_app', 'cmobile_app_trx_delta3',
    'rankn_mextraccion_autoservicio', 'kmes', 'mactivos_margen_ratioavg6', 'rankp_Visa_mpagado',
    'Visa_delinquency_delta3', 'rankp_thomebanking_1', 'ccaja_seguridad', 'mcuenta_debitos_automaticos',
    'rankp_ccomisiones_otras_1', 'mcomisiones_mantenimiento', 'cmobile_app_trx_delta2', 'Master_delinquency_delta2',
    'rankp_mtransferencias_recibidas_1', 'cprestamos_personales_lag3', 'rankp_mpagomiscuentas', 'Master_msaldototal',
    'ccomisiones_mantenimiento', 'ctarjeta_visa_transacciones_lag1', 'mautoservicio_lag1', 'internet_max6',
    'matm', 'ctarjeta_visa_lag3', 'mttarjeta_visa_debitos_automaticos_delta3', 'ccallcenter_transacciones_delta3',
    'mcomisiones_mantenimiento_lag3', 'rankp_Visa_mlimitecompra', 'mtransferencias_recibidas_lag1',
    'mpayroll_delta1', 'internet_tend6', 'ccaja_ahorro_delta2', 'ccaja_ahorro_delta1', 'rankp_mtransferencias_recibidas',
    'Visa_mpagominimo_delta3', 'rankn_Visa_status', 'rankp_Master_mpagominimo', 'ctarjeta_debito_transacciones_min6',
    'rankp_mtransferencias_emitidas', 'tcallcenter', 'cmobile_app_trx_lag1', 'rankp_Master_msaldopesos',
    'tmobile_app_lag2', 'rankn_mpagomiscuentas', 'rankn_ccaja_ahorro', 'mrentabilidad_min6',
    'cprestamos_personales_delta1', 'Visa_msaldodolares_lag1', 'cmobile_app_trx', 'rankp_mrentabilidad_annual',
    'tmobile_app', 'ctarjeta_debito_transacciones_lag2', 'rankp_Master_mconsumospesos', 'tcuentas',
    'tcallcenter_delta2', 'mcaja_ahorro_dolares_ratioavg6', 'ccaja_ahorro_delta3', 'ctarjeta_visa_transacciones_delta1',
    'rankp_cpagomiscuentas', 'Visa_fultimo_cierre_lag2', 'Master_Fvencimiento', 'mrentabilidad', 'mcomisiones_delta3',
    'mpasivos_margen_lag3', 'ccomisiones_mantenimiento_lag3', 'active_quarter_min6', 'Master_delinquency_lag1',
    'cmobile_app_trx_delta1', 'Visa_mpagominimo_lag3', 'matm_lag1', 'mcaja_ahorro_delta1', 'Master_delinquency_delta3',
    'mcomisiones_otras_delta3', 'cpagomiscuentas', 'mcomisiones_tend6', 'Master_Finiciomora_lag1',
    'mcaja_ahorro_dolares_delta2', 'cproductos_min6', 'mcaja_ahorro_delta3', 'mcaja_ahorro_dolares_delta3',
    'mcuenta_corriente_lag2', 'rankn_mcomisiones_mantenimiento', 'mrentabilidad_lag2', 'mcuenta_corriente_max6',
    'mpasivos_margen_delta1', 'rankp_Master_cconsumos', 'rankp_ctransferencias_recibidas', 'active_quarter',
    'Master_fechaalta_lag1', 'cdescubierto_preacordado_lag1', 'tcallcenter_delta3', 'rankn_mcomisiones_mantenimiento_1',
    'mpayroll_sobre_edad_delta1', 'internet_delta3', 'mrentabilidad_max6', 'rankn_ccomisiones_mantenimiento',
    'Visa_cconsumos_delta2', 'tmobile_app_lag1', 'ccallcenter_transacciones', 'rankn_mcuenta_debitos_automaticos_1',
    'Visa_fechaalta_lag1', 'ctransferencias_recibidas', 'mcaja_ahorro_dolares_delta1', 'rankp_cmobile_app_trx',
    'internet_lag2', 'Visa_msaldopesos_lag2', 'Master_mfinanciacion_limite', 'ccaja_seguridad_lag1',
    'Visa_madelantopesos_lag1', 'ccaja_seguridad_lag2', 'Visa_mconsumototal_delta1', 'active_quarter_lag1',
    'mactivos_margen_lag3', 'Visa_Fvencimiento', 'ccallcenter_transacciones_delta1', 'rankp_chomebanking_transacciones',
    'rankn_ccomisiones_otras', 'rankn_mtransferencias_recibidas', 'cpayroll_trx_delta3', 'tmobile_app_delta3',
    'rankn_mcuenta_debitos_automaticos', 'mcomisiones_mantenimiento_lag2', 'cpayroll_trx_delta2', 'mrentabilidad_lag3',
    'mpasivos_margen_delta3', 'mextraccion_autoservicio_lag2', 'internet_ratioavg6', 'mextraccion_autoservicio_lag3',
    'rankn_ccallcenter_transacciones', 'mtarjeta_master_consumo_lag1', 'Visa_cconsumos_delta3',
    'mtransferencias_recibidas_lag2', 'Master_Finiciomora_delta1', 'Visa_cconsumos_lag2', 'ccomisiones_otras_lag3',
    'rankp_chomebanking_transacciones_1', 'ctarjeta_visa_lag1', 'cproductos_lag1', 'Visa_fultimo_cierre_delta1',
    'tmobile_app_lag3', 'mrentabilidad_lag1', 'rankn_Visa_mfinanciacion_limite_1', 'Visa_mlimitecompra',
    'ctarjeta_visa_debitos_automaticos_delta3', 'rankp_Master_mlimitecompra_1', 'rankp_mrentabilidad', 'mcomisiones_lag1',
    'Visa_msaldototal_delta3', 'Visa_msaldopesos_delta3', 'rankn_thomebanking', 'mttarjeta_visa_debitos_automaticos_delta2',
    'mcomisiones_max6', 'Master_mpagominimo_delta1', 'cliente_antiguedad_min6', 'ccaja_ahorro_lag1', 'ccomisiones_otras',
    'ccajas_consultas_delta2', 'internet_lag1', 'mactivos_margen_tend6', 'Visa_mconsumototal_lag1',
    'rankn_Master_mpagominimo', 'mcomisiones_delta2', 'Visa_fultimo_cierre', 'mcuenta_corriente_lag3',
    'rankn_Visa_mconsumosdolares', 'tcuentas_lag3', 'ctarjeta_visa_transacciones_lag2', 'mpasivos_margen_delta2',
    'mtarjeta_master_consumo_delta3', 'mrentabilidad_ratioavg6', 'rankn_mrentabilidad', 'rankp_Master_mfinanciacion_limite',
    'tcallcenter_delta1', 'Master_fultimo_cierre_delta2', 'ctarjeta_visa_lag2', 'rankp_Visa_Fvencimiento',
    'Visa_madelantopesos', 'ccajas_consultas_delta3', 'mcuentas_saldo_lag3', 'ctarjeta_visa_transacciones_lag3',
    'mcomisiones_otras_delta2', 'Visa_mpagospesos_delta3', 'Visa_madelantopesos_delta1', 'rankn_tcallcenter',
    'kmes_lag1', 'tcuentas_lag2', 'tmobile_app_delta1', 'rankp_Master_mpagospesos', 'Visa_cadelantosefectivo_lag1',
    'mcaja_ahorro_dolares_tend6', 'Master_mpagominimo_delta3', 'Master_mfinanciacion_limite_delta3',
    'rankp_mcaja_ahorro_dolares', 'internet_lag3', 'chomebanking_transacciones', 'thomebanking', 'Visa_Fvencimiento_lag1',
    'mcomisiones', 'Master_msaldototal_lag1', 'Visa_fechaalta_lag2', 'rankn_mtransferencias_recibidas_1',
    'Master_mfinanciacion_limite_delta1', 'rankp_Master_mfinanciacion_limite_1', 'Visa_fechaalta_lag3',
    'Visa_madelantodolares_lag1', 'rankn_Visa_Fvencimiento', 'tcuentas_lag1', 'mextraccion_autoservicio_lag1',
    'Visa_msaldototal_lag2', 'matm_other', 'mactivos_margen_delta3', 'tcuentas_max6', 'ccaja_seguridad_lag3',
    'Master_mpagominimo_delta2', 'rankn_Master_msaldopesos', 'cliente_antiguedad', 'ctarjeta_visa_debitos_automaticos_delta2',
    'ccajas_transacciones_delta3', 'mextraccion_autoservicio_delta3', 'mplazo_fijo_dolares', 'cliente_edad_min6',
    'Master_msaldopesos_lag2', 'Visa_msaldopesos_lag3', 'mcomisiones_otras_lag1', 'Visa_mconsumospesos_delta1',
    'mrentabilidad_delta3', 'Master_mpagominimo_lag1', 'Visa_status_lag1', 'Visa_mpagospesos_delta2',
    'rankp_Master_mlimitecompra', 'ccajas_transacciones'
]

FEATURES_SELECCION_730 = [
    'ctrx_quarter_normalizado', 'rankp_mcaja_ahorro', 'ctrx_quarter', 'rankn_mcaja_ahorro',
    'rankp_ctarjeta_visa_transacciones', 'rankp_ctrx_quarter', 'rankn_ctrx_quarter', 'cpayroll_trx',
    'mcaja_ahorro', 'ctarjeta_visa_transacciones', 'mpayroll_sobre_edad', 'mtarjeta_visa_consumo',
    'rankp_ctarjeta_visa', 'rankn_mtarjeta_visa_consumo', 'rankp_mtarjeta_visa_consumo',
    'mtarjeta_visa_consumo_ratioavg6', 'rankn_mpasivos_margen', 'ctarjeta_visa_ratioavg6',
    'rankp_mcuentas_saldo', 'mcaja_ahorro_lag1', 'ctrx_quarter_lag1', 'mcaja_ahorro_min6',
    'cproductos_ratioavg6', 'mpayroll', 'rankp_mpasivos_margen', 'rankp_Visa_msaldopesos',
    'rankn_mcuentas_saldo', 'ctrx_quarter_normalizado_lag1', 'mcuentas_saldo', 'rankn_mcuenta_corriente',
    'ctrx_quarter_ratioavg6', 'ctarjeta_visa_transacciones_ratioavg6', 'cdescubierto_preacordado_ratioavg6',
    'rankp_mcuenta_corriente', 'Visa_mconsumototal', 'Visa_mpagospesos', 'Visa_Finiciomora',
    'cdescubierto_preacordado', 'rankn_ctarjeta_visa_transacciones', 'mpasivos_margen_max6',
    'mcuenta_corriente', 'Visa_msaldopesos', 'rankp_cdescubierto_preacordado', 'mprestamos_personales',
    'rankp_cpayroll_trx', 'ctarjeta_master_ratioavg6', 'ccomisiones_mantenimiento_tend6', 'mcaja_ahorro_max6',
    'cdescubierto_preacordado_min6', 'rankp_mprestamos_personales', 'rankn_mpayroll', 'cproductos_tend6',
    'mpayroll_sobre_edad_lag1', 'mautoservicio', 'rankn_mprestamos_personales', 'mcomisiones_mantenimiento_ratioavg6',
    'rankp_Visa_mconsumospesos', 'ctarjeta_visa_delta2', 'mpayroll_ratioavg6', 'mcaja_ahorro_ratioavg6',
    'Master_status', 'cpayroll_trx_ratioavg6', 'Visa_mconsumospesos', 'Visa_delinquency', 'mprestamos_personales_ratioavg6',
    'foto_mes', 'mprestamos_personales_min6', 'mcuenta_corriente_tend6', 'mcuentas_saldo_lag1',
    'rankp_Visa_mpagominimo', 'ctarjeta_visa', 'ctarjeta_debito_transacciones', 'mcuentas_saldo_tend6',
    'cprestamos_personales_ratioavg6', 'rankp_ccaja_ahorro', 'mcuenta_corriente_ratioavg6',
    'rankp_cprestamos_personales', 'cproductos_delta1', 'mpayroll_lag1', 'Visa_msaldototal', 'mpayroll_lag2',
    'rankp_cproductos', 'rankn_cpayroll_trx', 'ctrx_quarter_lag2', 'rankp_mpayroll', 'cproductos_delta2',
    'mpayroll_sobre_edad_ratioavg6', 'mrentabilidad_annual_max6', 'rankp_tcallcenter',
    'ccomisiones_mantenimiento_ratioavg6', 'Visa_mpagospesos_ratioavg6', 'ctarjeta_master',
    'rankp_mextraccion_autoservicio', 'rankp_cproductos_1', 'cdescubierto_preacordado_delta2',
    'mautoservicio_ratioavg6', 'rankp_Visa_cconsumos', 'Visa_msaldototal_lag1', 'rankp_mactivos_margen',
    'Visa_status', 'mcuenta_corriente_delta2', 'mpasivos_margen', 'rankn_Visa_msaldototal', 'mcaja_ahorro_tend6',
    'ctrx_quarter_normalizado_ratioavg6', 'mpasivos_margen_tend6', 'mrentabilidad_annual', 'cpayroll_trx_lag2',
    'rankp_mautoservicio', 'mprestamos_personales_lag2', 'rankn_cproductos_1', 'rankp_mtarjeta_master_consumo',
    'rankp_Master_mconsumospesos', 'cproductos', 'Visa_delinquency_tend6', 'rankn_cdescubierto_preacordado',
    'rankn_cprestamos_personales', 'mcuentas_saldo_lag2', 'mtarjeta_visa_consumo_tend6', 'rankn_mautoservicio',
    'mprestamos_personales_delta1', 'rankn_mactivos_margen', 'mactivos_margen', 'rankp_Master_msaldopesos',
    'cpayroll_trx_lag1', 'mprestamos_personales_max6', 'rankp_ccallcenter_transacciones', 'Visa_mpagominimo',
    'ctarjeta_debito_transacciones_ratioavg6', 'ctarjeta_visa_transacciones_tend6', 'internet', 'mcuentas_saldo_ratioavg6',
    'rankp_ccaja_seguridad_1', 'ctrx_quarter_normalizado_delta1', 'mextraccion_autoservicio',
    'rankp_mcomisiones_mantenimiento_1', 'mcomisiones_mantenimiento_delta2', 'ccomisiones_mantenimiento',
    'ctrx_quarter_tend6', 'mprestamos_personales_lag1', 'Master_delinquency', 'mcomisiones_mantenimiento_tend6',
    'mcuenta_corriente_min6', 'mpayroll_sobre_edad_lag2', 'cdescubierto_preacordado_tend6', 'rankn_Visa_mpagospesos',
    'Visa_status_tend6', 'rankp_mcuenta_debitos_automaticos', 'rankp_ccajas_consultas', 'rankp_Master_msaldototal',
    'Master_status_tend6', 'rankn_Visa_msaldopesos', 'mrentabilidad_annual_lag2', 'rankp_ccomisiones_mantenimiento',
    'mtarjeta_master_consumo', 'rankp_Visa_msaldototal', 'mpayroll_tend6', 'Visa_delinquency_ratioavg6',
    'cprestamos_personales_min6', 'rankp_Visa_status', 'mactivos_margen_max6', 'rankp_mplazo_fijo_dolares',
    'ctarjeta_master_lag1', 'rankn_ccomisiones_otras', 'rankp_mcuenta_debitos_automaticos_1',
    'ccuenta_debitos_automaticos_ratioavg6', 'ctrx_quarter_delta1', 'rankp_ccuenta_debitos_automaticos',
    'Visa_msaldopesos_ratioavg6', 'ctarjeta_master_delta2', 'mpasivos_margen_delta1', 'ccomisiones_mantenimiento_delta2',
    'rankn_Visa_mpagominimo', 'Visa_mpagospesos_lag1', 'mcuenta_corriente_lag1', 'rankp_Master_cconsumos',
    'mcuentas_saldo_delta2', 'cprestamos_personales', 'Visa_msaldototal_ratioavg6', 'Visa_delinquency_lag1',
    'rankn_ccaja_ahorro', 'rankp_mpagomiscuentas', 'mpayroll_sobre_edad_tend6', 'mcuentas_saldo_delta1',
    'mtransferencias_recibidas', 'Visa_delinquency_delta2', 'rankp_ccaja_seguridad', 'Visa_Finiciomora_lag1',
    'mrentabilidad_max6', 'rankn_mrentabilidad_annual', 'rankn_mpagomiscuentas', 'mactivos_margen_min6',
    'rankp_cpagomiscuentas', 'mpagomiscuentas', 'rankp_mrentabilidad_annual', 'mcomisiones_lag2', 'Visa_cconsumos',
    'internet_lag2', 'rankp_Master_mpagominimo', 'cprestamos_personales_tend6', 'mpayroll_sobre_edad_delta2',
    'ctarjeta_visa_transacciones_max6', 'cextraccion_autoservicio_ratioavg6', 'Visa_mpagominimo_tend6',
    'mpayroll_delta2', 'rankp_mtransferencias_emitidas', 'ctarjeta_visa_debitos_automaticos_ratioavg6',
    'rankp_Master_mfinanciacion_limite_1', 'mpasivos_margen_ratioavg6', 'rankp_Master_mpagospesos',
    'cprestamos_personales_lag2', 'rankp_mtransferencias_recibidas', 'ctarjeta_visa_tend6', 'internet_lag1',
    'cpagomiscuentas', 'ctrx_quarter_normalizado_lag2', 'rankn_Visa_status', 'Visa_mpagominimo_delta1',
    'mcomisiones_mantenimiento', 'Visa_cconsumos_ratioavg6', 'rankp_cextraccion_autoservicio', 'rankn_cproductos',
    'mcomisiones_otras_ratioavg6', 'rankp_mtransferencias_recibidas_1', 'mtarjeta_master_consumo_ratioavg6',
    'internet_ratioavg6', 'ccomisiones_otras_ratioavg6', 'Master_msaldopesos', 'Master_mfinanciacion_limite',
    'ctrx_quarter_delta2', 'Master_mpagospesos_ratioavg6', 'mpasivos_margen_lag2', 'ctrx_quarter_normalizado_delta2',
    'cliente_edad_lag2', 'rankp_mrentabilidad', 'mrentabilidad_annual_lag1', 'Visa_fechaalta', 'mactivos_margen_tend6',
    'mcaja_ahorro_delta2', 'mactivos_margen_ratioavg6', 'rankp_mcomisiones_mantenimiento', 'Visa_mpagosdolares',
    'mcuentas_saldo_min6', 'Master_mpagominimo', 'Visa_msaldototal_lag2', 'mactivos_margen_lag1', 'ccaja_ahorro',
    'internet_max6', 'rankp_Visa_mconsumosdolares', 'mcuentas_saldo_max6', 'mcomisiones_mantenimiento_delta1',
    'mttarjeta_visa_debitos_automaticos_ratioavg6', 'mpasivos_margen_lag1', 'Visa_mpagominimo_lag2',
    'Visa_mpagominimo_delta2', 'mcuenta_corriente_max6', 'cpayroll_trx_tend6', 'mcomisiones_mantenimiento_lag2',
    'Visa_msaldopesos_lag1', 'mcomisiones_ratioavg6', 'Visa_fechaalta_lag1', 'Master_fechaalta', 'cextraccion_autoservicio',
    'rankn_mcomisiones_mantenimiento_1', 'Master_Fvencimiento', 'ccaja_ahorro_max6', 'cprestamos_personales_delta2',
    'Visa_mpagospesos_tend6', 'Master_status_delta2', 'ctarjeta_master_transacciones_ratioavg6',
    'mprestamos_personales_tend6', 'mcuenta_corriente_delta1', 'cliente_antiguedad', 'cliente_antiguedad_lag1',
    'Master_mpagominimo_tend6', 'Visa_mfinanciacion_limite', 'mrentabilidad', 'rankp_Master_mlimitecompra_1',
    'cpagomiscuentas_ratioavg6', 'rankp_thomebanking', 'ccuenta_debitos_automaticos_ratioavg6', 'ccomisiones_otras',
    'rankp_ccomisiones_otras', 'rankn_mtransferencias_recibidas', 'mtransferencias_recibidas_lag1',
    'Visa_msaldopesos_lag2', 'cliente_edad_min6', 'Visa_mconsumospesos_lag2', 'rankn_mtarjeta_master_consumo',
    'cprestamos_personales_lag1', 'mrentabilidad_lag1', 'Master_Fvencimiento_ratioavg6',
    'rankn_mcuenta_debitos_automaticos_1', 'ccaja_seguridad_lag2', 'mtransferencias_recibidas_ratioavg6',
    'ctarjeta_master_transacciones', 'mtarjeta_visa_consumo_delta2', 'Master_msaldototal',
    'rankp_Visa_mfinanciacion_limite_1', 'mcomisiones_max6', 'mrentabilidad_annual_delta2',
    'ccomisiones_mantenimiento_lag2', 'mpayroll_sobre_edad_delta1', 'ctarjeta_visa_transacciones_delta1',
    'rankp_ccajas_transacciones', 'rankp_mttarjeta_visa_debitos_automaticos', 'ctarjeta_debito_transacciones_max6',
    'mcomisiones_otras_lag2', 'Master_mpagominimo_lag1', 'rankp_Master_mfinanciacion_limite', 'Master_fultimo_cierre',
    'cdescubierto_preacordado_delta1', 'mprestamos_hipotecarios_max6', 'rankn_ccajas_consultas',
    'mprestamos_prendarios_max6', 'cseguro_vida_max6', 'ccaja_ahorro_tend6', 'ctarjeta_master_descuentos',
    'matm_other', 'Master_Finiciomora', 'Master_msaldodolares', 'Master_madelantopesos', 'Master_mpagado',
    'Master_delinquency_tend6', 'ccuenta_debitos_automaticos_tend6', 'Master_Finiciomora_tend6',
    'mplazo_fijo_dolares_tend6', 'tmobile_app_tend6', 'tcuentas_max6', 'rankn_ctransferencias_emitidas',
    'tcuentas_min6', 'mprestamos_prendarios_min6', 'ctarjeta_debito_min6', 'mprestamos_hipotecarios_min6',
    'cseguro_vida_min6', 'ccallcenter_transacciones_delta1', 'Visa_cconsumos_delta1', 'Visa_msaldodolares_delta1',
    'mcheques_emitidos_rechazados_delta1', 'ccuenta_debitos_automaticos_delta1', 'mprestamos_prendarios_delta2',
    'tcuentas_lag1', 'chomebanking_transacciones_delta1', 'rankp_ctarjeta_debito',
    'rankp_chomebanking_transacciones_1', 'rankp_Visa_mpagado', 'ccallcenter_transacciones_delta2',
    'ccaja_ahorro_delta2', 'mcuenta_debitos_automaticos_delta2', 'ccuenta_debitos_automaticos_delta2',
    'Master_delinquency_delta2', 'tcallcenter_delta2', 'Visa_status_delta2',
    'mttarjeta_visa_debitos_automaticos_delta2', 'chomebanking_transacciones_delta2', 'rankn_Visa_mlimitecompra',
    'rankn_thomebanking', 'mcheques_emitidos_rechazados_lag1', 'numero_de_cliente'
]

FEATURE_SETS = {
    'seleccion_500': FEATURES_SELECCION_500,
    'seleccion_730': FEATURES_SELECCION_730
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def download_dataset_from_gcs(gcs_url: str, local_path: str):
    """Descarga el dataset desde Google Cloud Storage"""
    if os.path.exists(local_path):
        logger.info(f"Dataset ya existe en {local_path}, omitiendo descarga")
        return
    
    logger.info(f"Descargando dataset desde {gcs_url}...")
    
    # Parsear la URL de GCS
    # Formato: gs://bucket/path/to/file
    if not gcs_url.startswith("gs://"):
        raise ValueError(f"URL de GCS inválida: {gcs_url}")
    
    url_parts = gcs_url[5:].split("/", 1)  # Remover "gs://"
    bucket_name = url_parts[0]
    blob_name = url_parts[1] if len(url_parts) > 1 else ""
    
    # Crear directorio local si no existe
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Descargar archivo
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    logger.info(f"Descargando {blob_name} desde bucket {bucket_name}...")
    blob.download_to_filename(local_path)
    logger.info(f"Dataset descargado exitosamente en {local_path}")

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def load_dataset(path_parquet: str, months: list[int]):
    """Carga el dataset y crea columnas objetivo"""
    if isinstance(months, str):
        months = [months]
    
    df_lazy = (
        pl.scan_parquet(path_parquet, low_memory=True)
        .filter(pl.col("foto_mes").is_in(months))
        .with_columns([
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0)
            .otherwise(1)
            .alias("y_train"),
            
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1)
            .otherwise(0)
            .alias("y_true"),
            
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(1)
            .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
            .when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
            .otherwise(None)
            .alias("w_train")
        ])
    )
    
    df = df_lazy.collect()
    return df

def load_dataset_undersampling_efficient(
    path_parquet: str,
    months: list[int],
    fraction: float = 0.1,
    seed: int = 480,
    stratified: bool = False
) -> pl.DataFrame:
    """Carga dataset con undersampling"""
    if isinstance(months, str):
        months = [months]
    
    if not path_parquet.endswith('.parquet'):
        raise ValueError(f"Tipo de archivo no soportado: {path_parquet}")
    
    logger.info(f"Cargando dataset desde {path_parquet} con undersampling (fracción={fraction})")
    
    df_lazy = (
        pl.scan_parquet(path_parquet, low_memory=True)
        .filter(pl.col("foto_mes").is_in(months))
        .with_columns([
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(0).otherwise(1).alias("y_train"),
            pl.when(pl.col("clase_ternaria") == "BAJA+2").then(1).otherwise(0).alias("y_true"),
            pl.when(pl.col("clase_ternaria") == "CONTINUA").then(1)
             .when(pl.col("clase_ternaria") == "BAJA+1").then(1.00001)
             .when(pl.col("clase_ternaria") == "BAJA+2").then(1.00002)
             .otherwise(None)
             .alias("w_train")
        ])
    )
    
    logger.info("Aplicando undersampling...")
    
    df_lazy = (
        df_lazy
        .with_columns(
            pl.when(pl.col("clase_ternaria") == "CONTINUA")
            .then(
                ((pl.col("numero_de_cliente").hash() + pl.lit(seed)).hash() % 1000000) / 1000000.0
            )
            .otherwise(None)
            .alias("_hash_val")
        )
        .filter(
            (pl.col("clase_ternaria") != "CONTINUA")
            | (pl.col("_hash_val") <= fraction)
        )
        .select(pl.all().exclude(["_hash_val"]))
    )
    
    df = df_lazy.collect()
    logger.info(f"Dataset cargado: {df.height} registros")
    
    return df

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO Y PREDICCIÓN
# ============================================================================

def train_model(
    params: dict,
    dtrain: lgb.Dataset,
    features: list[str]
) -> lgb.Booster:
    """Entrena un modelo LightGBM"""
    train_params = params.copy()
    train_params['deterministic'] = True
    train_params['bagging_fraction_seed'] = train_params['seed']
    train_params['feature_fraction_seed'] = train_params['seed']
    
    if train_params.get('bagging_freq', 0) == 0:
        train_params.pop('neg_bagging_fraction', None)
        train_params.pop('pos_bagging_fraction', None)
        train_params.pop('bagging_fraction', None)
    
    logger.info(f"Entrenando modelo con {len(features)} features, {train_params.get('num_boost_round')} rounds")
    modelo = lgb.train(train_params, dtrain)
    logger.info("Entrenamiento completado")
    
    return modelo

def predict_testset(
    modelo: lgb.Booster,
    months: list[int],
    df: pl.DataFrame
) -> pl.DataFrame:
    """Genera predicciones para el testset"""
    df = df.filter(pl.col("foto_mes").is_in(months))
    
    clientes = df["numero_de_cliente"].to_numpy()
    X = df.select(modelo.feature_name()).to_numpy()
    
    logger.info(f"Generando predicciones para {df.height} registros")
    y_pred = modelo.predict(X)
    
    resultados = pl.DataFrame({
        "numero_de_cliente": clientes,
        "y_pred": y_pred,
        "foto_mes": df["foto_mes"]
    })
    
    return resultados

def merge_predictions(pred_acumuladas: pl.DataFrame, n_submissions: int = 10000) -> pl.DataFrame:
    """Combina predicciones de múltiples modelos promediando"""
    cols_pred = [c for c in pred_acumuladas.columns if c.startswith('y_pred_')]
    
    if not cols_pred:
        raise ValueError("No se encontraron columnas de predicción")
    
    pred_final = pred_acumuladas.with_columns(
        (pl.sum_horizontal(cols_pred) / len(cols_pred)).alias('y_pred_mean')
    )
    
    pred_final = (
        pred_final
        .sort('y_pred_mean', descending=True)
        .with_row_index('row_idx')
        .with_columns([
            pl.when(pl.col('row_idx') < n_submissions).then(1).otherwise(0).alias('predict')
        ])
    )
    
    return pred_final

# ============================================================================
# FUNCIONES ORQUESTADORAS
# ============================================================================

def execute_config(config: dict, dataset_path: str, val_months: list[int]) -> pl.DataFrame:
    """Ejecuta una configuración completa y retorna predicciones finales"""
    experiment_name = config['experiment_name']
    logger.info(f"=== Ejecutando {experiment_name} ===")
    
    fixed_params = config['fixed_params']
    model_names = [key for key in config.keys() if key.startswith("model_")]
    model_names.sort()
    
    logger.info(f"Modelos a ejecutar: {model_names}")
    
    df_valid = load_dataset(path_parquet=dataset_path, months=val_months)
    
    model_predictions = []
    
    for model_name in model_names:
        logger.info(f"\n--- Procesando {model_name} ---")
        model_config = config[model_name]
        
        features_all = []
        for feature_name in model_config['chosen_features']:
            if feature_name in FEATURE_SETS:
                features_all.extend(FEATURE_SETS[feature_name])
            else:
                raise ValueError(f"Feature set '{feature_name}' no encontrado en FEATURE_SETS")
        features_all = list(set(features_all))
        
        if 'clase_ternaria' in features_all:
            features_all.remove('clase_ternaria')
        
        features_train = features_all.copy()
        
        params = model_config['params'].copy()
        params.update(fixed_params)
        
        months = model_config['months']
        undersampling_fraction = model_config.get('undersampling_fraction', 1.0)
        semillerio = model_config.get('semillerio', 1)
        n_submissions = model_config.get('n_submissions', 11000)
        
        use_undersampling = (undersampling_fraction is not None and 
                            undersampling_fraction < 1.0 and 
                            undersampling_fraction > 0.0)
        
        if use_undersampling:
            df_train = load_dataset_undersampling_efficient(
                path_parquet=dataset_path,
                months=months,
                fraction=undersampling_fraction,
                seed=0,  # Seed base para el primer experimento
                stratified=False
            )
        else:
            df_train = load_dataset(path_parquet=dataset_path, months=months)
        
        X_train = df_train.select(features_train).to_numpy()
        y_train = df_train["y_train"].to_numpy()
        w_train = df_train["w_train"].to_numpy()
        
        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            feature_name=features_train,
            free_raw_data=True
        )
        
        del df_train, X_train, y_train, w_train
        gc.collect()
        
        # Entrenar semillerío
        pred_acumuladas = None
        semillerio_seeds = [i for i in range(semillerio)]
        
        for sem_idx, sem_seed in enumerate(semillerio_seeds):
            logger.info(f"  Entrenando modelo {sem_idx + 1}/{semillerio} (seed {sem_seed})")
            
            params_sem = params.copy()
            params_sem["seed"] = sem_seed
            params_sem["verbose"] = -1
            
            # Entrenar modelo
            model = train_model(params=params_sem, dtrain=dtrain, features=features_train)
            
            # Predecir
            resultados = predict_testset(modelo=model, months=val_months, df=df_valid)
            
            # Acumular predicciones
            pred_df = resultados.select(['numero_de_cliente', 'foto_mes', 'y_pred']).clone()
            pred_df = pred_df.rename({'y_pred': f'y_pred_{sem_seed}'})
            
            if pred_acumuladas is None:
                base_cols = resultados.select(['numero_de_cliente', 'foto_mes']).clone()
                pred_acumuladas = base_cols.join(pred_df, on=['numero_de_cliente', 'foto_mes'], how='left')
            else:
                pred_acumuladas = pred_acumuladas.join(pred_df, on=['numero_de_cliente', 'foto_mes'], how='left')
            
            del model
            gc.collect()
        
        # Merge de predicciones del semillerio
        pred_final_model = merge_predictions(pred_acumuladas, n_submissions=n_submissions)
        
        # Guardar predicción final de este modelo
        model_predictions.append(pred_final_model.select(['numero_de_cliente', 'foto_mes', 'y_pred_mean']))
        
        del pred_acumuladas, pred_final_model, dtrain
        gc.collect()
    
    # Si hay múltiples modelos, ensamblar sus predicciones
    if len(model_predictions) > 1:
        logger.info(f"\n--- Ensamblando {len(model_predictions)} modelos ---")
        ensemble_pred_df = None
        
        for model_idx, pred_df in enumerate(model_predictions):
            pred_df_renamed = pred_df.select(['numero_de_cliente', 'foto_mes', 'y_pred_mean']).rename(
                {'y_pred_mean': f'y_pred_model_{model_idx}'}
            )
            
            if ensemble_pred_df is None:
                ensemble_pred_df = pred_df_renamed.clone()
            else:
                ensemble_pred_df = ensemble_pred_df.join(
                    pred_df_renamed,
                    on=['numero_de_cliente', 'foto_mes'],
                    how='full',
                    coalesce=True
                )
        
        # Calcular promedio de todos los modelos
        pred_cols_all = [c for c in ensemble_pred_df.columns if c.startswith('y_pred_model_')]
        ensemble_pred_df = ensemble_pred_df.with_columns(
            (pl.sum_horizontal(pred_cols_all) / len(pred_cols_all)).alias('y_pred_mean')
        )
        
        # Seleccionar top N
        n_submissions_config = config[model_names[0]].get('n_submissions', 11000)
        ensemble_pred_df = (
            ensemble_pred_df
            .sort('y_pred_mean', descending=True)
            .with_row_index('row_idx')
            .with_columns([
                pl.when(pl.col('row_idx') < n_submissions_config).then(1).otherwise(0).alias('predict')
            ])
            .select(['numero_de_cliente', 'foto_mes', 'y_pred_mean', 'predict'])
        )
        
        logger.info(f"Ensamble de {len(model_predictions)} modelos completado")
        return ensemble_pred_df.select(['numero_de_cliente', 'foto_mes', 'y_pred_mean'])
    else:
        return model_predictions[0].select(['numero_de_cliente', 'foto_mes', 'y_pred_mean'])

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO ENSAMBLE STANDALONE")
    logger.info("=" * 80)
    
    logger.info("\n[1/5] Descargando dataset desde GCS...")
    download_dataset_from_gcs(DATASET_GCS_URL, LOCAL_DATASET_PATH)
    
    logger.info("\n[2/5] Ejecutando Config 1...")
    pred_config1 = execute_config(CONFIG_1, LOCAL_DATASET_PATH, VAL_MONTH)
    
    logger.info("\n[3/5] Ejecutando Config 2...")
    pred_config2 = execute_config(CONFIG_2, LOCAL_DATASET_PATH, VAL_MONTH)
    
    logger.info("\n[4/5] Ensamblando predicciones finales...")
    
    pred_config1_renamed = pred_config1.select(['numero_de_cliente', 'foto_mes', 'y_pred_mean']).rename(
        {'y_pred_mean': 'y_pred_config1'}
    )
    pred_config2_renamed = pred_config2.select(['numero_de_cliente', 'foto_mes', 'y_pred_mean']).rename(
        {'y_pred_mean': 'y_pred_config2'}
    )
    
    ensemble_final = pred_config1_renamed.join(
        pred_config2_renamed,
        on=['numero_de_cliente', 'foto_mes'],
        how='full',
        coalesce=True
    )
    
    # Promediar predicciones
    ensemble_final = ensemble_final.with_columns(
        ((pl.col('y_pred_config1') + pl.col('y_pred_config2')) / 2.0).alias('y_pred_mean')
    )
    
    # Seleccionar top 11000
    ensemble_final = (
        ensemble_final
        .sort('y_pred_mean', descending=True)
        .with_row_index('row_idx')
        .filter(pl.col('row_idx') < N_SUBMISSIONS)
        .select('numero_de_cliente')
    )
    
    # 5. Guardar resultado final
    logger.info("\n[5/5] Guardando resultado final...")
    output_file = "ensamble_meses_11000.csv"
    
    ensemble_final.select('numero_de_cliente').write_csv(
        output_file, 
        include_header=False
    )
    logger.info(f"Resultado guardado en {output_file}")
    logger.info(f"Total de clientes seleccionados: {ensemble_final.height}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PROCESO COMPLETADO")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()


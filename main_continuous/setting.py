import numpy as np
import pandas as pd
# no apache now
state_col = ['shock_index','age', 'gender', 'weight', 'readmission', 'sirs','elixhauser_vanwalraven',
             'MechVent', 
             'heartrate', 'respiratoryrate', 'spo2', 'temperature',
             'sbp', 'dbp', 'mbp', 'lactate', 'bicarbonate', 'pao2', 'paco2', 'pH',
             'hemoglobin', 'baseexcess', 'chloride', 'glucose', 'calcium','ionized_calcium','albumin',
             'potassium', 'sodium', 'co2', 'pao2fio2ratio', 'wbc', 'platelet', 'bun',
             'creatinine', 'ptt', 'pt', 'inr', 'ast', 'alt', 'bilirubin_total',
             'gcs', 'fio2', 'urine_output', 'output_total', 'sofa_24hours','magnesium','bloc']

next_state_col = ['next_shock_index','next_age','next_gender','next_weight','next_readmission','next_sirs','next_elixhauser_vanwalraven',
                  'next_MechVent',
                  'next_heartrate','next_respiratoryrate','next_spo2','next_temperature', 
                  'next_sbp','next_mbp','next_dbp','next_lactate','next_bicarbonate','next_pao2','next_paco2','next_pH',
                  'next_hemoglobin','next_baseexcess', 'next_chloride','next_glucose','next_calcium','next_ionized_calcium','next_albumin',
                  'next_potassium','next_sodium','next_co2', 'next_pao2fio2ratio', 'next_wbc',  'next_platelet',  'next_bun', 
                  'next_creatinine','next_ptt', 'next_pt','next_inr','next_ast', 'next_alt','next_bilirubin_total',
                  'next_gcs','next_fio2','next_urine_output', 'next_output_total','next_sofa_24hours','next_magnesium','next_bloc']

context_state_col=['context_shock_index','context_age','context_gender','context_weight','context_readmission','context_sirs',
 'context_elixhauser_vanwalraven','context_MechVent','context_heartrate',
 'context_respiratoryrate','context_spo2','context_temperature','context_sbp','context_dbp','context_mbp',
 'context_lactate','context_bicarbonate','context_pao2',
 'context_paco2','context_pH','context_hemoglobin','context_baseexcess',
 'context_chloride','context_glucose','context_calcium','context_ionized_calcium','context_albumin',
 'context_potassium','context_sodium','context_co2','context_pao2fio2ratio','context_wbc',
 'context_platelet','context_bun','context_creatinine', 'context_ptt', 'context_pt',
 'context_inr', 'context_ast', 'context_alt', 'context_bilirubin_total', 'context_gcs',
 'context_fio2', 'context_urine_output', 'context_output_total', 'context_sofa_24hours', 'context_magnesium',
 'context_bloc']

context_next_state_col =['context_next_shock_index','context_next_age','context_next_gender','context_next_weight',
 'context_next_readmission','context_next_sirs','context_next_elixhauser_vanwalraven','context_next_MechVent',
 'context_next_heartrate','context_next_respiratoryrate','context_next_spo2',
 'context_next_temperature','context_next_sbp','context_next_mbp',
 'context_next_dbp','context_next_lactate','context_next_bicarbonate','context_next_pao2',
 'context_next_paco2','context_next_pH','context_next_hemoglobin','context_next_baseexcess','context_next_chloride',
 'context_next_glucose','context_next_calcium','context_next_ionized_calcium',
 'context_next_albumin','context_next_potassium','context_next_sodium','context_next_co2',
 'context_next_pao2fio2ratio','context_next_wbc','context_next_platelet','context_next_bun',
 'context_next_creatinine','context_next_ptt','context_next_pt','context_next_inr',
 'context_next_ast','context_next_alt','context_next_bilirubin_total','context_next_gcs',
 'context_next_fio2','context_next_urine_output','context_next_output_total',
 'context_next_sofa_24hours','context_next_magnesium','context_next_bloc']

master_action_num = 4
action_dis_col = ['iv_fluids_quantile', 'vasopressors_quantile']
action_con_col = ['iv_fluids', 'vasopressors']
action_ori_con_col = ['ori_iv_fluids', 'ori_vasopressors']
ai_action_con_col = ['ai_action_IV_only', 'ai_action_Vasso_only']
ai_action_dis_col = ['ai_action_dis_IV_only', 'ai_action_dis_Vasso_only']
reward_col= ['mortality_hospital','ori_sofa_24hours','next_ori_sofa_24hours','ori_lactate','next_ori_lactate']
per_flag = True
per_alpha = 0.6     # PER hyperparameter
per_epsilon = 0.01  # PER hyperparameter
beta_start =0.9
MODEL = 'QMIX' # 'DQN' or 'FQI'
GAMMA = 0.99
ACTION_SPACE = 2
SEED = 123

hidden_factor = 8

ITERATION_ROUND = 7001 #40000  #150000 800
ITERATION_ROUND_QMIX = 8000 #20000
ITERATION_ROUND_IV = 8000 # 10000
ITERATION_ROUND_Vaso = 8000 #5000
ITERATION_ROUND_PRETRAIN = 40000
TOTAL_ITERATION_NUM = 1
BATCH_SIZE = 128

lr_Q = 5e-4
lr_V = 3e-4
lr_actor = 2e-4
lr_actor1 = 3e-4
lr_actor2 = 3e-4
replace_target_iter = 100
nn = 32



lr_alpha = 1e-4
lambda_A = 0.85  # only for discrete qmix
Q_threshold = 20


REWARD_FUN = 'reward_mortality_sofa_lactate'

def reward_only_long(x):
    res = 0
    if (x['done'] == 1 and x['mortality_hospital'] == 1):
        res += -15
    elif (x['done'] == 1 and x['mortality_hospital'] == 0):
        res += 15
    elif x['done'] == 0:
        res = 0
    else:
        res = np.nan
    return res

def reward_mortality_sofa_lactate(x):
    res = 0
    if (x['done'] == 1 and x['mortality_hospital'] == 1):
        res = -15
    elif (x['done'] == 1 and x['mortality_hospital'] == 0):
        res = 15
    elif x['done'] == 0:
        if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
            res += -0.025 #C0
        res += -0.125 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
        res += -2 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
    else:
        res = np.nan
    return res

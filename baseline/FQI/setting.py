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

# action_dis_col = ['iv_fluids_level', 'vasopressors_level']
action_dis_col = ['iv_fluids_quantile', 'vasopressors_quantile']
reward_col= ['mortality_hospital','ori_sofa_24hours','next_ori_sofa_24hours','ori_lactate','next_ori_lactate']

SEED = 1
ITERATION_ROUND = 150000  #150000 800
ACTION_SPACE = 25
BATCH_SIZE = 256
per_flag = True
per_alpha = 0.6     # PER hyperparameter
per_epsilon = 0.01  # PER hyperparameter
beta_start =0.9
MODEL = 'DQN' # 'DQN' or 'FQI'
GAMMA = 0.99


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

def reward_mortality_sofa_lactate2(x):
    res = 0
    if (x['done'] == 1 and x['mortality_hospital'] == 1):
        res = -100
    elif (x['done'] == 1 and x['mortality_hospital'] == 0):
        res = 100
    elif x['done'] == 0:
        if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
            res += -0.025 #C0
        res += -0.125 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
        res += -2 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
    else:
        res = np.nan
    return res


def reward_mortality_sofa_lactate3(x):
    res = 0
    if (x['done'] == 1 and x['mortality_hospital'] == 1):
        res = -15
    elif (x['done'] == 1 and x['mortality_hospital'] == 0):
        res = 15
    elif x['done'] == 0:
        if ((x['next_ori_sofa_24hours'] == x['ori_sofa_24hours']) and x['next_ori_sofa_24hours']>0):
            res += -0.025 #C0
        res += -0.5 * (x['next_ori_sofa_24hours'] - x['ori_sofa_24hours']) #C1
        res += -4 * np.tanh(x['next_ori_lactate']-x['ori_lactate']) #C2
    else:
        res = np.nan
    return res

def score_mortality_sofa_lactate_mbp(x):
    result = 0
    if x['done'] == 0:
        if (x['next_ori_lactate']) <2:
            result += 2
        elif (x['next_ori_lactate']) <4:
            result +=1
        elif (x['next_ori_lactate']<x['ori_lactate']):
            result += 0.5
        else:
            result -= 1
        
        if(x['next_ori_sofa_24hours']<x['ori_sofa_24hours']):
            result += 2
        
        if(x['next_ori_mbp']<=80 and x['next_ori_mbp']>=70):
            result += 2
    else:
        result = np.nan
    return result

def score_mortality_sofa_lactate_mbp_v3(x):
    result = 0
    if x['done'] == 0:
        if (x['next_ori_lactate'] > x['ori_lactate']):
            result +=0

        else:
            result += 1


        if (x['next_ori_sofa_24hours'] > x['ori_sofa_24hours']):
            result += 0
        else:
            result +=1

        if ((x['next_ori_mbp'] > 80 or x['next_ori_mbp'] < 70) and (x['ori_mbp'] <= 80 and x['ori_mbp'] >= 70)):
            result += 0
        else:
            result +=1
    
    
    else:
        result = np.nan
    return result


def get_one_hot_Encs(actions_all):
    actions_all['Group_ID'] = actions_all['iv_fluids_quantile'] * 5 + actions_all['vasopressors_quantile'] - 6
    actions_all['Group_ID'] = pd.to_numeric(actions_all['Group_ID'], downcast='integer')
    original_array = np.array(actions_all['Group_ID'])
    one_hot_ary = np.array([[0] * 25] * len(actions_all))

    for i in range(len(one_hot_ary)):
        one_hot_ary[i][original_array[i]] = 1
    df = pd.DataFrame(one_hot_ary)
    df.columns = ['A_0', 'A_1', 'A_2', 'A_3', 'A_4', 'A_5', 'A_6', 'A_7', 'A_8', 'A_9', 'A_10', 
                   'A_11', 'A_12', 'A_13', 'A_14', 'A_15', 'A_16', 'A_17','A_18', 'A_19', 'A_20', 
                   'A_21', 'A_22', 'A_23', 'A_24']
    return df

"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import sys
import time

sys.path.append('../../env/')

import pandas as pd
import numpy as np
import tensorflow as tf
import alg_singleAC
from tqdm import tqdm
import logging
import argparse
import setting
from evaluation_singleAC import *
# from evaluation_graphs_continuous import *

state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
action_con_col = setting.action_con_col
ai_action_con_col = setting.ai_action_con_col

ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
STATE_DIM = len(state_col) #48 
REWARD_FUN = setting.REWARD_FUN

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),)

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

def compute_master_action(iv_fluids_quantile, vasopressors_quantile):
    if iv_fluids_quantile==1 and vasopressors_quantile==1:
        master_action=0
    elif iv_fluids_quantile==1 and vasopressors_quantile>1:
        master_action=1
    elif iv_fluids_quantile>1 and vasopressors_quantile==1:
        master_action=2
    else:
        master_action = 3
    return master_action

def convert_to_unit(old_value,old_max, old_min, new_max, new_min):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value


def train_single(RL, data, first_run=True, writer = None):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)  
        memory_array = np.concatenate([np.array(data[state_col]),
                                       np.array(data[action_con_col]), 
                                       np.array(data['reward']).reshape(-1, 1), 
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data[next_state_col])], axis = 1)
        np.save('single_agent_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING QMIX AGENT \n')

    EPISODE = int(ITERATION_ROUND)
    for i in tqdm(range(EPISODE)):
        RL.train_step(i, writer = writer)
        

    loss = RL.cost_his
    return loss

def train_function(config, data, MEMORY_SIZE):
    
    start_time = time.time()
    
    seed = setting.SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
#     tf.random.set_seed(seed)
    
    dir_name = 'single_AC'
    model_name = 'model.ckpt'
    summarize = False
    
    os.makedirs('../../result/%s'%dir_name, exist_ok=True)

    l_state = STATE_DIM
    l_action = 2   
    N_agent = 1


# ################################ Single Agent AC network continuous action ###############################

   
    
    tf.reset_default_graph()

    alg = alg_singleAC.singleAC(N_agent, l_state, l_action, config['nn_qmix'], memory_size=len(data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.compat.v1.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
    loss = train_single(alg, data, first_run=True, writer = writer)
    # save model
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    
    # evaluate model
    eval_obs = data[state_col]
    actions_int = alg.run_actor(eval_obs, sess)

    a_0 = data[action_con_col[0]]
    a_1 = data[action_con_col[1]]

    phys_qmix = alg.run_phys_Q_continuous(sess, list_obs = eval_obs, a_0=a_0, a_1=a_1)
    ai_qmix = alg.run_RL_Q_continuous(sess,list_obs = eval_obs, a_0=actions_int[:,0], a_1=actions_int[:,1])
    
    result_array = np.concatenate([data.values, actions_int, phys_Q,ai_Q], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(data.columns)+['ai_action_IV', 'ai_action_Vasso','phys_Q','ai_Q'])


# ############################################

 
    run_time = display_time((time.time()-start_time))
    print("done!")   
    print("Total run time with {} episodes:\n {}".format(setting.ITERATION_ROUND, run_time))    
    print(result.head(1))
    print("start evaluation")
    run_eval(result, loss, datatype = 'eICU')
    

if __name__ == "__main__":
    
    with open('../config_qmix.json', 'r') as f:
        config = json.load(f)
        
    df = pd.read_csv('../../../sepsis_RL/mimic_v2/data_rl_4h_train_test_split.csv')
    df.fillna(0, inplace=True)
    
    vaso_min=np.min(df['vasopressors'])
    vaso_max = np.max(df['vasopressors'])
    iv_min = np.min(df['iv_fluids'])
    iv_max = np.max(df['iv_fluids'])
    df['vasopressors_old'] = df['vasopressors']
    df['iv_fluids_old'] = df['iv_fluids']
    df['vasopressors'] = df.apply(lambda x: convert_to_unit(x['vasopressors_old'],vaso_max, vaso_min, 0.5,-0.5), axis=1)
    df['iv_fluids'] = df.apply(lambda x: convert_to_unit(x['iv_fluids_old'],iv_max, iv_min, 0.5,-0.5), axis=1)    

    #train data
    data = df[df['train_test']=="train"]
    data =data.reset_index()
    
#     #test data
#     data_test = df[df['train_test']=="test"]
#     data_test =data_test.reset_index()

    MEMORY_SIZE = len(data)

    train_function(config, data, MEMORY_SIZE)        

"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import random
import sys
import time
import math

# sys.path.append('../')

import pandas as pd
import numpy as np
import tensorflow as tf
import Qmix_alg as alg_discrete
from tqdm import tqdm
import logging
import argparse
import setting
from Qmix_evaluation import *


state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
action_con_col = setting.action_con_col
ai_action_con_col = setting.ai_action_con_col
ai_action_dis_col = setting.ai_action_dis_col
ITERATION_ROUND_PRETRAIN = setting.ITERATION_ROUND_PRETRAIN
ITERATION_ROUND_QMIX = setting.ITERATION_ROUND_QMIX

ITERATION_ROUND = setting.ITERATION_ROUND
TOTAL_ITERATION_NUM = setting.TOTAL_ITERATION_NUM
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
STATE_DIM = len(state_col) #48 
REWARD_FUN = setting.REWARD_FUN
context_state_col = setting.context_state_col
context_next_state_col = setting.context_next_state_col
hidden_factor = setting.hidden_factor
l_state = len(setting.state_col)
nn = setting.nn
Q_threshold = setting.Q_threshold
FM_list = ['FM_' + str(i) for i in range(setting.hidden_factor)]
next_FM_list = ['next_FM_' + str(i) for i in range(setting.hidden_factor)]
FM_context_list = ['FM_' + str(i) for i in range(setting.hidden_factor)]
FM_context_list.extend(['FM_context' + str(i) for i in range(setting.hidden_factor)])
next_FM_context_list = ['next_FM_' + str(i) for i in range(setting.hidden_factor)]
next_FM_context_list.extend(['next_FM_context' + str(i) for i in range(setting.hidden_factor)])



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
        master_action=2
    elif iv_fluids_quantile>1 and vasopressors_quantile==1:
        master_action=1
    else:
        master_action = 3
    return master_action

def convert_to_unit(old_value,old_max, old_min, new_max, new_min):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return new_value


def train_mixer(RL, data, use_FM, first_run=True, writer = None, epoch = 1):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)  
        actions = data.apply(lambda x: x[action_dis_col[0]] * 5 + x[action_dis_col[1]]  -6, axis =1)

        memory_array = np.concatenate([np.array(data[state_col]),
                                       np.array(data[next_state_col]),
                                       np.array(actions).reshape(-1, 1), 
                                       np.array(data[action_dis_col]-1),
                                       np.array(data['reward']).reshape(-1, 1), 
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data[FM_context_list]),
                                       np.array(data[next_FM_context_list])],axis = 1)
        np.save('../result/discrete/Qmix_discrete_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load('../result/discrete/Qmix_discrete_memory.npy')

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
#     print('\nDiscreteSTART TRAINING Qmix_Discrete AGENT %s/%s \n' % (epoch,TOTAL_ITERATION_NUM))
    if use_FM>0:
        print('\nSTART TRAINING QMIX AGENT with Embedding K{}\n'.format(setting.hidden_factor))
    else:
        print('\nSTART TRAINING QMIX AGENT WITH NO EMBEDDING\n')
    EPISODE = int(ITERATION_ROUND_QMIX)
    for i in tqdm(range(EPISODE)):
        RL.train_step(i, use_FM, writer = writer)
        

    loss = RL.cost_his
  
    return loss

def train_function(df, use_FM):
    
    start_time = time.time()
    
    seed = setting.SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    summarize = False


    l_state = STATE_DIM
    l_action = 5   
    N_agent = 2
    master_action_num = setting.master_action_num
    
    data = df[df['train_test']=="train"]
    data =data.reset_index(drop = True)
    if use_FM>0:
        input_dim = 2*setting.hidden_factor
    else:
        input_dim = STATE_DIM



    print("loading data...")
    df = pd.read_csv('../result/data/mimic_embeddings_K'+str(setting.hidden_factor)+'.csv')
    df['master_action'] = df.apply(lambda x: compute_master_action(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1)

    df['Q_phys_no_action'] = df['Q_phys_no_action'].apply(lambda x: -Q_threshold if x<-Q_threshold else x if x<Q_threshold else Q_threshold)


# ################################ Qmix Discrete action ###############################
   
    mixer_data = df[(df['master_action']==3) & (df['train_test']=="train")]
    mixer_data = mixer_data.reset_index(drop = True)
    
    tf.reset_default_graph()

    alg = alg_discrete.Qmix_discrete(N_agent, l_state, hidden_factor, input_dim, l_action, nn, memory_size=len(mixer_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.compat.v1.summary.FileWriter('models/Qmix', sess.graph)
    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
    mixer_loss = train_mixer(alg, mixer_data, use_FM, first_run=True, writer = writer, epoch = 1)
    # save model
    saver.save(sess, 'models/qmix/model.ckpt')
    
    i = 2
    while(i<=TOTAL_ITERATION_NUM):
        new_saver = tf.compat.v1.train.import_meta_graph('models/qmix/model.ckpt.meta')
        new_saver.restore(sess, 'models/qmix/model.ckpt')
        loss = train_single(alg, mixer_data, first_run=False, writer = writer, epoch = i)
        # save model
        saver.save(sess, 'models/qmix/model.ckpt')
        i = i+1
    
    # evaluate model
    if (use_FM>0):
        eval_state = np.array(df[FM_context_list])
        eval_obs = np.stack((eval_state, eval_state)).reshape((-1,2*setting.hidden_factor))
#     elif (use_FM):
#         eval_state = np.array(combined_data[FM_list])
#         eval_obs = np.stack((eval_state, eval_state)).reshape((-1,setting.hidden_factor))
    else:
        eval_state = np.array(df[state_col])
        eval_obs = np.stack((eval_state, eval_state)).reshape((-1,len(state_col))) 
        
#     iv_only = df[ai_action_dis_col[0]].astype('int64')
#     vasso_only = df[ai_action_dis_col[1]].astype('int64')

    iv_only = (df[action_dis_col[0]]-1).astype('int64')
    vasso_only = (df[action_dis_col[1]]-1).astype('int64')
    iv_actions, vaso_actions = alg.run_actor(eval_obs, sess, iv_only = iv_only, vasso_only = vasso_only)
    a_0 = (df[action_dis_col[0]]-1)
    a_1 = (df[action_dis_col[1]]-1)

    phys_qmix = alg.run_phys_Q(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0, a_1=a_1, iv_only = iv_only, vasso_only = vasso_only)
    ai_qmix = alg.run_RL_Q(sess, list_state=eval_state, list_obs = eval_obs, a_0=iv_actions, a_1=vaso_actions, iv_only = iv_only, vasso_only = vasso_only)
    
    result_array = np.concatenate([df.values, iv_actions, vaso_actions, phys_qmix,ai_qmix], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(df.columns)+['ai_action_qmix_IV', 'ai_action_qmix_Vasso','Q_phys_qmix','Q_ai_qmix'])
        
        
    result.to_csv('../result/Qmix_single_discrete_result.csv', encoding = 'gb18030')


 
    run_time = display_time((time.time()-start_time))
    print("done!")   
    print("Total run time with {} episodes:\n {}".format(setting.ITERATION_ROUND, run_time))    
    print("start evaluation")
    train_result = result[result['train_test']=="train"]
    train_result = train_result.reset_index(drop=True)
    test_result = result[result['train_test']=="test"]
    test_result = test_result.reset_index(drop =True)
    

    if use_FM>0:
        print("evaluating train result")
        run_eval(train_result, mixer_loss, datatype = 'mimic', phase = "Qmix_train_Embedding")
        print("evaluating test result")
        run_eval(test_result, mixer_loss, datatype = 'mimic', phase = "Qmix_test_Embedding")
    else:
        print("evaluating train result")
        run_eval(train_result, mixer_loss, datatype = 'mimic', phase = "Qmix_train")
        print("evaluating test result")
        run_eval(test_result,  mixer_loss, datatype = 'mimic', phase = "Qmix_test")    
   
   ############################################


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_FM', '-e', type=float, required=True)
    args = parser.parse_args()  
        

  
    df = pd.read_csv('../result/data/data_rl_4h_train_test_split_3steps.csv')
    
    df.fillna(0, inplace=True)
    max_vaso = df.vasopressors.quantile(0.99)
    df['vasopressors']=df['vasopressors'].apply(lambda x: x if x<max_vaso else max_vaso )
    max_iv = df.iv_fluids.quantile(0.99)
    df['iv_fluids']=df['iv_fluids'].apply(lambda x: x if x<max_iv else max_iv )   
    
#     vaso_min=np.min(df['vasopressors'])
#     vaso_max = np.max(df['vasopressors'])
#     iv_min = np.min(df['iv_fluids'])
#     iv_max = np.max(df['iv_fluids'])
#     df['vasopressors_old'] = df['vasopressors']
#     df['iv_fluids_old'] = df['iv_fluids']
#     df['vasopressors'] = df.apply(lambda x: convert_to_unit(x['vasopressors_old'],vaso_max, vaso_min, 0.5, -0.5 ), axis=1)
#     df['iv_fluids'] = df.apply(lambda x: convert_to_unit(x['iv_fluids_old'],iv_max, iv_min, 0.5,-0.5), axis=1)    
    
    df['master_action'] = df.apply(lambda x: compute_master_action(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1)
    
    print("process data done")


    train_function(df, args.use_FM)
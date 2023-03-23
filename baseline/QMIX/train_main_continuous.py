"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import random
import sys
import time

sys.path.append('../env/')

import pandas as pd
import numpy as np
import tensorflow as tf
import alg_qmix_continuous as alg_qmix
from tqdm import tqdm
import logging
import argparse
import setting
from evaluation_graphs_quick import *
# from evaluation_graphs_continuous import *

state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
action_con_col = setting.action_con_col
ai_action_con_col = setting.ai_action_con_col
ITERATION_ROUND = setting.ITERATION_ROUND
ITERATION_ROUND_QMIX = setting.ITERATION_ROUND_QMIX
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

def train_master_RL(RL, data, first_run=True):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]), 
                            np.array(data['master_action']).reshape(-1, 1),
                            np.array(data['reward']).reshape(-1, 1), 
                            np.array(data['done']).reshape(-1, 1),
#                             np.array(data['Q_phys_IV_only']).reshape(-1,1),
#                             np.array(data['Q_phys_Vasso_only']).reshape(-1,1),
#                             np.array(data['Q_phys_qmix']).reshape(-1,1), 
                            np.array(data[next_state_col])],
                            axis = 1)
        np.save('master_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING MASTER AGENT\n')

    EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND)
  
    for i in tqdm(range(EPISODE)):
        RL.learn(i)
    loss = RL.cost_his
    return loss

def train_single_RL_IV(RL, data, first_run=True, writer = None):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]), 
                            np.array(data[action_con_col[0]]).reshape(-1,1), 
                            np.array(data['reward']).reshape(-1, 1), 
                            np.array(data['done']).reshape(-1, 1),
                            np.array(data[next_state_col])],
                            axis = 1)
        np.save('IV_only_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING IV AGENT \n')


    EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND_QMIX)        
    for i in tqdm(range(EPISODE)):
        RL.train_step_single_AC(i, writer = writer)
        

    IV_loss = RL.cost_his
    return IV_loss

def train_single_RL_Vasso(RL, data, first_run=True,writer = None):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)        
        memory_array = np.concatenate([np.array(data[state_col]), 
                            np.array(data[action_con_col[1]]).reshape(-1,1), 
                            np.array(data['reward']).reshape(-1, 1), 
                            np.array(data['done']).reshape(-1, 1),
                            np.array(data[next_state_col])],
                            axis = 1)
        np.save('IV_only_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING VASSO AGENT\n')


    EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND_QMIX)        
    for i in tqdm(range(EPISODE)):
        RL.train_step_single_AC(i, writer = writer)
        

    Vasso_loss = RL.cost_his
    return Vasso_loss

def train_mixer(RL, data, first_run=True, writer = None):
    if first_run:
         # reward function
#         data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
#         actions = data.apply(lambda x: x[action_dis_col[0]] * 5 + x[action_dis_col[1]]  -6, axis =1)


        
#         memory_array = np.concatenate([np.array(data[state_col]), 
#                             np.array(actions).reshape(-1, 1), 
#                             np.array(data['reward']).reshape(-1, 1), 
#                             np.array(data['done']).reshape(-1, 1),
#                             np.array(data[next_state_col])],
#                             axis = 1)
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)  
        memory_array = np.concatenate([np.array(data[state_col]),
                                       np.array(data[action_con_col]), 
                                       np.array(data[ai_action_con_col]),
                                       np.array(data['reward']).reshape(-1, 1), 
                                       np.array(data['done']).reshape(-1, 1),
                                       np.array(data[next_state_col])], axis = 1)
        np.save('mixer_memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING QMIX AGENT \n')
    MEMORY_SIZE_mix = len(data)

    EPISODE = int(MEMORY_SIZE_mix / BATCH_SIZE * ITERATION_ROUND_QMIX)        
    for i in tqdm(range(EPISODE)):
        RL.train_step_new(i, writer = writer)
        

    loss = RL.cost_his
    return loss

def train_function(config, data, MEMORY_SIZE):
    
    start_time = time.time()
    
    seed = setting.SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
#     tf.random.set_seed(seed)
    
    dir_name = 'qmix'
    model_name = 'model.ckpt'
    summarize = False
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)

    l_state = STATE_DIM
    l_action = 2   
    N_agent = 2
    master_action_num = 4

        
# ################### single_AC network for IV only ################### 
# #     tf.reset_default_graph()
# #     IV_only_data = result.copy()
#     combined_data = data.copy()
#     IV_only_data = data[data['master_action']==1]
#     IV_only_data = IV_only_data.reset_index()
    

#     alg = alg_qmix.Single_AC(1, l_state, l_action, config['nn_qmix'], memory_size=len(IV_only_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

#     config_proto = tf.ConfigProto()
#     config_proto.gpu_options.allow_growth = True
#     sess = tf.Session(config=config_proto)
#     sess.run(tf.global_variables_initializer())
#     sess.run(alg.list_initialize_target_ops)
#     writer = tf.compat.v1.summary.FileWriter('../results/%s' % dir_name, sess.graph)
#     saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
#     iv_loss = train_single_RL_IV(alg, IV_only_data, first_run=True, writer = writer)
#     # save model
#     saver.save(sess, '../results/IV_only/%s' % (model_name))
#     # evaluate single IV model
#     eval_state = combined_data[state_col]
#     eval_obs = np.array(eval_state).reshape((-1,l_state))
#     actions_int = alg.run_actor(eval_obs, sess)
#     a_0 = combined_data['ori_iv_fluids']
#     phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0)
#     ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0])
    
#     result_array = np.concatenate([combined_data.values, actions_int, phys_qmix,ai_qmix], axis=1)
#     result = pd.DataFrame(result_array, 
#                           columns=list(combined_data.columns)+['ai_action_IV_only','Q_phys_IV_only','Q_ai_IV_only'])
#     print("result")
#     print(result.head(1))
# ################### single_AC network for Vasso only ################### 

# #     Vasso_only_data = result.copy()
#     combined_data = result.copy()
#     Vasso_only_data = data[data['master_action']==2]
#     Vasso_only_data = Vasso_only_data.reset_index()
    
#     tf.reset_default_graph()
#     alg = alg_qmix.Single_AC(1, l_state, l_action, config['nn_qmix'], memory_size=len(Vasso_only_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

#     config_proto = tf.ConfigProto()
#     config_proto.gpu_options.allow_growth = True
#     sess = tf.Session(config=config_proto)
#     sess.run(tf.global_variables_initializer())
#     sess.run(alg.list_initialize_target_ops)
#     writer = tf.compat.v1.summary.FileWriter('../results/%s' % dir_name, sess.graph)
#     saver = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
#     vasso_loss = train_single_RL_Vasso(alg, Vasso_only_data, first_run=True, writer = writer)
#     # save model
#     saver.save(sess, '../results/Vasso_only/%s' % (model_name))
#     # evaluate single IV model
#     eval_state = combined_data[state_col]
#     eval_obs = np.array(eval_state).reshape((-1,l_state))
#     actions_int = alg.run_actor(eval_obs, sess)
#     a_0 = combined_data['ori_vasopressors']
#     phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0)
#     ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0])
    
#     result_array = np.concatenate([combined_data.values, actions_int, phys_qmix,ai_qmix], axis=1)
#     result = pd.DataFrame(result_array, 
#                           columns=list(combined_data.columns)+['ai_action_Vasso_only','Q_phys_Vasso_only','Q_ai_Vasso_only'])


#     print("result")
#     print(result.head(1))
# # ################################ Qmix network ###############################
# #     data_mix = result[(result['master_ai_action']==3) & (result['master_action']==3)]
#     combined_data = result.copy()
# #     print(combined_data.columns)
#     mixer_data = combined_data[combined_data['master_action']==3]
#     mixer_data = mixer_data.reset_index()
    
#     tf.reset_default_graph()

#     alg = alg_qmix.Qmix(N_agent, l_state, l_action, config['nn_qmix'], memory_size=len(mixer_data), batch_size=BATCH_SIZE, e_greedy_increment=0.001)    

#     config_proto = tf.ConfigProto()
#     config_proto.gpu_options.allow_growth = True
#     sess = tf.Session(config=config_proto)
#     sess.run(tf.global_variables_initializer())
#     sess.run(alg.list_initialize_target_ops)
#     writer = tf.compat.v1.summary.FileWriter('../results/%s' % dir_name, sess.graph)
#     saver_mixer = tf.compat.v1.train.Saver(max_to_keep=100)
    
    
#     mixer_loss = train_mixer(alg, mixer_data, first_run=True, writer = writer)
#     # save model
#     saver_mixer.save(sess, '../results/%s/%s' % (dir_name, model_name))
    
#     # evaluate model
#     eval_state = combined_data[state_col]
#     eval_obs = np.stack((eval_state, eval_state)).reshape((-1,l_state))
#     actions_int = alg.run_actor(eval_obs, sess)
# #     a_0 = (data[action_dis_col[0]]-1)
# #     a_1 = (data[action_dis_col[1]]-1)
#     a_0 = combined_data[action_con_col[0]]
#     a_1 = combined_data[action_con_col[1]]
#     iv_only = combined_data[ai_action_con_col[0]]
#     vasso_only = combined_data[ai_action_con_col[1]]
#     phys_qmix = alg.run_phys_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0, a_1=a_1, iv_only = iv_only, vasso_only = vasso_only)
#     ai_qmix = alg.run_RL_Q_continuous(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0], a_1=actions_int[:,1], iv_only = iv_only, vasso_only = vasso_only)
    
#     result_array = np.concatenate([combined_data.values, actions_int, phys_qmix,ai_qmix], axis=1)
#     result = pd.DataFrame(result_array, 
#                           columns=list(combined_data.columns)+['ai_action_qmix_IV', 'ai_action_qmix_Vasso','Q_phys_qmix','Q_ai_qmix'])

################### Master Agent to decide no_action, IV_only, Vasso_only, or Qmix ###################
    tf.reset_default_graph()
#     combined_data = result.copy()
    combined_data = data.copy()


    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    
    RL_master = alg_qmix.DuelingDQN(n_actions=master_action_num, n_features=STATE_DIM, memory_size=MEMORY_SIZE,
                                   batch_size=BATCH_SIZE, e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)
    
    sess.run(tf.global_variables_initializer())

    master_loss = train_master_RL(RL_master, combined_data, first_run=True)
    # save model
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, 'models/duel_DQN')
    new_saver = tf.compat.v1.train.import_meta_graph('models/duel_DQN.meta')
    new_saver.restore(sess, 'models/duel_DQN')
    
    eval_q = sess.run(RL_master.q_eval, feed_dict={RL_master.s: combined_data[state_col]})    
    result_array = np.concatenate([combined_data.values, eval_q], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(combined_data.columns)+['Q_0', 'Q_1', 'Q_2', 'Q_3'])
    

    Q_list = ['Q_' + str(i) for i in range(master_action_num)]
    result['Q_ai_master'] = np.max(result[Q_list],axis = 1)
    result['ai_action_master'] = np.argmax(np.array(result[Q_list]),axis = 1)
    print("result")
    print(result.head(1))
# ############################################

 
    run_time = display_time((time.time()-start_time))
    print("done!")   
    print("Total run time with {} episodes: {}".format(setting.ITERATION_ROUND, run_time))    
    print(result.head(1))
    print("start evaluation")
    run_eval(result, master_loss, iv_loss, vasso_loss, mixer_loss, datatype = 'eICU')
    

if __name__ == "__main__":
    
    with open('config_qmix.json', 'r') as f:
        config = json.load(f)
        
    df = pd.read_csv('../../sepsis_RL/mimic_v2/data_rl_4h_train_test_split.csv')
    df.fillna(0, inplace=True)
    
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
    
    df['master_action'] = df.apply(lambda x: compute_master_action(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1)

    #train data
    data = df[df['train_test']=="train"]
    data =data.reset_index()
    
#     #test data
#     data_test = df[df['train_test']=="test"]
#     data_test =data_test.reset_index()

    MEMORY_SIZE = len(data)

    train_function(config, data, MEMORY_SIZE)        

"""Trains off-policy algorithms, such as QMIX and IQL."""

import json
import os
import random
import sys
import time

sys.path.append('../env/')

import pandas as pd
import numpy as np
import tensorflow as tf
import alg_qmix_new as alg_qmix
from tqdm import tqdm
import logging
import argparse
import setting
from evaluation_graphs import *

state_col = setting.state_col
next_state_col = setting.next_state_col
action_dis_col = setting.action_dis_col
ITERATION_ROUND = setting.ITERATION_ROUND
ACTION_SPACE = setting.ACTION_SPACE
BATCH_SIZE = setting.BATCH_SIZE
STATE_DIM = len(state_col) #48 
REWARD_FUN = setting.REWARD_FUN

def train(RL, data, first_run=True, writer = None):
    if first_run:
         # reward function
        data['reward'] = data.apply(eval('setting.' + REWARD_FUN) , axis = 1)
        actions = data.apply(lambda x: x[action_dis_col[0]] * 5 + x[action_dis_col[1]]  -6, axis =1)

        
        memory_array = np.concatenate([np.array(data[state_col]), 
                            np.array(actions).reshape(-1, 1), 
                            np.array(data['reward']).reshape(-1, 1), 
                            np.array(data['done']).reshape(-1, 1),
                            np.array(data[next_state_col])],
                            axis = 1)
        np.save('memory.npy', memory_array)
        
    else:
        
        memory_array = np.load(memory_array)

    print('\nSTART store_transition\n')
    RL.store_transition(memory_array)
    
    print('\nSTART TRAINING\n')

    EPISODE = int(MEMORY_SIZE / BATCH_SIZE * ITERATION_ROUND)        
    for i in tqdm(range(EPISODE)):
        RL.train_step(i, writer = writer)
        

    loss = RL.cost_his
    return loss


def train_function(config, data, MEMORY_SIZE):

    
    seed = setting.SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    
    dir_name = 'qmix'
    model_name = 'model.ckpt'
    summarize = False
    
    os.makedirs('../results/%s'%dir_name, exist_ok=True)

    l_state = STATE_DIM
    l_action = ACTION_SPACE
   
    N_agent = 2
    

    alg = alg_qmix.Alg(N_agent, l_state, l_action, config['nn_qmix'], memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, e_greedy_increment=0.001)

    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    sess.run(alg.list_initialize_target_ops)
    writer = tf.summary.FileWriter('../results/%s' % dir_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=100)
    
    loss = train(alg, data, first_run=True, writer = writer)
    # save model
    saver = tf.train.Saver()
#     saver.save(sess, 'models/duel_DQN')
    saver.save(sess, '../results/%s/%s' % (dir_name, model_name))
    
#     # evaluate model
#     eval_q = sess.run(RL_model.q_eval, feed_dict={RL_model.s: data[state_col]})
#     print(np.where(eval_q<0))
    
#     result_array = np.concatenate([data.values, eval_q], axis=1)
#     result = pd.DataFrame(result_array, 
#                           columns=list(data.columns)+['Q_0', 'Q_1', 'Q_2', 'Q_3', 'Q_4', 'Q_5', 'Q_6', 'Q_7', 'Q_8', 'Q_9', 'Q_10', 'Q_11', 'Q_12', 'Q_13', 'Q_14', 'Q_15', 'Q_16', 'Q_17','Q_18', 'Q_19', 'Q_20', 'Q_21', 'Q_22', 'Q_23', 'Q_24'])
                         
    
#     print(eval_q.shape, type(eval_q))
#     print(eval_q)
    
    
#     eval = Evaluation()
#     run_eval(result, loss, args.data)

    # evaluate model
    eval_state = data[state_col]
    eval_obs = np.stack((eval_state, eval_state)).reshape((-1,l_state))
    actions_int = alg.run_actor(eval_obs, sess)
    a_0 = (data[action_dis_col[0]]-1)
    a_1 = (data[action_dis_col[1]]-1)
    phys_qmix = alg.run_phys_Q(sess, list_state=eval_state, list_obs = eval_obs, a_0=a_0, a_1=a_1)
    ai_qmix = alg.run_RL_Q(sess, list_state=eval_state, list_obs = eval_obs, a_0=actions_int[:,0], a_1=actions_int[:,1])
    
    result_array = np.concatenate([data.values, actions_int, phys_qmix,ai_qmix], axis=1)
    result = pd.DataFrame(result_array, 
                          columns=list(data.columns)+['A_0', 'A_1','phys_qmix','ai_qmix'])

    print("done!")
    print("result")
    print(result.head(1))
    print("start evaluation")
    run_eval(result, loss, datatype = 'eICU')
    

if __name__ == "__main__":
    
    with open('config_qmix.json', 'r') as f:
        config = json.load(f)
        
    df = pd.read_csv('../../sepsis_RL/mimic_v2/data_rl_4h_train_test_split.csv')
    df.fillna(0, inplace=True)
    #train data
    data = df[df['train_test']=="train"]
    data =data.reset_index()
    
#     #test data
#     data_test = df[df['train_test']=="test"]
#     data_test =data_test.reset_index()

    MEMORY_SIZE = len(data)
#     save_dir = "./models/DQN_reward2/"
#     save_path = "./models/DQN_reward2/ckpt" #The path to save patient model to.rk
    train_function(config, data, MEMORY_SIZE)        

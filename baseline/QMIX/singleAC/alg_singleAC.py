"""Implementation of QMIX."""

import numpy as np
import tensorflow as tf
import networks_singleAC as networks
import sys
import setting
import pandas as pd

# DuelingDQN line 487
# SingleAC line 675

class singleAC(object):


    def __init__(self,  n_agents, l_state, l_action, nn, 
                 memory_size=500, batch_size = 32, e_greedy=0.9, e_greedy_increment=None,replace_target_iter=200, 
                 sess = None):
        """
        Args:
            n_agents: number of agents on the team controlled by this alg
            l_state, l_action: int
            nn: dictionary with neural net sizes
        """

        self.l_state = l_state
        self.l_action = l_action
        self.nn = nn
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.l_state *2+4))

        self.n_agents = n_agents
        self.tau = 0.01
        self.lr_Q =setting.lr_Q
        self.lr_V = setting.lr_V
        self.lr_actor = setting.lr_actor
        self.gamma = 0.99

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_max = e_greedy        
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0

        # Initialize computational graph
        self.create_networks()
        self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops()
        self.create_train_op()
        self.cost_his = []
        self.replace_target_iter = replace_target_iter
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.create_summary()


    def create_networks(self):

        # Placeholders
    
        self.obs = tf.placeholder(tf.float32, [None, self.l_state], 'obs')
        self.actions = tf.placeholder(tf.float32, [None, self.l_action], 'actions')

        # Individual agent networks
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy_main"):
            self.actions_selected, self.log_probs = networks.actor(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_action)
            
        with tf.variable_scope("V_main"):
            self.V = networks.critic(self.obs, self.actions, self.nn['n_h1'], self.nn['n_h2'])
            
        with tf.variable_scope("V_target"):
            self.V_target = networks.critic(self.obs, self.actions, self.nn['n_h1'], self.nn['n_h2'])        
        
        
            
    def get_assign_target_ops(self):
        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []
        
        # individual network        
        list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
        map_name_V_main = {v.name.split('main')[1] : v for v in list_V_main}
        list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_target')
        map_name_V_target = {v.name.split('target')[1] : v for v in list_V_target}
        if len(list_V_main) != len(list_V_target):
            raise ValueError("get_initialize_target_ops : lengths of V_main and V_target do not match")
        for name, var in map_name_V_main.items():
            list_initial_ops.append( map_name_V_target[name].assign(var) )
        for name, var in map_name_V_main.items():
            list_update_ops.append( map_name_V_target[name].assign( self.tau*var + (1-self.tau)*map_name_V_target[name] ) )


        return list_initial_ops, list_update_ops     
    


    def create_train_op(self):
        """Ops for training low-level policy."""
        
#         self.actions_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
        # self.probs shape is [batch size * traj length, l_action]
        # now log_probs shape is [batch size * traj length]
#         log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), axis=1)+1e-15) 

        # Critic train op
        self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
        self.V_opt = tf.train.AdamOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)
        # Policy train op
        self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
        self.V_td_error = self.V_td_target - self.V_evaluated
        self.policy_loss = -tf.reduce_mean( tf.multiply( self.log_probs, self.V_td_error ) )
        self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)
        
#         # TD target calculated in train_step() using Mixer_target
#         self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
#         self.loss_mixer = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))

#         self.mixer_opt = tf.train.AdamOptimizer(self.lr_Q)
#         self.mixer_op = self.mixer_opt.minimize(self.loss_mixer)


    def process_actions(self, n_steps, actions):
        """
        actions must have shape [time, n_agents],
        and values are action indices
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, l_action]
        # so each agent gets its own 1-hot row vector
        actions_1hot = np.zeros([n_steps, self.n_agents, self.l_action], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        actions_1hot[grid[0], grid[1], actions] = 1

        # In-place reshape of actions to [time*n_agents, l_action]
        actions_1hot.shape = (n_steps*self.n_agents, self.l_action)

        return actions_1hot

    
    def store_transition(self, memory_array):
        self.memory = memory_array
        

        
    def create_summary(self):
        # mixer network
#         summaries_Q = [tf.summary.scalar('loss_mixer', self.loss_mixer)]
#         mixer_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
#         for v in mixer_main_variables:
#             summaries_Q.append(tf.summary.histogram(v.op.name, v))
#         grads = self.mixer_opt.compute_gradients(self.loss_mixer, mixer_main_variables)
#         for grad, var in grads:
#             if grad is not None:
#                 summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

        #  actor network       
        summaries_policy = [tf.compat.v1.summary.scalar('policy_loss', self.policy_loss)]
        policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
        for v in policy_variables:
            summaries_policy.append(tf.summary.histogram(v.op.name, v))
        grads = self.policy_opt.compute_gradients(self.policy_loss, policy_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_policy.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_policy = tf.compat.v1.summary.merge(summaries_policy)

 
        #  critic network
        summaries_Q = [tf.compat.v1.summary.scalar('V_loss', self.loss_V)]
        V_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
        for v in V_variables:
            summaries_Q.append(tf.summary.histogram(v.op.name, v))
        grads = self.V_opt.compute_gradients(self.loss_V, V_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )                
        self.summary_op_Q = tf.compat.v1.summary.merge(summaries_Q)
        

        
    def run_actor(self, list_obs, sess):
        """Get actions for all agents as a batch.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of action 
        """
        # convert to batch
        obs = np.array(list_obs)
        feed = {self.obs : obs}
        actions_selected = sess.run(self.actions_selected, feed_dict=feed)
        actions = actions_selected.reshape((-1, 2))

        return actions


    def train_step(self, i, writer=None):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.list_update_target_ops)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        done_vec = (1-batch_memory[:, self.l_state + 3]).reshape(-1, 1)
        done_vec = np.squeeze(done_vec)
        
        
        # Each agent for each time step is now a batch entry

        n_steps = self.batch_size
        obs = batch_memory[:, :self.l_state]

        a_0 = self.l_state
        a_1 = self.l_state + 2
        action_reshaped = batch_memory[:, a_0:a_1]

        reward = batch_memory[:, self.l_state + 2]
        rewards = np.squeeze(reward)
        obs_next = batch_memory[:, -self.l_state:]

        # Train critic
        feed = {self.obs : obs_next, self.actions : action_reshaped}
        V_target_res, V_next_res = self.sess.run([self.V_target, self.V], feed_dict=feed)
        V_target_res = np.squeeze(V_target_res)
        V_next_res = np.squeeze(V_next_res)
        done_multiplier = np.squeeze(done_vec)

        V_td_target = rewards + self.gamma * V_target_res * done_multiplier

        feed = {self.V_td_target : V_td_target,
                self.obs : obs, 
                self.actions : action_reshaped}
        _, V_res = self.sess.run([self.V_op, self.V], feed_dict=feed)
        
         
        # Train actor
        V_res = np.squeeze(V_res)
        V_td_target = rewards + self.gamma * V_next_res * done_multiplier
        
        feed = {self.obs : obs,
                self.actions : action_reshaped, self.V_td_target : V_td_target,
                self.V_evaluated : V_res}
        
        action_selected = self.sess.run([self.actions_selected], feed_dict=feed)
        _ = self.sess.run([self.policy_op], feed_dict=feed)
        
        self.epsilon = tf.cond(tf.greater(self.epsilon, self.epsilon_max), lambda: self.epsilon_max, lambda: (self.epsilon + self.epsilon_increment))

        self.learn_step_counter += 1 
        


    def run_phys_Q_continuous(self, sess, list_obs=None, a_0=None, a_1=None):
        """Get qmix value for the physician's action
        
        Args:
            list_obs: list of vectors, one per agent
            sess: TF session

        Returns: np.array of phys qmix values
        """
        # convert to batch
        obs = np.array(list_obs)
        
        a_0 = np.array(a_0).reshape((-1,1))
        a_1 = np.array(a_1).reshape((-1,1))
        actions = np.concatenate((a_0, a_1), axis = 1)

 
        feed = {
                self.actions : actions,
                self.obs : obs}
        phys_Qmix = sess.run(self.V, feed_dict=feed)
        return phys_Qmix
    
    def run_RL_Q_continuous(self, sess, list_obs=None, a_0=None, a_1=None):
        """Get qmix value for the physician's action
        
        Args:
            list_obs: list of vectors, one per agent
            sess: TF session

        Returns: np.array of phys qmix values
        """
        # convert to batch
        state = np.array(list_state)
        obs = np.array(list_obs)
        a_0 = np.array(a_0).reshape((-1,1))
        a_1 = np.array(a_1).reshape((-1,1))
        actions = np.concatenate((a_0, a_1), axis = 1)

        
        feed = {
                self.actions : actions,
                self.obs : obs}
        
        RL_Qmix = sess.run(self.V, feed_dict=feed)
        return RL_Qmix

"""Implementation of CONTINUOUS HMARL."""

import numpy as np
import tensorflow as tf
import embedding_networks as networks
import sys
import setting
import pandas as pd
import wandb
wandb.init(project='Main_Continuous')

np.random.seed(setting.SEED)
tf.set_random_seed(setting.SEED)

class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
            REWARD_THRESHOLD = 20,
            reg_lambda = 5,
            pretrain = True,
            K_hidden_factor = 16
    ):
        self.sess = sess
        self.REWARD_THRESHOLD = REWARD_THRESHOLD
        self.reg_lambda = reg_lambda
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = setting.replace_target_iter
        self.memory_size = memory_size
        self.memory_col_num = 2*self.n_features + 3
        self.memory = np.zeros((self.memory_size, self.memory_col_num))
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.hidden_factor = K_hidden_factor
        self.weights = tf.Variable(tf.random.normal([self.n_features, self.hidden_factor], 0.0, 0.01),
                                   name='feature_embeddings',dtype=tf.float32)  # features_M * K
        
        self.pretrain = pretrain
        

      
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        if output_graph:
            self.writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            

            
            with tf.variable_scope('FM_s'):
                temp = tf.tensordot(s, self.weights, axes = 0)
                b = tf.shape(temp)[1]
                eye = tf.eye(b, dtype=temp.dtype)
                inp_masked = temp * tf.expand_dims(eye, 2)
                nonzero_embeddings = tf.tensordot(inp_masked, tf.ones(b, temp.dtype), [[2], [0]]) # None * l_state * K

                summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1) # None * K
                summed_features_emb_square = tf.square(summed_features_emb)  # None * K
                squared_features_emb = tf.square(nonzero_embeddings)
                squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K
                FM = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K

                Em_State = tf.add(FM, summed_features_emb) # None * K

            with tf.variable_scope('l1'):
                
                w1 = tf.get_variable('w1', [self.hidden_factor, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.matmul(Em_State, w1) + b1
                l1 = tf.maximum(l1, l1*0.5)


            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out, Em_State


        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
         # None * K
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval, self.Em_State= build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.reg_vector = tf.maximum(tf.abs(self.q_eval)-self.REWARD_THRESHOLD,0)
            self.reg_term = tf.reduce_sum(self.reg_vector)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+ self.reg_lambda*self.reg_term
        with tf.variable_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next_tmp, _ = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)
            self.done = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='done')
            self.q_next = tf.multiply(self.q_next_tmp, self.done)

    
    def store_transition(self, memory_array):
        self.memory = memory_array

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    def choose_low_level_Q(self, master_action, Q_no, Q_iv, Q_vasso, Q_mix):
        if master_action == 0:
            Q_return = Q_no
        elif master_action == 1:
            Q_return = Q_iv
        elif master_action == 2:
            Q_return=Q_vasso
        else:
            Q_return = Q_mix
        return Q_return
    def learn(self, i, pretrain = True):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        done_vec = np.tile((1-batch_memory[:, self.n_features + 2]).reshape(-1, 1), self.n_actions)
        states = batch_memory[:, :self.n_features]
        next_states = batch_memory[:, -self.n_features:]
        
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: states})
        
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: next_states, self.done: done_vec}) # next observation
        


        q_target = q_eval.copy()


        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
 
        

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: states,
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)
        wandb.log({"Pretrain_Q_loss": self.cost})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1        
        
        if i % 100 == 0:
            tf.compat.v1.summary.scalar('loss', self.loss)
            merged_summary = tf.compat.v1.summary.merge_all() 
            sum_summary = self.sess.run(merged_summary, feed_dict={self.s_: batch_memory[:, -self.n_features:], self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
            self.writer.add_summary(sum_summary, i)    





            
class Single_AC(object):


    def __init__(self,  n_agents, l_state, hidden_factor,input_dim,l_action, nn, lr_V, lr_actor,
                 memory_size=500, batch_size = 128, e_greedy=0.9, e_greedy_increment=None,replace_target_iter=200, 
                 sess = None):
        """
        Args:
            n_agents: number of agents on the team controlled by this alg
            l_state, l_action: int
            nn: dictionary with neural net sizes
        """

        self.l_state = l_state
        self.hidden_factor =hidden_factor
        self.input_dim = input_dim
        self.l_action = 1
        self.nn = nn
        self.memory_size = memory_size
        self.memory_col_num = 2*self.l_state+4*self.hidden_factor + 3
        self.memory = np.zeros((self.memory_size, self.memory_col_num))

        self.n_agents = 1
        self.tau = 0.01
        self.lr_Q =setting.lr_Q
        self.lr_V = lr_V
        self.lr_actor = setting.lr_actor
        self.gamma = 0.99

        self.agent_labels = np.eye(self.n_agents)
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
        self.replace_target_iter = setting.replace_target_iter
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.create_summary()


    def create_networks(self):

        # Placeholders

        self.obs = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], 'obs')
        self.actions = tf.compat.v1.placeholder(tf.float32, [None, 1], 'actions')

        # Individual agent networks

        with tf.variable_scope("Policy_main"):
            self.actions_selected, self.log_probs = networks.actor(self.obs, self.nn, self.nn, self.l_action)
            
        with tf.variable_scope("V_main"):
            self.V = networks.critic(self.obs, self.actions, self.nn, self.nn)
            
        with tf.variable_scope("V_target"):
            self.V_target = networks.critic(self.obs, self.actions, self.nn, self.nn)        
                 
            
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
        """Ops for training AC policy."""
        
        # Critic train op
        self.V_td_target = tf.compat.v1.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
        self.V_opt = tf.compat.v1.train.RMSPropOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)
        # Policy train op
        self.V_evaluated = tf.compat.v1.placeholder(tf.float32, [None], 'V_evaluated')

        self.policy_term1 = tf.multiply(self.log_probs, self.V_td_target)
        self.policy_loss = -tf.reduce_mean(self.policy_term1 - self.V_evaluated)
        self.policy_opt = tf.compat.v1.train.RMSPropOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)
        


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

        #  actor network       
        summaries_policy = [tf.compat.v1.summary.scalar('policy_loss', self.policy_loss)]
        policy_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
        for v in policy_variables:
            summaries_policy.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.policy_opt.compute_gradients(self.policy_loss, policy_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_policy.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_policy = tf.compat.v1.summary.merge(summaries_policy)

 
        #  critic network
        summaries_Q = [tf.compat.v1.summary.scalar('V_loss', self.loss_V)]
        V_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
        for v in V_variables:
            summaries_Q.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.V_opt.compute_gradients(self.loss_V, V_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )                
        self.summary_op_Q = tf.compat.v1.summary.merge(summaries_Q)
        

        
    def run_actor(self, list_obs, sess):
        """Get actions for all agents as a batch.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of action integers
        """
        # convert to batch
        obs = np.array(list_obs)
        feed = {self.obs : obs}
        actions_selected = sess.run(self.actions_selected, feed_dict=feed)
        actions = actions_selected.reshape((-1, self.n_agents))

        return actions
    def run_phys_Q(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None):
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
        actions = actions.astype(int)
        n_steps = actions.shape[0]

        actions_1hot = self.process_actions(n_steps, actions)
        feed = {self.state : state,
                self.actions_1hot : actions_1hot,
                self.obs : obs}
        phys_Qmix = sess.run(self.mixer, feed_dict=feed)
        return phys_Qmix
    def run_RL_Q(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None):
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
        actions = actions.astype(int)
        n_steps = actions.shape[0]

        actions_1hot = self.process_actions(n_steps, actions)
        feed = {self.state : state,
                self.actions_1hot : actions_1hot,
                self.obs : obs}
        RL_Qmix = sess.run(self.mixer, feed_dict=feed)
        return RL_Qmix
    


    def train_step_single_AC(self, i, use_FM, writer=None, iv_action_only = True):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.list_update_target_ops)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        actions = batch_memory[:, 2*self.l_state] # None * 1
        action_reshaped = actions.reshape((-1,1))
        reward = batch_memory[:, 2*self.l_state + 1] #None * 1
        rewards = np.squeeze(reward.reshape((-1,1))) #None * 1
        done_vec = (1-batch_memory[:, 2*self.l_state + 2]).reshape(-1, 1)
        done_vec = np.squeeze(done_vec)        
        if (use_FM>0):
            obs = np.array(batch_memory[:,-4*self.hidden_factor:-2*self.hidden_factor]) #-4K: -2K
            obs = obs.reshape((-1,2*self.hidden_factor)) # None * 2K
            obs_next = batch_memory[:,-2*self.hidden_factor:] # -2K:
            obs_next = obs_next.reshape((-1,2*self.hidden_factor)) # None * 2K

        else:
            obs = np.array(batch_memory[:,:self.l_state]) # 0 : l_state
            obs = obs.reshape((-1,self.l_state)) # None * l_state
            obs_next = batch_memory[:, self.l_state:2*self.l_state] #l_state:2*l_state
            obs_next = obs_next.reshape((-1,self.l_state))



        
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

        _, V_res,self.cost = self.sess.run([self.V_op, self.V,self.loss_V], feed_dict=feed)
        self.cost_his.append(self.cost)
        wandb.log({"single agent Q_loss": self.cost})
        
         
        # Train actor
        V_res = np.squeeze(V_res)
        V_td_target = rewards + self.gamma * V_next_res * done_multiplier
        
        feed = {self.obs : obs,
                self.actions : action_reshaped,
                self.V_td_target : V_td_target,
                self.V_evaluated : V_res}
        
        _, l_policy = self.sess.run([self.policy_op, self.policy_loss], feed_dict=feed)
        wandb.log({"policy_loss": l_policy}) 


        self.learn_step_counter += 1 
        


    def run_phys_Q_continuous(self, sess, list_state=None, list_obs=None, a_0=None):
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
#         a_1 = np.array(a_1).reshape((-1,1))
        
#         actions = np.concatenate((a_0, a_1), axis = 1)
        actions_reshaped = a_0
#         actions = actions.astype(int)

 
        feed = {
#                 self.state : state,
                self.actions : actions_reshaped,
                self.obs : obs}
        phys_Q = sess.run(self.V, feed_dict=feed)
        return phys_Q
    
    def run_RL_Q_continuous(self, sess, list_state=None, list_obs=None, a_0=None):
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

        actions_reshaped = a_0

        feed = {
                self.actions : actions_reshaped,
                self.obs : obs}
        RL_Q = sess.run(self.V, feed_dict=feed)
        return RL_Q

    
class Qmix(object):


    def __init__(self,  n_agents, l_state, hidden_factor, input_dim, l_action, nn, 
                 memory_size=500, batch_size = 32, e_greedy=0.9, e_greedy_increment=None,replace_target_iter=200, 
                 sess = None):
        """
        Args:
            n_agents: number of agents on the team controlled by this alg
            l_state, l_action: int
            nn: dictionary with neural net sizes
        """

        self.l_state = l_state
        self.hidden_factor = hidden_factor
        self.input_dim = input_dim
        self.l_action = 1
        self.nn = nn
        self.memory_size = memory_size
        self.memory_col_num = 2*self.l_state+4*self.hidden_factor + 6
        self.memory = np.zeros((self.memory_size, self.memory_col_num))

        self.n_agents = 2
        self.tau = 0.01
        
        self.lr_Q =setting.lr_Q
        self.lr_V = setting.lr_V
        self.lr_actor1 = setting.lr_actor1
        self.lr_actor2 = setting.lr_actor2        
        self.gamma = 0.99

        self.agent_labels = np.eye(self.n_agents)
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
        self.replace_target_iter = setting.replace_target_iter
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.create_summary()


    def create_networks(self):

        # Placeholders
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], 'state')
        self.obs = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], 'obs')
        self.actions = tf.compat.v1.placeholder(tf.float32, [None, 1], 'actions')
        self.single_actions = tf.compat.v1.placeholder(tf.float32, [None, 1], 'single_actions')

        # Individual agent networks
        self.epsilon = tf.compat.v1.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy_main1"):
            self.actions_selected1, self.log_probs1 = networks.actor(self.obs, self.nn, self.nn, self.l_action)
        with tf.variable_scope("Policy_main2"):
            self.actions_selected2, self.log_probs2 = networks.actor(self.obs, self.nn, self.nn, self.l_action)           
        with tf.variable_scope("V_main"):
            self.V = networks.critic_mixer(self.obs, self.actions, self.single_actions, self.nn, self.nn)
            
        with tf.variable_scope("V_target"):
            self.V_target = networks.critic_mixer(self.obs, self.actions, self.single_actions, self.nn, self.nn)        
        
        # To extract Q-value from V and V_target
        self.mixer_q_input = tf.reshape( self.V, [-1, self.n_agents] )
        self.mixer_target_q_input = tf.reshape( self.V_target, [-1, self.n_agents] )

        # Mixing network
        with tf.variable_scope("Mixer_main"):
            self.mixer = networks.Qmix_mixer(self.mixer_q_input, self.state, self.input_dim, self.n_agents, self.nn)
        with tf.variable_scope("Mixer_target"):
            self.mixer_target = networks.Qmix_mixer(self.mixer_target_q_input, self.state, self.input_dim, self.n_agents, self.nn)           
            
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

        # mixer network 
        list_Mixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
        map_name_Mixer_main = {v.name.split('main')[1] : v for v in list_Mixer_main}
        list_Mixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_target')
        map_name_Mixer_target = {v.name.split('target')[1] : v for v in list_Mixer_target}
        
        if len(list_Mixer_main) != len(list_Mixer_target):
            raise ValueError("get_initialize_target_ops : lengths of Mixer_main and Mixer_target do not match")
        
        # ops for equating main and target
        for name, var in map_name_Mixer_main.items():
            # create op that assigns value of main variable to
            # target variable of the same name
            list_initial_ops.append( map_name_Mixer_target[name].assign(var) )
        
        # ops for slow update of target toward main
        for name, var in map_name_Mixer_main.items():
            # incremental update of target towards main
            list_update_ops.append( map_name_Mixer_target[name].assign( self.tau*var + (1-self.tau)*map_name_Mixer_target[name] ) )

        return list_initial_ops, list_update_ops     
    


    def create_train_op(self):
        """Ops for training low-level policy."""
        
        self.actions_taken = tf.compat.v1.placeholder(tf.float32, [None, self.l_action], 'action_taken')

        # Critic train op
        self.V_td_target = tf.compat.v1.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
        self.V_opt = tf.compat.v1.train.RMSPropOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)
        # Policy train op
        self.V_evaluated = tf.compat.v1.placeholder(tf.float32, [None], 'V_evaluated')

        self.log_probs1_reshape = tf.squeeze(self.log_probs1)
        self.policy_term1 = tf.multiply( self.log_probs1_reshape, self.V_td_target)
        self.policy_term_half1 = self.policy_term1[:self.batch_size]
        self.V_evaluated_half1 = self.V_evaluated[:self.batch_size]
        self.policy_loss1 = -tf.reduce_mean(self.policy_term_half1 - self.V_evaluated_half1)
        self.policy_opt1 = tf.train.RMSPropOptimizer(self.lr_actor1)
        self.policy_op1 = self.policy_opt1.minimize(self.policy_loss1)
        
        self.log_probs2_reshape = tf.squeeze(self.log_probs2)
        self.policy_term2 = tf.multiply(self.log_probs2_reshape , self.V_td_target)
        self.policy_term_half2 = self.policy_term2[-self.batch_size:]
        self.V_evaluated_half2 = self.V_evaluated[-self.batch_size:]
        self.policy_loss2 = -tf.reduce_mean(self.policy_term_half2 - self.V_evaluated_half2)
        self.policy_opt2 = tf.train.RMSPropOptimizer(self.lr_actor2)
        self.policy_op2 = self.policy_opt2.minimize(self.policy_loss2)
        
        
        # TD target calculated in train_step() using Mixer_target
        self.td_target = tf.compat.v1.placeholder(tf.float32, [None], 'td_target')
        self.loss_mixer = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))

        self.mixer_opt = tf.compat.v1.train.RMSPropOptimizer(self.lr_Q)
        self.mixer_op = self.mixer_opt.minimize(self.loss_mixer)


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


    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.l_action)
        return action
    
    def store_transition(self, memory_array):
        self.memory = memory_array
        

        
    def create_summary(self):
        # mixer network
        summaries_Q = [tf.compat.v1.summary.scalar('loss_mixer', self.loss_mixer)]
        mixer_main_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
        for v in mixer_main_variables:
            summaries_Q.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.mixer_opt.compute_gradients(self.loss_mixer, mixer_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )

        #  actor network1       
        summaries_policy1 = [tf.compat.v1.summary.scalar('policy_loss1', self.policy_loss1)]
        policy_variables1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main1')
        for v in policy_variables1:
            summaries_policy1.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.policy_opt1.compute_gradients(self.policy_loss1, policy_variables1)
        for grad, var in grads:
            if grad is not None:
                summaries_policy1.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )
       
        self.summary_op_policy = tf.compat.v1.summary.merge(summaries_policy1)
        
        #  actor network2     
        summaries_policy2 = [tf.compat.v1.summary.scalar('policy_loss2', self.policy_loss2)]
        policy_variables2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main2')
        for v in policy_variables2:
            summaries_policy2.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.policy_opt2.compute_gradients(self.policy_loss2, policy_variables2)
        for grad, var in grads:
            if grad is not None:
                summaries_policy2.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op_policy = tf.compat.v1.summary.merge(summaries_policy2)
 
        #  critic network

        V_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_main')
        for v in V_variables:
            summaries_Q.append(tf.compat.v1.summary.histogram(v.op.name, v))
        grads = self.V_opt.compute_gradients(self.loss_V, V_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.compat.v1.summary.histogram(var.op.name+'/gradient', grad) )                
        self.summary_op_Q = tf.compat.v1.summary.merge(summaries_Q)
        

        
    def run_actor(self, list_obs, sess, iv_only = None, vasso_only = None):
        """Get actions for all agents as a batch.
        
        Args:
            list_obs: list of vectors, one per agent
            epsilon: exploration parameter
            sess: TF session

        Returns: np.array of action integers
        """
        # convert to batch
        obs = np.array(list_obs)
        n_steps = len(iv_only)
        iv_only = np.array(iv_only).reshape((-1,1)).astype('float32')
        vasso_only = np.array(vasso_only).reshape((-1,1)).astype('float32')
        ai_single_actions = np.concatenate((vasso_only, iv_only), axis = 0)        
        feed = {self.obs : obs, self.single_actions :ai_single_actions }
        actions_selected1, actions_selected2 = sess.run([self.actions_selected1,self.actions_selected2], feed_dict=feed)
        actions1 = actions_selected1[:n_steps]
        actions2 = actions_selected2[-n_steps:]
        iv_actions = actions1.reshape((-1,1))
        vaso_actions = actions2.reshape((-1,1))        
        return iv_actions, vaso_actions
    
    def run_phys_Q(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None):
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
        actions = actions.astype(int)
        n_steps = actions.shape[0]

        actions_1hot = self.process_actions(n_steps, actions)
        feed = {self.state : state,
                self.actions_1hot : actions_1hot,
                self.obs : obs}
        phys_Qmix = sess.run(self.mixer, feed_dict=feed)
        return phys_Qmix
    def run_RL_Q(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None):
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
        actions = actions.astype(int)
        n_steps = actions.shape[0]

        actions_1hot = self.process_actions(n_steps, actions)
        feed = {self.state : state,
                self.actions_1hot : actions_1hot,
                self.obs : obs}
        RL_Qmix = sess.run(self.mixer, feed_dict=feed)
        return RL_Qmix
    


    def train_step(self, i, use_FM,  writer=None):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.list_update_target_ops)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        phys_actions = np.array(batch_memory[:, 2*self.l_state:2*self.l_state+2])
        action_reshaped = phys_actions.reshape((-1,1))
        iv_only = 2*self.l_state + 2
        vasso_only = 2*self.l_state + 3
        ai_iv_only = np.array(batch_memory[:, iv_only]).reshape((-1,1))
        ai_vasso_only = np.array(batch_memory[:, vasso_only]).reshape((-1,1))
        ai_single_actions = np.concatenate((ai_vasso_only, ai_iv_only), axis = 1)
        ai_single_actions_reshaped = ai_single_actions.reshape((-1,1))
#         print("ai_single_actions_reshaped{}\n".format(ai_single_actions_reshaped.shape))
#         print("action_reshaped{}".format(action_reshaped.shape))
        
        reward = batch_memory[:, 2*self.l_state + 4] #None * 1
        rewards = np.squeeze(np.stack((reward, reward)).reshape((-1,1)))
        done_vec = (1-batch_memory[:, 2*self.l_state + 5]).reshape(-1, 1)
        done_vec = np.squeeze(done_vec)
        done_multiplier = np.squeeze(np.stack((done_vec, done_vec)).reshape((-1,1)))
        if (use_FM>0):
            state =np.array( batch_memory[:,-4*self.hidden_factor:-2*self.hidden_factor]).reshape((-1,2*self.hidden_factor)) #-4K: -2K
            obs = np.stack((state, state)).reshape((-1,2*self.hidden_factor)) # 2None * 2K
            state_next = np.array(batch_memory[:,-2*self.hidden_factor:]).reshape((-1,2*self.hidden_factor)) # -2K:
            obs_next = np.stack((state_next, state_next)).reshape((-1,2*self.hidden_factor)) #2None * 2K

        else:
            state = np.array(batch_memory[:,:self.l_state]).reshape((-1,self.l_state))# 0 : l_state
            obs = np.stack((state, state)).reshape((-1,self.l_state)) # None * l_state
            state_next = np.array(batch_memory[:, self.l_state:2*self.l_state]).reshape((-1,self.l_state)) #l_state:2*l_state
            obs_next = np.stack((state_next, state_next)).reshape((-1,self.l_state))        



        # Train critic
        feed = {self.obs : obs_next, self.actions : action_reshaped,
                self.single_actions : ai_single_actions_reshaped}
        V_target_res, V_next_res = self.sess.run([self.V_target, self.V], feed_dict=feed)
        V_target_res = np.squeeze(V_target_res)
        V_next_res = np.squeeze(V_next_res)
        
#         done_multiplier = np.squeeze(np.stack((done_vec, done_vec)).reshape((-1,1)))
#         print("reward shape:{}".format(reward.shape))
#         print("done shape:{}".format(done_multiplier.shape))
#         print("V_target_res shape:{}".format(V_target_res.shape))
        V_td_target = rewards + self.gamma * V_target_res * done_multiplier
#         print("V_td_target shape:{}".format(V_td_target.shape))
        feed = {self.V_td_target : V_td_target,
                self.obs : obs, 
                self.actions : action_reshaped,
                self.single_actions : ai_single_actions_reshaped}
#         if i % 100 == 0:
        _, V_res, loss_Q = self.sess.run([self.V_op, self.V, self.loss_V], feed_dict=feed)
        wandb.log({"SingleQ_loss": loss_Q})    
        
         
        # Train actor
        V_res = np.squeeze(V_res)
        V_td_target = rewards + self.gamma * V_next_res * done_multiplier
        
        feed = {self.obs : obs,
                self.actions : action_reshaped, self.V_td_target : V_td_target,
                self.V_evaluated : V_res,
                self.single_actions : ai_single_actions_reshaped}
        
        action_selected1, action_selected2 = self.sess.run([self.actions_selected1, self.actions_selected2], feed_dict=feed)
#         _ = self.sess.run([self.policy_op], feed_dict=feed)
        _, l_policy1 = self.sess.run([self.policy_op1, self.policy_loss1], feed_dict=feed)
        wandb.log({"policy1_loss": l_policy1}) 
        _, l_policy2 = self.sess.run([self.policy_op2, self.policy_loss2], feed_dict=feed)
        wandb.log({"policy2_loss": l_policy2})         


        # Get Q_tot target value
        action_combine1 = np.array(action_selected1[:self.batch_size]).reshape((-1,1))
        action_combine2 = np.array(action_selected2[-self.batch_size:]).reshape((-1,1))
        action_selected = np.concatenate((action_combine1, action_combine2), axis=0)
        

        feed = {self.state : state_next,
                self.actions : action_selected,
                self.obs : obs_next,
                self.single_actions : ai_single_actions_reshaped}


        Q_tot_target = self.sess.run(self.mixer_target, feed_dict=feed)

        target = reward + self.gamma * np.squeeze(Q_tot_target) * done_vec

        feed = {self.state : state,
                self.actions : action_reshaped,
                self.obs : obs,
                self.td_target : target,
                self.V_td_target : V_td_target,
                self.single_actions : ai_single_actions_reshaped}

        _, self.cost= self.sess.run([self.mixer_op,self.loss_mixer], feed_dict=feed)
        self.cost_his.append(self.cost)
        wandb.log({"Qmix_loss": self.cost}) 

        self.learn_step_counter += 1 
        
#         if i % 100 == 0:        
#             summary, _ = self.sess.run([self.summary_op_Q,self.mixer_op], feed_dict=feed)
#             writer.add_summary(summary, i) 


    def run_phys_Q_continuous(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None, iv_only = None, vasso_only = None):
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
        actions_reshaped = actions.reshape(-1,1)

        iv_only = np.array(iv_only).reshape((-1,1))
        vasso_only = np.array(vasso_only).reshape((-1,1))
        ai_single_actions = np.concatenate((vasso_only, iv_only), axis = 1)
        ai_single_actions_reshaped = ai_single_actions.reshape(-1,1)
 
        feed = {self.state : state,
                self.actions : actions_reshaped,
                self.obs : obs,
                self.single_actions : ai_single_actions_reshaped}
        phys_Qmix = sess.run(self.mixer, feed_dict=feed)
        return phys_Qmix
    
    def run_RL_Q_continuous(self, sess, list_state=None, list_obs=None, a_0=None, a_1=None, iv_only = None, vasso_only = None):
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
        actions_reshaped = actions.reshape(-1,1)
        
        iv_only = np.array(iv_only).reshape((-1,1))
        vasso_only = np.array(vasso_only).reshape((-1,1))
        ai_single_actions = np.concatenate((vasso_only, iv_only), axis = 1)
        ai_single_actions_reshaped = ai_single_actions.reshape(-1,1)



        feed = {self.state : state,
                self.actions : actions_reshaped,
                self.obs : obs,
                self.single_actions : ai_single_actions_reshaped}
        
        RL_Qmix = sess.run(self.mixer, feed_dict=feed)
        return RL_Qmix
    
    
    
class MasterDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            hidden_factor,
            input_dim,
            learning_rate=0.001,
            reward_decay=0.99,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
            sess=None,
            REWARD_THRESHOLD = 20,
            reg_lambda = 5,
            pretrain = True
    ):
        self.sess = sess
        self.REWARD_THRESHOLD = REWARD_THRESHOLD
        self.reg_lambda = reg_lambda
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_factor = hidden_factor
        self.input_dim = input_dim
        self.lr = setting.lr_Q
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = setting.replace_target_iter
        self.memory_size = memory_size
        self.memory_col_num = 2*self.n_features+4*self.hidden_factor + 7
        self.memory = np.zeros((self.memory_size, self.memory_col_num))        
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0

        
        self.pretrain = pretrain

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        if output_graph:
            self.writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):

                     
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.input_dim, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.matmul(s, w1) + b1
                l1 = tf.maximum(l1, l1*0.5)


            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)

            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], name='s')  # input
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval= build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('master_loss'):
            self.reg_vector = tf.maximum(tf.abs(self.q_eval)-self.REWARD_THRESHOLD,0)
            self.reg_term = tf.reduce_sum(self.reg_vector)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+ self.reg_lambda*self.reg_term
        with tf.variable_scope('train'):
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.input_dim], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next_tmp = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)
            self.done = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='done')
            self.q_next = tf.multiply(self.q_next_tmp, self.done)

    
    def store_transition(self, memory_array):
        self.memory = memory_array

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    def choose_low_level_Q(self, master_action, Q_no, Q_iv, Q_vasso, Q_mix):
        if master_action == 0:
            Q_return = Q_no
        elif master_action == 1:
            Q_return = Q_iv
        elif master_action == 2:
            Q_return=Q_vasso
        else:
            Q_return = Q_mix
        return Q_return
    def learn(self, i, use_FM, pretrain = True):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, 2*self.n_features].astype(int)
        reward = batch_memory[:, 2*self.n_features + 1]
        
        actions = batch_memory[:, 2*self.n_features] # None * 1
        action_reshaped = actions.reshape((-1,1))
        reward = batch_memory[:, 2*self.n_features + 1] #None * 1
        rewards = np.squeeze(reward.reshape((-1,1))) #None * 1
        done_vec = np.tile((1-batch_memory[:, 2*self.n_features + 2]).reshape(-1, 1), self.n_actions)       
        if (use_FM>0):
            states = np.array(batch_memory[:,-4*self.hidden_factor:-2*self.hidden_factor]).reshape((-1,2*self.hidden_factor)) #-4K: -2K            
            next_states = np.array(batch_memory[:,-2*self.hidden_factor:-self.hidden_factor]).reshape((-1,self.hidden_factor)) # -2K:-K
           
        else:
            states = np.array(batch_memory[:,:self.n_features]).reshape((-1, self.n_features)) # 0 : l_state
            next_states = np.array(batch_memory[:, self.n_features:2*self.n_features]).reshape((-1, self.n_features)) #l_state:2*l_state
           


        
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: next_states, self.done: done_vec}) # next observation
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: states})


        q_target = q_eval.copy()




        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
 
        

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: states,
                                                self.q_target: q_target})

        self.cost_his.append(self.cost)
        wandb.log({"Master_Q_loss": self.cost})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1        
        
#         if i % 100 == 0:
#             tf.compat.v1.summary.scalar('master_loss', self.loss)
#             merged_summary = tf.compat.v1.summary.merge_all() 
#             sum_summary = self.sess.run(merged_summary, feed_dict={self.s_: next_states, self.s: states, self.q_target: q_target})
#             self.writer.add_summary(sum_summary, i)    

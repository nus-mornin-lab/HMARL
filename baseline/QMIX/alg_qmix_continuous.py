"""Implementation of QMIX."""

import numpy as np
import tensorflow as tf
import networks_continuous as networks
import sys
import setting


class Qmix(object):


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
        self.memory = np.zeros((self.memory_size, self.l_state *2+2))

        self.n_agents = n_agents
        self.tau = 0.01
        self.lr_Q =setting.lr_Q
        self.lr_V = setting.lr_V
        self.lr_actor = setting.lr_actor
        self.gamma = 0.99

        self.agent_labels = np.eye(self.n_agents)
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
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_state], 'obs')
        self.actions = tf.placeholder(tf.float32, [None, 1], 'actions')
        self.single_actions = tf.placeholder(tf.float32, [None, 1], 'single_actions')

        # Individual agent networks
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy_main"):
            self.actions_selected, self.log_probs = networks.actor(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_action)
            
        with tf.variable_scope("V_main"):
            self.V = networks.critic_mixer(self.obs, self.actions, self.single_actions, self.nn['n_h1'], self.nn['n_h2'])
            
        with tf.variable_scope("V_target"):
            self.V_target = networks.critic_mixer(self.obs, self.actions, self.single_actions, self.nn['n_h1'], self.nn['n_h2'])        
        
        # To extract Q-value from V and V_target
        self.mixer_q_input = tf.reshape( self.V, [-1, self.n_agents] )
        self.mixer_target_q_input = tf.reshape( self.V_target, [-1, self.n_agents] )

        # Mixing network
        with tf.variable_scope("Mixer_main"):
            self.mixer = networks.Qmix_mixer(self.mixer_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])
        with tf.variable_scope("Mixer_target"):
            self.mixer_target = networks.Qmix_mixer(self.mixer_target_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])           
            
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
        
        self.actions_taken = tf.placeholder(tf.float32, [None, self.l_action], 'action_taken')
        # self.probs shape is [batch size * traj length, l_action]
        # now log_probs shape is [batch size * traj length]
#         log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.actions_taken), axis=1)+1e-15) 

        # Critic train op
        self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target')
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V)))
        self.V_opt = tf.compat.v1.train.AdamOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)
        # Policy train op
        self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
        self.V_td_error = self.V_td_target - self.V_evaluated
        self.policy_loss = -tf.reduce_mean( tf.multiply( self.log_probs, self.V_td_error ) )
        self.policy_opt = tf.compat.v1.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)
        
        # TD target calculated in train_step() using Mixer_target
        self.td_target = tf.placeholder(tf.float32, [None], 'td_target')
        self.loss_mixer = tf.reduce_mean(tf.square(self.td_target - tf.squeeze(self.mixer)))

        self.mixer_opt = tf.compat.v1.train.AdamOptimizer(self.lr_Q)
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
            summaries_Q.append(tf.summary.histogram(v.op.name, v))
        grads = self.mixer_opt.compute_gradients(self.loss_mixer, mixer_main_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_Q.append( tf.summary.histogram(var.op.name+'/gradient', grad) )

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
#         summaries_Q = [tf.summary.scalar('V_loss', self.loss_V)]
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
    


    def train_step_new(self, i, writer=None):
        if self.learn_step_counter % self.replace_target_iter == 0:
#             self.sess.run(self.replace_target_op)
            self.sess.run(self.list_update_target_ops)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

#         done_vec = np.tile((1-batch_memory[:, self.l_state + 2]).reshape(-1, 1), self.l_action)
        done_vec = (1-batch_memory[:, self.l_state + 5]).reshape(-1, 1)
        done_vec = np.squeeze(done_vec)
        
        
        # Each agent for each time step is now a batch entry

        n_steps = self.batch_size
        state = batch_memory[:, :self.l_state]
        obs = np.stack((state, state)).reshape((-1,self.l_state))

# ############################# discrete action############################# 
#         eval_act_index = batch_memory[:, self.l_state].astype(int)
#         act_index = eval_act_index.reshape((eval_act_index.shape[0], 1))
#         a_0 = (act_index/5).astype(int)
#         a_1 = (act_index%5).astype(int)
#         actions = np.concatenate((a_0, a_1), axis = 1)
# ############################# phys continuous action############################# 
        a_0 = self.l_state
        a_1 = self.l_state + 2
        phys_actions = batch_memory[:, a_0:a_1]
        action_reshaped = phys_actions.reshape((-1,1))
################################# ai single agent continuous action ########################
        iv_only = self.l_state + 2
        vasso_only = self.l_state + 3
        ai_iv_only = (batch_memory[:, iv_only]).reshape((-1,1))
        ai_vasso_only = (batch_memory[:, vasso_only]).reshape((-1,1))
#         print("ai_iv_only shape:{}".format(ai_iv_only.shape))
        ai_single_actions = np.concatenate((ai_vasso_only, ai_iv_only), axis = 1)
        ai_single_actions_reshaped = ai_single_actions.reshape((-1,1))

        reward = batch_memory[:, self.l_state + 4]
        rewards = np.squeeze(np.stack((reward, reward)).reshape((-1,1)))
        state_next = batch_memory[:, -self.l_state:]
        obs_next = np.stack((state_next, state_next)).reshape((-1,self.l_state))
        
#         print("action_reshaped:{}".format(action_reshaped.shape))
#         print("obs:{}".format(obs.shape))

        # Train critic
        feed = {self.obs : obs_next, self.actions : action_reshaped,
                self.single_actions : ai_single_actions_reshaped}
        V_target_res, V_next_res = self.sess.run([self.V_target, self.V], feed_dict=feed)
        V_target_res = np.squeeze(V_target_res)
        V_next_res = np.squeeze(V_next_res)
        done_multiplier = np.squeeze(np.stack((done_vec, done_vec)).reshape((-1,1)))
#         print("reward shape:{}".format(reward.shape))
#         print("done shape:{}".format(done_multiplier.shape))
#         print("V_target_res shape:{}".format(V_target_res.shape))
        V_td_target = rewards + self.gamma * V_target_res * done_multiplier
#         print("V_td_target shape:{}".format(V_td_target.shape))
        feed = {self.V_td_target : V_td_target,
                self.obs : obs, 
                self.actions : action_reshaped,
                self.single_actions : ai_single_actions_reshaped}
        
        _, V_res = self.sess.run([self.V_op, self.V], feed_dict=feed)
        
         
        # Train actor
        V_res = np.squeeze(V_res)
        V_td_target = rewards + self.gamma * V_next_res * done_multiplier
        
        feed = {self.epsilon : 0.5, self.obs : obs,
                self.actions : action_reshaped, self.V_td_target : V_td_target,
                self.V_evaluated : V_res,
                self.single_actions : ai_single_actions_reshaped}
        
        action_selected = self.sess.run([self.actions_selected], feed_dict=feed)
        if i % 100 == 0:        
            summary, _ = self.sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
            writer.add_summary(summary, i) 
            
#         # Get argmax actions from target networks
#         feed = {self.obs : obs_next}
#         action_selected = self.sess.run(self.action_selected, feed_dict=feed) # [batch*n_agents]

#         # Convert to 1-hot
#         actions_target_1hot = np.zeros([n_steps * self.n_agents, self.l_action], dtype=int)
#         actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1

        # Get Q_tot target value
        action_selected = np.squeeze(np.array(action_selected)).reshape(-1,1)
#         action_selected = np.squeeze(action_temp).reshape(-1,1)
        feed = {self.state : state_next,
                self.actions : action_selected,
                self.obs : obs_next,
                self.single_actions : ai_single_actions_reshaped}

#         print("state_next shape:{}".format(state_next.shape))
#         print("action_selected shape:{}".format(action_selected.shape))
#         print("obs_next shape:{}".format(obs_next.shape))
        Q_tot_target = self.sess.run(self.mixer_target, feed_dict=feed)
#         print("reward shape:{}".format(reward.shape))
#         print("np.squeeze(Q_tot_target) shape:{}".format(np.squeeze(Q_tot_target).shape))
#         print("done_vec shape:{}".format(done_vec.shape))
        target = reward + self.gamma * np.squeeze(Q_tot_target) * done_vec

        feed = {self.state : state,
                self.actions : action_reshaped,
                self.obs : obs,
                self.td_target : target,
                self.V_td_target : V_td_target,
                self.single_actions : ai_single_actions_reshaped}

        _, self.cost= self.sess.run([self.mixer_op,self.loss_mixer], feed_dict=feed)
        self.cost_his.append(self.cost)

#         self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.epsilon = tf.cond(tf.greater(self.epsilon, self.epsilon_max), lambda: self.epsilon_max, lambda: (self.epsilon + self.epsilon_increment))

        self.learn_step_counter += 1 
        
        if i % 100 == 0:        
            summary, _ = self.sess.run([self.summary_op_Q,self.mixer_op], feed_dict=feed)
            writer.add_summary(summary, i) 
            
        self.sess.run(self.list_update_target_ops)

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
            reg_lambda = 5
    ):
        self.sess = sess
        self.REWARD_THRESHOLD = REWARD_THRESHOLD
        self.reg_lambda = reg_lambda
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        if output_graph:
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.matmul(s, w1) + b1
#                 l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                l1 = tf.maximum(l1, l1*0.5)
#                 w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
#                 b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
#                 l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

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
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.reg_vector = tf.maximum(tf.abs(self.q_eval)-self.REWARD_THRESHOLD,0)
            self.reg_term = tf.reduce_sum(self.reg_vector)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+ self.reg_lambda*self.reg_term
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next_tmp = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)
            self.done = tf.placeholder(tf.float32, [None, self.n_actions], name='done')
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
    def choose_low_level_Q(master_action, Q_no, Q_iv, Q_vasso, Q_mix):
        if master_action == 0:
            Q_return = Q_no
        elif master_action == 1:
            Q_return = Q_iv
        elif master_action == 2:
            Q_return=Q_vasso
        else:
            Q_return = Q_mix
        return Q_return
    def learn(self, i):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        done_vec = np.tile((1-batch_memory[:, self.n_features + 2]).reshape(-1, 1), self.n_actions)
        # print(done_vec)
        # print(done_vec.shape)
        # print(batch_memory[:, self.n_features + 2].reshape(-1, 1))
        # print(batch_memory[:, self.n_features + 2].reshape(-1, 1).shape)
        
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:], self.done: done_vec}) # next observation
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})
        # action一一对应

        q_target = q_eval.copy()
        q2_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        low_level_Qs = batch_memory[:, (self.n_features + 3):(self.n_features + 6)]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        masterAct_Qs = np.concatenate([eval_act_index.reshape(-1,1), low_level_Qs], axis = 1)
        Q_return = pd.DataFrame(masterAct_Qs).apply(lambda x: choose_low_level_Q(x[0], x[1],x[1], x[2], x[3]), axis = 1)
        Q_return = np.array(Q_return)
        q2_target[batch_index, eval_act_index] = Q_return
        

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # print(i, self.cost, type(self.cost), type(np.nan), self.cost == np.nan)
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1        
        
        if i % 100 == 0:
            tf.compat.v1.summary.scalar('loss', self.loss)
            merged_summary = tf.summary.merge_all() 
            sum = self.sess.run(merged_summary, feed_dict={self.s_: batch_memory[:, -self.n_features:], self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
            self.writer.add_summary(sum, i)    


# class DuelingDQN:
#     def __init__(
#             self,
#             n_actions,
#             n_features,
#             learning_rate=0.001,
#             reward_decay=0.99,
#             e_greedy=0.9,
#             replace_target_iter=200,
#             memory_size=500,
#             batch_size=32,
#             e_greedy_increment=None,
#             output_graph=False,
#             dueling=True,
#             sess=None,
#             REWARD_THRESHOLD = 20,
#             reg_lambda = 5
#     ):
#         self.sess = sess
#         self.REWARD_THRESHOLD = REWARD_THRESHOLD
#         self.reg_lambda = reg_lambda
#         self.n_actions = n_actions
#         self.n_features = n_features
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon_max = e_greedy
#         self.replace_target_iter = replace_target_iter
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.epsilon_increment = e_greedy_increment
#         self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

#         self.dueling = dueling      # decide to use dueling DQN or not

#         self.learn_step_counter = 0
#         self.memory = np.zeros((self.memory_size, n_features*2+2))
#         self._build_net()
#         t_params = tf.get_collection('target_net_params')
#         e_params = tf.get_collection('eval_net_params')
#         self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]


#         if output_graph:
#             self.writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
#         self.cost_his = []

#     def _build_net(self):
#         def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
#             with tf.variable_scope('l1'):
#                 w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
#                 b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
#                 l1 = tf.matmul(s, w1) + b1
# #                 l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
#                 l1 = tf.maximum(l1, l1*0.5)
# #                 w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names)
# #                 b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
# #                 l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

#             if self.dueling:
#                 # Dueling DQN
#                 with tf.variable_scope('Value'):
#                     w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
#                     b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
#                     self.V = tf.matmul(l1, w2) + b2

#                 with tf.variable_scope('Advantage'):
#                     w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
#                     b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                     self.A = tf.matmul(l1, w2) + b2

#                 with tf.variable_scope('Q'):
#                     out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
#             else:
#                 with tf.variable_scope('Q'):
#                     w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
#                     b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                     out = tf.matmul(l1, w2) + b2

#             return out

#         # ------------------ build evaluate_net ------------------
#         self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
#         self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
#         with tf.variable_scope('eval_net'):
#             c_names, n_l1, w_initializer, b_initializer = \
#                 ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
#                 tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

#             self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

#         with tf.variable_scope('loss'):
#             self.reg_vector = tf.maximum(tf.abs(self.q_eval)-self.REWARD_THRESHOLD,0)
#             self.reg_term = tf.reduce_sum(self.reg_vector)
#             self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))+ self.reg_lambda*self.reg_term
#         with tf.variable_scope('train'):
#             self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

#         # ------------------ build target_net ------------------
#         self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
#         with tf.variable_scope('target_net'):
#             c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

#             self.q_next_tmp = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)
#             self.done = tf.placeholder(tf.float32, [None, self.n_actions], name='done')
#             self.q_next = tf.multiply(self.q_next_tmp, self.done)

    
#     def store_transition(self, memory_array):
#         self.memory = memory_array

#     def choose_action(self, observation):
#         observation = observation[np.newaxis, :]
#         if np.random.uniform() < self.epsilon:  # choosing action
#             actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
#             action = np.argmax(actions_value)
#         else:
#             action = np.random.randint(0, self.n_actions)
#         return action

#     def choose_low_level_Q(master_action, Q_no, Q_iv, Q_vasso, Q_mix):
#         if master_action == 0:
#             Q_return = Q_no
#         elif master_action == 1:
#             Q_return = Q_iv
#         elif master_action == 2:
#             Q_return=Q_vasso
#         else:
#             Q_return = Q_mix
#         return Q_return
    
#     def learn(self, i):
#         if self.learn_step_counter % self.replace_target_iter == 0:
#             self.sess.run(self.replace_target_op)
#             # print('\ntarget_params_replaced\n')

#         sample_index = np.random.choice(self.memory_size, size=self.batch_size)
#         batch_memory = self.memory[sample_index, :]

#         done_vec = np.tile((1-batch_memory[:, self.n_features + 2]).reshape(-1, 1), self.n_actions)
#         # print(done_vec)
#         # print(done_vec.shape)
#         # print(batch_memory[:, self.n_features + 2].reshape(-1, 1))
#         # print(batch_memory[:, self.n_features + 2].reshape(-1, 1).shape)
        
#         q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:], self.done: done_vec}) # next observation
#         q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})
#         # action一一对应

#         q_target = q_eval.copy()
# #         q2_target = q_eval.copy()

#         batch_index = np.arange(self.batch_size, dtype=np.int32)
#         eval_act_index = batch_memory[:, self.n_features].astype(int)
#         reward = batch_memory[:, self.n_features + 1]

#         q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
#         # apply the double q-trick
# #         low_level_Qs = batch_memory[:, (self.n_features + 2):(self.n_features + 5)]
# #         q_target[batch_index, eval_act_index] = low_level_Qs.apply(lambda x: choose_low_level_Q(x['iv_fluids_quantile'], x['vasopressors_quantile']), axis=1)
# #         min_q_target = tf.minimum(q_target, q2_target)

#         _, self.cost = self.sess.run([self._train_op, self.loss],
#                                      feed_dict={self.s: batch_memory[:, :self.n_features],
#                                                 self.q_target: q_target})
#         # print(i, self.cost, type(self.cost), type(np.nan), self.cost == np.nan)
#         self.cost_his.append(self.cost)

#         self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
#         self.learn_step_counter += 1        
        
#         if i % 100 == 0:
#             tf.compat.v1.summary.scalar('loss', self.loss)
#             merged_summary = tf.compat.v1.summary.merge_all() 
#             sum = self.sess.run(merged_summary, feed_dict={self.s_: batch_memory[:, -self.n_features:], self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
#             self.writer.add_summary(sum, i)
            
            
class Single_AC(object):


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
        self.memory = np.zeros((self.memory_size, self.l_state *2+2))

        self.n_agents = n_agents
        self.tau = 0.01
        self.lr_Q =setting.lr_Q
        self.lr_V = setting.lr_V
        self.lr_actor = setting.lr_actor
        self.gamma = 0.99

        self.agent_labels = np.eye(self.n_agents)
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
        self.state = tf.placeholder(tf.float32, [None, self.l_state], 'state')
        self.obs = tf.placeholder(tf.float32, [None, self.l_state], 'obs')
        self.actions = tf.placeholder(tf.float32, [None, 1], 'actions')

        # Individual agent networks
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        with tf.variable_scope("Policy_main"):
            self.actions_selected, self.log_probs = networks.actor(self.obs, self.nn['n_h1'], self.nn['n_h2'], self.l_action)
            
        with tf.variable_scope("V_main"):
            self.V = networks.critic(self.obs, self.actions, self.nn['n_h1'], self.nn['n_h2'])
            
        with tf.variable_scope("V_target"):
            self.V_target = networks.critic(self.obs, self.actions, self.nn['n_h1'], self.nn['n_h2'])        
        
        # To extract Q-value from V and V_target
        self.mixer_q_input = tf.reshape( self.V, [-1, self.n_agents] )
        self.mixer_target_q_input = tf.reshape( self.V_target, [-1, self.n_agents] )

#         # Mixing network
#         with tf.variable_scope("Mixer_main"):
#             self.mixer = networks.Qmix_mixer(self.mixer_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])
#         with tf.variable_scope("Mixer_target"):
#             self.mixer_target = networks.Qmix_mixer(self.mixer_target_q_input, self.state, self.l_state, self.n_agents, self.nn['n_h_mixer'])           
            
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

#         # mixer network 
#         list_Mixer_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_main')
#         map_name_Mixer_main = {v.name.split('main')[1] : v for v in list_Mixer_main}
#         list_Mixer_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Mixer_target')
#         map_name_Mixer_target = {v.name.split('target')[1] : v for v in list_Mixer_target}
        
#         if len(list_Mixer_main) != len(list_Mixer_target):
#             raise ValueError("get_initialize_target_ops : lengths of Mixer_main and Mixer_target do not match")
        
#         # ops for equating main and target
#         for name, var in map_name_Mixer_main.items():
#             # create op that assigns value of main variable to
#             # target variable of the same name
#             list_initial_ops.append( map_name_Mixer_target[name].assign(var) )
        
#         # ops for slow update of target toward main
#         for name, var in map_name_Mixer_main.items():
#             # incremental update of target towards main
#             list_update_ops.append( map_name_Mixer_target[name].assign( self.tau*var + (1-self.tau)*map_name_Mixer_target[name] ) )

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
    


    def train_step_single_AC(self, i, writer=None):
        if self.learn_step_counter % self.replace_target_iter == 0:
#             self.sess.run(self.replace_target_op)
            self.sess.run(self.list_update_target_ops)

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        done_vec = (1-batch_memory[:, self.l_state + 2]).reshape(-1, 1)
        done_vec = np.squeeze(done_vec)
        
        
        # Each agent for each time step is now a batch entry

        n_steps = self.batch_size
        state = batch_memory[:, :self.l_state]
        obs = state.reshape((-1,self.l_state))

# ############################# discrete action############################# 
#         eval_act_index = batch_memory[:, self.l_state].astype(int)
#         act_index = eval_act_index.reshape((eval_act_index.shape[0], 1))
#         a_0 = (act_index/5).astype(int)
#         a_1 = (act_index%5).astype(int)
#         actions = np.concatenate((a_0, a_1), axis = 1)
# ############################# continuous action############################# 
        a_0 = self.l_state
        actions = batch_memory[:, a_0]
##############################################################################

        

        action_reshaped = actions.reshape((-1,1))

        reward = batch_memory[:, self.l_state + 1]
        rewards = np.squeeze(reward.reshape((-1,1)))
        state_next = batch_memory[:, -self.l_state:]
        obs_next = state_next.reshape((-1,self.l_state))
        
#         print("action_reshaped:{}".format(action_reshaped.shape))
#         print("obs:{}".format(obs.shape))

        # Train critic
        feed = {self.obs : obs_next, self.actions : action_reshaped}
        V_target_res, V_next_res = self.sess.run([self.V_target, self.V], feed_dict=feed)
        V_target_res = np.squeeze(V_target_res)
        V_next_res = np.squeeze(V_next_res)
        done_multiplier = np.squeeze(done_vec.reshape((-1,1)))
#         print("reward shape:{}".format(reward.shape))
#         print("done shape:{}".format(done_multiplier.shape))
#         print("V_target_res shape:{}".format(V_target_res.shape))
        V_td_target = rewards + self.gamma * V_target_res * done_multiplier
#         print("V_td_target shape:{}".format(V_td_target.shape))
        feed = {self.V_td_target : V_td_target,
                self.obs : obs, 
                self.actions : action_reshaped}
        
        _, V_res, self.cost = self.sess.run([self.V_op, self.V, self.loss_V], feed_dict=feed)
        
        self.cost_his.append(self.cost)
        
         
        # Train actor
        V_res = np.squeeze(V_res)
        V_td_target = rewards + self.gamma * V_next_res * done_multiplier
        
        feed = {self.epsilon : 0.5, self.obs : obs,
                self.actions : action_reshaped, self.V_td_target : V_td_target,
                self.V_evaluated : V_res}
        
        action_selected = self.sess.run([self.actions_selected], feed_dict=feed)
        if i % 100 == 0:        
            summary, _ = self.sess.run([self.summary_op_policy, self.policy_op], feed_dict=feed)
            writer.add_summary(summary, i) 
            
#         # Get argmax actions from target networks
#         feed = {self.obs : obs_next}
#         action_selected = self.sess.run(self.action_selected, feed_dict=feed) # [batch*n_agents]

#         # Convert to 1-hot
#         actions_target_1hot = np.zeros([n_steps * self.n_agents, self.l_action], dtype=int)
#         actions_target_1hot[np.arange(n_steps*self.n_agents), argmax_actions] = 1

#         # Get Q_tot target value
#         action_selected = np.squeeze(np.array(action_selected)).reshape(-1,1)
# #         action_selected = np.squeeze(action_temp).reshape(-1,1)
#         feed = {self.state : state_next,
#                 self.actions : action_selected,
#                 self.obs : obs_next}

# #         print("state_next shape:{}".format(state_next.shape))
# #         print("action_selected shape:{}".format(action_selected.shape))
# #         print("obs_next shape:{}".format(obs_next.shape))
#         Q_tot_target = self.sess.run(self.mixer_target, feed_dict=feed)
# #         print("reward shape:{}".format(reward.shape))
# #         print("np.squeeze(Q_tot_target) shape:{}".format(np.squeeze(Q_tot_target).shape))
# #         print("done_vec shape:{}".format(done_vec.shape))
#         target = reward + self.gamma * np.squeeze(Q_tot_target) * done_vec

#         feed = {self.state : state,
#                 self.actions : action_reshaped,
#                 self.obs : obs,
#                 self.td_target : target,
#                 self.V_td_target : V_td_target}

#         _, self.cost= self.sess.run([self.mixer_op,self.loss_mixer], feed_dict=feed)
#         self.cost_his.append(self.cost)

#         self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.epsilon = tf.cond(tf.greater(self.epsilon, self.epsilon_max), lambda: self.epsilon_max, lambda: (self.epsilon + self.epsilon_increment))

        self.learn_step_counter += 1 
        
#         if i % 100 == 0:        
#             summary, _ = self.sess.run([self.summary_op_Q,self.mixer_op], feed_dict=feed)
#             writer.add_summary(summary, i) 
            
        self.sess.run(self.list_update_target_ops)

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

 
        feed = {self.state : state,
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
#         a_1 = np.array(a_1).reshape((-1,1))
        
#         actions = np.concatenate((a_0, a_1), axis = 1)
        actions_reshaped = a_0
#         actions = actions.astype(int)

#         actions_1hot = self.process_actions(n_steps, actions)
        feed = {self.state : state,
                self.actions : actions_reshaped,
                self.obs : obs}
        RL_Q = sess.run(self.V, feed_dict=feed)
        return RL_Q
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



def get_variable(name, shape):

    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0,0.01))


def actor(obs, n_h1, n_h2, n_actions):
    """
    Args:
        obs: TF placeholder
        n_h1: int
        n_h2: int
        n_actions: int
    """

    h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')

    return probs


def critic(obs, n_h1, n_h2,n_actions):

    """
    Args:
        obs: np.array
        n_h1: int
        n_h2: int
        n_actions: int
    """   
    h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu6, use_bias=True, name='V_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu6, use_bias=True, name='V_h2')
#     h3 = tf.layers.dense(inputs=h2, units=n_h2, activation=tf.nn.relu, use_bias=True, name='V_h3')
    q = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='V_out')
#     out = tf.clip_by_value(q, -100.0, 100.0 )
#     tmp = tf.keras.activations.tanh(q)
#     out = tf.multiply(tmp, c2)
    return q

def critic_mixer(obs, single_agent_actions,n_h1, n_h2,n_actions):

    """
    Args:
        obs: np.array
        single_agent_actions: 1-hot np.array
        n_h1: int
        n_h2: int
        n_actions: int
    """
#     c2 = tf.constant(15, dtype=tf.float32, shape=None, name='c2')
    state_action_single = tf.concat([obs, tf.cast(single_agent_actions, dtype=tf.float32)], axis=1)
    h1 = tf.layers.dense(inputs=state_action_single, units=n_h1, activation=tf.nn.relu6, use_bias=True, name='V_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu6, use_bias=True, name='V_h2')
    q = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='V_out')

    return q

def Qmix_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state: shape [batch, state_dim]
        state_dim: integer
        n_agents: integer
        n_h_mixer: integer
    """
    agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])

    # n_h_mixer * n_agents because result will be reshaped into matrix
    hyper_w_1 = get_variable('hyper_w_1', [state_dim, n_h_mixer*n_agents]) 
    hyper_w_final = get_variable('hyper_w_final', [state_dim, n_h_mixer])

    hyper_b_1 = tf.get_variable('hyper_b_1', [state_dim, n_h_mixer])

    hyper_b_final_l1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=tf.nn.relu,
                                       use_bias=False, name='hyper_b_final_l1')
    hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                    use_bias=False, name='hyper_b_final')

    # First layer
    w1 = tf.abs(tf.matmul(state, hyper_w_1))
    b1 = tf.matmul(state, hyper_b_1)
    w1_reshaped = tf.reshape(w1, [-1, n_agents, n_h_mixer]) # reshape into batch of matrices
    b1_reshaped = tf.reshape(b1, [-1, 1, n_h_mixer])
    # [batch, 1, n_h_mixer]
    hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
    
    # Second layer
    w_final = tf.abs(tf.matmul(state, hyper_w_final))
    w_final_reshaped = tf.reshape(w_final, [-1, n_h_mixer, 1]) # reshape into batch of matrices
    b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])

    # [batch, 1, 1]
    y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped

    q_tot = tf.reshape(y, [-1, 1])
    out = tf.clip_by_value(q_tot, -100.0, 100.0 )
    return out
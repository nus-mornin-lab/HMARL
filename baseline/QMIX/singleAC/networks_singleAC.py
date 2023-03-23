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
        actions: TF placeholder
        n_h1: int
        n_h2: int
        n_actions: int
    """
    c1 = tf.constant(0.5, dtype=tf.float32, shape=None, name='c1')
    h1 = tf.layers.dense(inputs=obs, units=n_h1, activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='actor_h2')
    mu = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_mean')
    log_sigma = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_std')
    # probs = tf.nn.softmax(out, name='actor_softmax')
    sigma = tf.exp(log_sigma)
    dist = tfp.distributions.Normal(mu, sigma)
    # TODO add tanh squashing here.
    action_ = dist.sample()
    tmp = tf.tanh(action_)
    action = tf.multiply(tmp, c1)
    # Calculate the log probability
    log_pi_ = dist.log_prob(action_)
    # Change log probability to account for tanh squashing as mentioned in
    # Appendix C of the paper
    EPSILON = 1e-16
    log_pi = log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 + EPSILON), axis=1,keepdims=True)
        # return action, log_pi

    return action, log_pi


def critic(obs, actions,n_h1, n_h2):

    """
    Args:
        obs: np.array
        role: 1-hot np.array
        n_h1: int
        n_h2: int
    """
    c2 = tf.constant(15, dtype=tf.float32, shape=None, name='c1')
    state_action = tf.concat([obs, tf.cast(actions, dtype=tf.float32)], axis=1)
    h1 = tf.layers.dense(inputs=state_action, units=n_h1, activation=tf.nn.relu, use_bias=True, name='V_h1')
    h2 = tf.layers.dense(inputs=h1, units=n_h2, activation=tf.nn.relu, use_bias=True, name='V_h2')
    q = tf.layers.dense(inputs=h2, units=1, activation=None, use_bias=True, name='V_out')
    tmp = tf.keras.activations.tanh(q)
    out = tf.multiply(tmp, c2)
    return out


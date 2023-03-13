import random
from tensorflow.keras.layers import Dense, Activation, Input, Dropout, Flatten
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys
from scipy.stats import entropy
import tensorflow_probability as tfp
import math
from tensorflow.python.util import nest
import time
import gym

EPS = 1e-6

LOG_STD_MAX = 2
LOG_STD_MIN = -20

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    print(e)

tfd = tfp.distributions

class ReplayBuffer(object):
    def __init__(self, mem_size, batch_size, obs_dim, n_actions, total_time_steps=20_000_000):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.state_memory = np.zeros([mem_size, obs_dim], dtype=np.float32)
        self.action_memory = np.zeros([mem_size, n_actions], dtype=np.float32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros([mem_size, obs_dim], dtype=np.float32)
        self.terminal_memory = np.zeros(mem_size, dtype=np.float32)
        self.obs_dim = obs_dim
        self.ptr, self.size = 0, 0

    def store(self, state, action, reward, new_state, done):

        self.state_memory[self.ptr] = state

        self.action_memory[self.ptr] = action

        self.reward_memory[self.ptr] = reward

        self.new_state_memory[self.ptr] = new_state

        self.terminal_memory[self.ptr] = done

        self.size = min(self.size+1, self.mem_size)
        self.ptr = (self.ptr+1) % self.mem_size


    def get_data(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        
        state = self.state_memory[idxs]

        action = self.action_memory[idxs]

        reward = self.reward_memory[idxs]

        new_state = self.new_state_memory[idxs]

        dones = self.terminal_memory[idxs]


        return state, action, reward, new_state, dones



    def is_train(self):
        if len(self.state_memory) < self.batch_size:
            return False
        else:
            return True

def apply_squashing_func(mu_, pi_, logp_pi):
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)

    logp_pi -= tf.math.reduce_sum(tf.math.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi


def swish(x):
    """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
    return tf.math.multiply(x, tf.math.sigmoid(x))

class distributions():
    def __init__(self):
        self.distributions = None


    def make_distributions(self, mean, log_std):
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_std = tf.exp(log_std)

        self.distribution = tfp.distributions.Normal(mean, action_std)

        act = self.distribution.sample()

        return act

    def log_prob(self, action):
        log_prob = self.distribution.log_prob(action)

        if len(log_prob.shape) > 1:
            log_prob = tf.reduce_sum(log_prob, axis=1)
        else:
            log_prob = tf.reduce_sum(log_prob)

        return log_prob

    def entropy(self):
        return self.distribution.entropy()

class Actor(Model):
    def __init__(self,acion_shape):
        super().__init__()
        self.distribution = distributions()
        self.action_dim = acion_shape
        self.d1 = Dense(256, activation='tanh')
        self.d2 = Dense(256, activation='tanh')
        self.mu_layer = Dense(acion_shape)
        self.log_std_layer = Dense(acion_shape)

    def call(self, state):

        a1 = self.d1(state)
        a2 = self.d2(a1)

        mu_ = self.mu_layer(a2)

        self.log_std = log_std = self.log_std_layer(a2)

        pi_ = self.distribution.make_distributions(mu_, log_std)

        logp_pi = self.distribution.log_prob(pi_)

        self.entropy = self.distribution.entropy()

        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)

        return deterministic_policy, policy, logp_pi

class Q_function(Model):
    def __init__(self):
        super().__init__()

        self.d1 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.d2 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.output_layer = Dense(1)

        self.d3 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.d4 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.sec_output_layer = Dense(1)

    def call(self, state, action):

        qf_h = tf.concat([state, action], axis=1)

        a1 = self.d1(qf_h)
        a2 = self.d2(a1)
        q1 = self.output_layer(a2)

        a3 = self.d3(qf_h)
        a4 = self.d4(a3)
        q2 = self.sec_output_layer(a4)

        return q1, q2


    @property
    def trainable_variables(self):
        return self.d1.trainable_variables + \
               self.d2.trainable_variables + \
               self.output_layer.trainable_variables + \
               self.d3.trainable_variables + \
               self.d4.trainable_variables + \
               self.sec_output_layer.trainable_variables



class Value_Function(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.d2 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.output_layer = Dense(1)

    def call(self, state):

        a1 = self.d1(state)
        a2 = self.d2(a1)
        output = self.output_layer(a2)

        return output


class Target_Value_Function(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.d2 = Dense(256, activation=swish,kernel_initializer=tf.keras.initializers.VarianceScaling(1.0))
        self.output_layer = Dense(1)

    def call(self, state):
        a1 = self.d1(state)
        a2 = self.d2(a1)
        output = self.output_layer(a2)

        return output





class Agent(object):
    def __init__(self, state_shape, n_actions,scale_reward=1,random_exploration=0.0, polyak=0.005, epochs=1, batch_size=64, lr=0.0003, gamma=0.99, mem_size=20_000, GAE=1.0, update_every=1, epsilon=1e-6, start_train=1000,seed=None):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.state_shape = state_shape
        self.n_actions = int(n_actions)
        self.polyak = polyak
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.memory = ReplayBuffer(mem_size=mem_size,batch_size=batch_size, obs_dim=int(state_shape), n_actions=int(n_actions))
        self.GAE = GAE
        self.update_every = update_every
        self.epsilon = epsilon
        self.input_shape = int(state_shape + n_actions)
        self.scale_reward = scale_reward
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.EPS = 1e-8

        self.start_train = start_train
        self.if_start = False

        self.target_entropy = self._get_default_target_entropy()

        self.log_ent_coef = 1.0

        self.log_ent_coef = np.log(self.log_ent_coef)

        self.log_ent_coef = self.get_variable(self.log_ent_coef, tf.float32)

        #self.GAE = 0.8172885

        self.update = 0

        self.polyak = 0.005

        self.random_exploration = random_exploration

        self.lr = 0.0003

        self.ent_coef = tf.exp(self.log_ent_coef).numpy()

        print(self.ent_coef)

        self.start_count = 0

        self.policy_opt = Adam(lr=self.lr)

        self.q1_opt = Adam(lr=self.lr)

        self.q2_opt = Adam(lr=self.lr)

        self.value_opt = Adam(lr=self.lr)

        self.ent_opt = Adam(lr=self.lr)

        self.policy = Actor(n_actions)

        self.q_function = Q_function()

        self.value_function = Value_Function()

        self.target_value_function = Target_Value_Function()


    def _get_default_target_entropy(self):

        target_entropy = -np.prod(self.n_actions).astype(np.float32)

        return target_entropy


    def get_variable(self,init_verb,dtype):
        verb = tf.convert_to_tensor(init_verb, dtype=dtype)

        verb = tf.Variable(verb, dtype=dtype)

        return verb

    def select_action(self, state):
        state = state[np.newaxis,:]

        if self.if_start and np.random.random() > self.random_exploration:

            mu, pi, log_pi = self.policy(state)
            mu, pi, log_pi = mu[0], pi[0], log_pi[0]
            action = pi.numpy()

        else:

           action = np.random.randn(self.n_actions)

        return action

    def store_data(self, state, action, reward, new_state, done):
        if not self.if_start:
        	state = state[np.newaxis,:]

        self.memory.store(state, action, reward, new_state, float(done))


    def vf_loss(self, state):

        mu, pi, log_pi = self.policy(state)

        q1_pi, q2_pi = self.q_function(state, pi)

        min_qf_pi = tf.minimum(q1_pi, q2_pi)

        v_backup =  tf.stop_gradient(min_qf_pi - self.ent_coef*log_pi)

        vf = self.value_function(state)

        valuef_loss = 0.5 * tf.keras.losses.MSE(vf,v_backup)

        return valuef_loss


    def critic_loss(self, state, new_state, reward, done, action, min_qf_pi, log_pi):


        v_backup =  tf.stop_gradient(min_qf_pi - self.ent_coef*log_pi)

        vf = self.value_function(state)

        valuef_loss = 0.5 * tf.reduce_mean((vf - v_backup) ** 2)

        target_value_function_pred_new_state = self.target_value_function(new_state)

        q_backup = tf.stop_gradient( self.scale_reward*reward +  (1-done)*self.gamma*target_value_function_pred_new_state )

        q1, q2  = self.q_function(state, action)

        qf1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)

        qf2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)

        self.qf1_loss = qf1_loss

        self.qf2_loss =  qf1_loss

        self.valuef_loss = valuef_loss

        critic_loss = qf1_loss + qf2_loss + valuef_loss

        return critic_loss


    def policy_loss(self, state):

        mu, pi, log_pi = self.policy(state)

        q1_pi, q2_pi = self.q_function(state, pi)

        min_qf_pi = tf.minimum(q1_pi, q2_pi)

        vf = self.value_function(state)

        policy_loss = tf.reduce_mean(self.ent_coef*log_pi - q1_pi)

        self.entropy = tf.reduce_mean(self.policy.entropy)

        policy_loss = tf.reduce_mean(policy_loss)

        self.policy_lossasd = policy_loss

        return policy_loss, min_qf_pi, log_pi


    def train_critic(self, state, action, done, reward, new_state):

        trainable_critic_variables = (self.q_function.trainable_variables +
                                      self.value_function.trainable_variables)

        with tf.GradientTape(watch_accessed_variables=False) as critic_tape:

            critic_tape.watch(trainable_critic_variables)

            critic_loss = self.critic_loss( state, new_state, reward, done, action )


        critic_grads = critic_tape.gradient(critic_loss, trainable_critic_variables)

        self.q1_opt.apply_gradients(zip(critic_grads, trainable_critic_variables))


    def train_actor(self, state):

        trainable_policy_variables = self.policy.trainable_variables

        with tf.GradientTape(watch_accessed_variables=False) as policy_tape:
            policy_tape.watch(trainable_policy_variables)

            policy_loss = self.policy_loss(state)

        grads_policy = policy_tape.gradient(policy_loss, trainable_policy_variables)

        self.policy_opt.apply_gradients(zip(grads_policy, trainable_policy_variables))




    def train_all(self, state, action, done, reward, new_state):

        target_value_function_pred_new_state = self.target_value_function(new_state)

        trainable_critic_variables = (self.q_function.trainable_variables +
                                      self.value_function.trainable_variables)


        trainable_policy_variables = self.policy.trainable_variables

        ent_trianable = [self.log_ent_coef]

        with tf.GradientTape(watch_accessed_variables=False) as policy_tape:

            policy_tape.watch(trainable_policy_variables)

            policy_loss, min_qf_pi, log_pi = self.policy_loss(state)



        with tf.GradientTape(watch_accessed_variables=False) as critic_tape:

            critic_tape.watch(trainable_critic_variables)

            critic_loss = self.critic_loss( state, new_state, reward, done, action, min_qf_pi, log_pi )

        with tf.GradientTape(watch_accessed_variables=False) as ent_tape:

            ent_tape.watch(ent_trianable)

            mu, pi, log_pi = self.policy(state)

            ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(log_pi + self.target_entropy))

            self.ent_coef_loss = ent_coef_loss


                

        grads_policy = policy_tape.gradient(policy_loss, trainable_policy_variables)

        policy_endind = self.apply_grads(grads_policy, trainable_policy_variables, self.policy_opt)


        with tf.control_dependencies([policy_endind]):

            critic_grads = critic_tape.gradient(critic_loss, trainable_critic_variables)

            critic_endind = self.apply_grads(critic_grads, trainable_critic_variables, self.q1_opt)

            with tf.control_dependencies([critic_endind]):

                ent_grads = ent_tape.gradient(ent_coef_loss, ent_trianable)
                self.ent_opt.apply_gradients(zip(ent_grads, ent_trianable))



    def apply_grads(self, grads,trainable_variables, opt):

        fa = list(zip(grads, trainable_variables))

        opt.apply_gradients(fa)


    def load_model(self):
        self.policy.load_weights('models/policy')


    def soft_update(self):
        target_wieghts = []
        for target, source in zip(self.target_value_function.get_weights(), self.value_function.get_weights()):
            fa = (1 - self.polyak) * target + self.polyak * source

            target_wieghts.append(fa)

        self.target_value_function.set_weights(target_wieghts)


    def hard_update(self, state, action):
        self.target_value_function(state)
        self.value_function(state)

        self.target_value_function.set_weights(self.value_function.get_weights())


    def update_ent_coef(self):
        ent_coef = tf.exp(self.log_ent_coef).numpy()

        if not ent_coef == 0.0:
            self.ent_coef = ent_coef


    def train(self, episode, step):
        state, action, reward, new_state, dones = self.memory.get_data()

        if self.start_train < step:

            self.if_start = True
            if step % self.update_every == 0:
                self.start_count += 1

                if self.start_count == 1:
                    self.hard_update(state, action)


                state, action, reward, new_state, dones = np.array(state), np.array(action), np.array(reward), np.array(new_state), np.array(dones)

                self.train_all(state, action, dones, reward, new_state)
                self.soft_update()
                self.update_ent_coef()


                sf = state[0][np.newaxis,:]

                mu, pi, log_pi = self.policy(sf)

                self.log_std = self.policy.log_std

                self.std_mu = np.std(mu)

                self.mu = mu
